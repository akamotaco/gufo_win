#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/transfer_pool.hpp>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "continuation_adapter.hpp"
#include "engine.hpp"
#include "text_contract.hpp"
#include "vision/encoder.hpp"
#include "vision/prompt.hpp"

namespace gufo::models::gemma4 {
namespace {

using ext::TextRunnerToken;

// Each Gemma 4 request owns its KV cache (global rows plus the sliding-window
// ring) and its snapshots; forwards of different requests are serialized.
constexpr std::string_view kStateAbi = "gemma4-rocm-session-v1";

TokenId EngineToken(TextRunnerToken token) {
  if (token >
      static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("Gemma 4 token ID exceeds engine range");
  }
  return static_cast<TokenId>(token);
}

std::vector<TokenId> EngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<TokenId> converted;
  converted.reserve(tokens.size());
  for (const auto token : tokens) {
    converted.push_back(EngineToken(token));
  }
  return converted;
}

bool IsSha256Hex(std::string_view value) {
  return value.size() == 64 &&
         value.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

std::vector<std::uint8_t> CompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t draft_tokens,
    std::uint32_t ring) {
  if (!IsSha256Hex(artifact_fingerprint) ||
      (has_mtp && !IsSha256Hex(mtp_fingerprint))) {
    throw std::invalid_argument(
        "Gemma 4 disk cache requires artifact fingerprints");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=gemma4\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=" << ChatTemplate::TemplateId() << '\n'
           << "chat_template_reference_sha256="
           << ChatTemplate::UnslothTemplateSha256() << '\n'
           << "state_abi=" << kStateAbi << '\n'
           << "payload_layout=gemma4-rocm-session-snapshot-v"
           << Session::kSnapshotPayloadVersion << '\n'
           << "kv_layout=f16-token-major-global+sliding-ring-logical\n"
           << "sliding_ring_slots=" << ring << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n";
  if (has_mtp) {
    identity << "draft_backend=gemma4-assistant-mtp-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << mtp_fingerprint << '\n'
             << "draft_tokens=" << draft_tokens << '\n';
  }
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

class State final : public ext::TextRunnerState {
public:
  /// The session of a component-cache slot, owned by the adapter.
  State(Session& session, cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}
  void Invalidate() noexcept override { (void)adapter_->Invalidate(*slot_); }
  void SetCancellationCheck(const CancellationCheck&) override {}
  [[nodiscard]] ext::TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }
  [[nodiscard]] Session& session() const { return *session_; }

  /// The current request's images; their embeddings are encoded on first use.
  void SetImages(std::shared_ptr<const vision::Prompt> prompt) {
    if (prompt != images_) {
      images_ = std::move(prompt);
      embeddings_.clear();
    }
  }
  [[nodiscard]] const vision::Prompt* images() const { return images_.get(); }
  const float* Embedding(vision::Encoder& encoder, std::size_t index) {
    auto& embedding = embeddings_[index];
    if (!embedding) {
      embedding = encoder.Encode(images_->images.at(index).pixels);
    }
    return embedding->data();
  }

private:
  Session* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::shared_ptr<const vision::Prompt> images_;
  std::map<std::size_t, std::shared_ptr<const vision::Encoder::Embedding>>
      embeddings_;
};

State& Get(ext::TextRunnerState& state) {
  auto* gemma = dynamic_cast<State*>(&state);
  if (gemma == nullptr) {
    throw std::logic_error("text runner state is not Gemma 4");
  }
  return *gemma;
}

const State& Get(const ext::TextRunnerState& state) {
  const auto* gemma = dynamic_cast<const State*>(&state);
  if (gemma == nullptr) {
    throw std::logic_error("text runner state is not Gemma 4");
  }
  return *gemma;
}

hip::ComponentCacheHooks CacheHooks() {
  return {
      [] { return ext::HostSnapshotCeilingBytes(); },
      [](std::string_view message) { server::Logger::Warn("cache", message); },
      [](std::string_view message) { server::Logger::Info("cache", message); }};
}

class Runner final : public ext::HipTextModelRunner {
public:
  Runner(std::shared_ptr<Model> model, std::uint32_t max_context,
         std::uint32_t ring, const std::string& artifact_fingerprint,
         const std::string& mtp_fingerprint,
         std::shared_ptr<vision::Encoder> vision, std::uint32_t image_tokens)
      : model_(std::move(model)),
        vision_(std::move(vision)),
        image_tokens_(image_tokens),
        max_context_(max_context) {
    if (!artifact_fingerprint.empty()) {
      component_identity_ = CompatibilityIdentity(
          artifact_fingerprint, mtp_fingerprint, model_->HasMtp(), max_context_,
          model_->DraftTokens(), ring);
    } else {
      // Without artifact fingerprints checkpoints stay in this process.
      std::ostringstream identity;
      identity << "schema=gufo-process-text-continuation-v1\n"
               << "model_instance=" << model_.get() << '\n'
               << "context_tokens=" << max_context_ << '\n'
               << "mtp=" << model_->HasMtp() << '\n'
               << "draft_max_tokens=" << model_->DraftTokens() << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }

  // Checkpoint rows and private states live in committed host pages
  // (registered with the GPU, read by the disk tier), transferred on pooled
  // streams.
  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t) const override {
    auto adapter = std::make_unique<ContinuationAdapter>(model_, max_context_,
                                                         component_identity_);
    auto* gemma = adapter.get();
    auto resources = hip::CreateHipComponentCacheResources(
        std::move(adapter), ledger, CacheHooks());
    // Durable captures require this hook, but the adapter lends no rows, so
    // persistence never reads a slot through it. A copy still waits for the
    // forwards that wrote its rows.
    resources->copy_observed_rows = [gemma](auto source, auto destination,
                                            cache::Stream& stream) {
      auto& transfer = dynamic_cast<hip::TransferStream&>(stream);
      gemma->AfterExecution(transfer.Native());
      return transfer.Copy(
          destination.data(), source.data(), destination.size(),
          hip::CopyDirection::kDeviceToHost, destination.size());
    };
    return resources;
  }
  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& gemma = dynamic_cast<ContinuationAdapter&>(adapter);
    return std::make_unique<State>(gemma.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(ext::TextRunnerState& state,
                               std::size_t position) const override {
    if (Get(state).session().Position() != position) {
      throw std::logic_error("Gemma 4 component state position differs");
    }
  }

  [[nodiscard]] ext::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities =
            ext::TextRunnerCapabilities{
                .supports_vision = vision_ != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = model_->HasMtp(),
                .batched_multi_token_decode = model_->HasMtp(),
                .batched_multi_token_decode_max_width = 8,
                .prefix_reuse = true,
                .in_pass_checkpoint = true,
            },
        .sampling_defaults = SamplingDefaults(),
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    // Snapshots are device allocations carved from host RAM on the APU: cap
    // both limits by the host and by the device heap.
    std::size_t snapshot_capacity = ext::HostSnapshotBudgetBytes();
    std::size_t snapshot_ceiling = ext::HostSnapshotCeilingBytes();
    if (capacity.has_value()) {
      snapshot_capacity = std::min(snapshot_capacity, *capacity);
      snapshot_ceiling = std::min(snapshot_ceiling, *capacity);
    }
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = model_->SessionBytes(max_context_),
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = snapshot_capacity,
        .retained_snapshot_ceiling_bytes = snapshot_ceiling,
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<ext::TextExecutionPlan> SupportedPlans()
      const override {
    // Up to eight sessions share one forward (decode widths stay exact).
    std::vector<ext::TextExecutionPlan> plans{
        {.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({.kind = ext::TextExecutionPlanKind::kBatched,
                       .physical_width = width});
    }
    return plans;
  }

  [[nodiscard]] std::string ToolDialectId() const override {
    return std::string(kToolDialectId);
  }
  [[nodiscard]] ReasoningMarkup ReasoningDelimiters() const override {
    return gemma4::ReasoningDelimiters();
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return model_->VocabSize();
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    const auto tokens = model_->Tokenize(text);
    return {tokens.begin(), tokens.end()};
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    auto prepared = PreparePrompt(request);
    if (!prepared) {
      return std::nullopt;
    }
    return std::move(prepared->tokens);
  }

  [[nodiscard]] std::optional<ext::TextPreparedPrompt> PreparePrompt(
      const ext::ChatRequest& request) const override {
    if (std::ranges::any_of(request.messages,
                            [](const auto& m) { return !m.images.empty(); })) {
      return PrepareImagePrompt(request);
    }
    std::string error;
    const auto rendered = ChatTemplate::Render(request.messages, Tools(request),
                                               Options(request), &error);
    if (!rendered) {
      throw std::invalid_argument("Gemma 4 chat template: " + error);
    }
    ext::TextPreparedPrompt prepared;
    prepared.tokens = Tokenize(rendered->text);
    // The stable prefix is shared by the next turn; the text after it starts
    // with a special token, so it tokenizes apart.
    if (rendered->stable_prefix_offset < rendered->text.size()) {
      const auto suffix = Tokenize(std::string_view(rendered->text)
                                       .substr(rendered->stable_prefix_offset));
      if (suffix.size() < prepared.tokens.size() &&
          std::equal(suffix.begin(), suffix.end(),
                     prepared.tokens.end() -
                         static_cast<std::ptrdiff_t>(suffix.size()))) {
        prepared.cache_prefix_tokens = prepared.tokens.size() - suffix.size();
      }
    }
    return prepared;
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ext::ChatRequest& request) const override {
    if (std::ranges::none_of(request.messages,
                             [](const auto& m) { return !m.images.empty(); })) {
      const auto prepared = PreparePrompt(request);
      if (!prepared) {
        return std::nullopt;
      }
      return prepared->tokens.size();
    }
    if (!vision_) {
      throw std::invalid_argument(
          "image input requires a matching --mmproj BF16 sidecar");
    }
    // Generation's CPU image preparation, without encoding. No context limit,
    // so oversized prompts can still be sized.
    return vision::Prepare(
               model_->tokenizer(), request.messages, Tools(request),
               Options(request), vision_->identity(),
               std::numeric_limits<std::uint32_t>::max(), image_tokens_)
        .tokens.size();
  }

  void SetPromptContext(
      ext::TextRunnerState& state,
      std::shared_ptr<const ext::TextPromptContext> context) const override {
    std::shared_ptr<const vision::Prompt> prompt;
    if (context) {
      const auto* image = dynamic_cast<const ImageContext*>(context.get());
      if (image == nullptr) {
        throw std::invalid_argument("invalid Gemma 4 prompt context");
      }
      prompt = image->prompt;
    }
    Get(state).SetImages(std::move(prompt));
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ext::ChatRequest& request) const override {
    const auto options = Options(request);
    if (!options.enable_thinking) {
      // Gemma 4 still opens an empty thought channel after tool responses.
      return ext::InitialOutputState::kAuto;
    }
    // After a tool response the prompt already opened the thought channel.
    std::string error;
    const auto rendered =
        ChatTemplate::Render(request.messages, Tools(request), options, &error);
    if (rendered && rendered->text.ends_with(kThoughtStart)) {
      return ext::InitialOutputState::kReasoning;
    }
    return ext::InitialOutputState::kAuto;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const auto token = static_cast<TokenId>(id);
          return sampling::ConstraintVocabulary::Piece{
              model_->Decode(std::span(&token, 1)), model_->IsStopToken(token)};
        });
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return model_->Decode(EngineTokens(tokens));
  }

  void PreparePrefixReuse(
      ext::TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& session = Get(state).session();
    if (session.Position() != prefix.size()) {
      throw std::logic_error("Gemma 4 reused prefix does not match checkpoint");
    }
    // A fully reused prompt decodes without a Sync; it still starts a request.
    session.BeginRequest();
  }

  [[nodiscard]] ext::TextPrefillStep Prefill(
      ext::TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& gemma = Get(state);
    const std::size_t end = StepEnd(gemma, prompt, offset, max_input_tokens);
    // A step that ends the model prompt short of the request's (a checkpoint
    // or the boundary before the generation prompt) evaluates the pool's
    // lookahead, the tokens up to the next stop, alongside; the next step
    // adopts them. Image rows are never looked ahead.
    std::span<const TextRunnerToken> lookahead;
    if (end == prompt.size() && !ImageAtOrAfter(gemma, end)) {
      lookahead = state.prefill_lookahead();
    }
    const auto prefix = EngineTokens(prompt.first(end));
    Sync(gemma, prefix, end, EngineTokens(lookahead));
    return {.consumed_tokens = end - offset,
            .decode_ready = end == prompt.size()};
  }

  // The checkpoint at `boundary` comes from the generation prompt's own
  // forward: the session syncs to the boundary with the rest of the step as
  // lookahead, the pool captures it there, and the step's end adopts the
  // looked-ahead rows. The component lease sizes the capture itself.
  [[nodiscard]] std::optional<std::size_t> PrefillCheckpointBytes(
      const ext::TextRunnerState& state,
      std::span<const TextRunnerToken> prompt, std::size_t offset,
      std::size_t budget, std::size_t boundary) const override {
    const auto& gemma = Get(state);
    if (offset != gemma.session().Position() || offset >= prompt.size()) {
      return std::nullopt;
    }
    const std::size_t end = StepEnd(gemma, prompt, offset, budget);
    if (!InPassBoundary(gemma, offset, end, boundary)) {
      return std::nullopt;
    }
    return model_->SessionBytes(max_context_);
  }

  [[nodiscard]] ext::TextPrefillStep PrefillThrough(
      ext::TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t budget,
      std::span<const std::size_t> boundaries,
      const InPassConsumer& consume) const override {
    auto& gemma = Get(state);
    // One boundary per pass (max_in_pass_checkpoints keeps its default).
    if (!consume || boundaries.size() != 1) {
      throw std::logic_error("invalid Gemma 4 in-pass checkpoint request");
    }
    const std::size_t boundary = boundaries.front();
    const std::size_t end = StepEnd(gemma, prompt, offset, budget);
    if (!InPassBoundary(gemma, offset, end, boundary)) {
      throw std::logic_error("Gemma 4 in-pass boundary is outside the step");
    }
    const auto prefix = EngineTokens(prompt.first(end));
    Sync(gemma, std::span(prefix).first(boundary), boundary,
         std::span(prefix).subspan(boundary));
    // The lease captures the session at the boundary through the adapter.
    const auto started = std::chrono::steady_clock::now();
    consume(0);
    const double capture_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
    Sync(gemma, prefix, end, {});
    server::Logger::Debug(
        "gemma4", "event=in_pass_checkpoint offset=" + std::to_string(offset) +
                      " boundary=" + std::to_string(boundary) +
                      " end=" + std::to_string(end) +
                      " capture_ms=" + std::to_string(capture_ms));
    return {.consumed_tokens = end - offset,
            .decode_ready = end == prompt.size(),
            .checkpoint_ms = capture_ms};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    auto& session = Get(state).session();
    if (session.Position() >= max_context_) {
      return {.stop = true, .piece = {}};
    }
    const auto logits = session.Logits();
    if (logits.empty()) {
      throw std::runtime_error("Gemma 4 token selection has no logits");
    }
    const auto token = static_cast<TokenId>(sampler.Sample(logits));
    auto logprobs =
        sampling::CollectLogprobs(logits, token, sampler.config().logprobs);
    if (state.stop_at_eos() && model_->IsStopToken(token)) {
      return {.stop = true,
              .token = 0,
              .piece = {},
              .logprobs = std::move(logprobs)};
    }
    return {.stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->TokenText(token),
            .logprobs = std::move(logprobs)};
  }

  void Advance(ext::TextRunnerState& state,
               TextRunnerToken token) const override {
    std::string error;
    if (!Get(state).session().Evaluate(EngineToken(token), &error)) {
      throw std::runtime_error("Gemma 4 decode failed: " + error);
    }
  }

  [[nodiscard]] std::optional<ext::TextDecodeSelection> PreviewFirstToken(
      ext::TextRunnerState& state,
      sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] ext::TextDecodeStep DecodeStep(
      ext::TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    auto& session = Get(state).session();
    // Select-and-advance samples Logits(), which only predicts the next
    // token while no emitted token is pending.
    if (!model_->HasMtp() || (max_tokens == 1 && !session.HasPendingToken())) {
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    if (session.Position() >= max_context_) {
      return {.selections = {}, .stop = true};
    }
    const auto before = session.Statistics();
    sampling::SamplerState working = sampler;
    Session::DecodeResult decoded;
    std::string error;
    if (!session.DecodeStep(
            std::min<std::size_t>(max_tokens, model_->DraftTokens() + 1),
            working, &decoded, &error, state.stop_at_eos())) {
      throw std::runtime_error("Gemma 4 MTP decode failed: " + error);
    }
    sampler.CopyDrawStateFrom(working);
    ext::TextDecodeStep step;
    AppendTokens(decoded, state.stop_at_eos(), &step);
    Count(before, session.Statistics(), &step);
    return step;
  }

  void AdvanceBatch(
      std::span<const ext::TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      return TextModelRunner::AdvanceBatch(advances);
    }
    std::vector<Session*> sessions;
    std::vector<TokenId> tokens;
    for (const auto& advance : advances) {
      sessions.push_back(&Get(advance.state.get()).session());
      tokens.push_back(EngineToken(advance.token));
    }
    std::string error;
    if (!Session::EvaluateBatch(sessions, tokens, &error)) {
      const auto failure = std::make_exception_ptr(
          std::runtime_error("Gemma 4 batched decode failed: " + error));
      for (const auto& advance : advances) {
        if (!advance.failure) {
          std::rethrow_exception(failure);
        }
        *advance.failure = failure;
      }
    }
  }

  [[nodiscard]] std::vector<ext::TextDecodeStep> DecodeBatch(
      std::span<const ext::TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || !model_->HasMtp()) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    std::vector<ext::TextDecodeStep> steps(decodes.size());
    std::vector<Session::DecodeResult> results(decodes.size());
    std::vector<Session::SpeculativeStats> before(decodes.size());
    std::vector<sampling::SamplerState> working;
    working.reserve(decodes.size());
    std::vector<Session::BatchDecode> batch;
    std::vector<std::size_t> index;
    for (std::size_t i = 0; i < decodes.size(); ++i) {
      auto& state = Get(decodes[i].state.get());
      auto& session = state.session();
      if (decodes[i].max_tokens == 1 && !session.HasPendingToken()) {
        // A one-token step evaluates its token, as DecodeStep does.
        try {
          steps[i] = DecodeStep(state, 1, decodes[i].sampler.get());
        } catch (...) {
          steps[i].failure = std::current_exception();
        }
        continue;
      }
      if (session.Position() >= max_context_) {
        steps[i].stop = true;
        continue;
      }
      before[i] = session.Statistics();
      working.push_back(decodes[i].sampler.get());
      batch.push_back({.session = &session,
                       .max_tokens = std::min<std::size_t>(
                           decodes[i].max_tokens, model_->DraftTokens() + 1),
                       .sampler = &working.back(),
                       .result = &results[i],
                       .stop_at_eos = state.stop_at_eos(),
                       .error = {}});
      index.push_back(i);
    }
    (void)Session::DecodeBatch(batch);
    const ext::TextExecutionPlan plan{
        .kind = ext::TextExecutionPlanKind::kBatched,
        .physical_width = batch.size()};
    for (std::size_t k = 0; k < batch.size(); ++k) {
      const std::size_t i = index[k];
      auto& step = steps[i];
      if (!batch[k].error.empty()) {
        step.failure = std::make_exception_ptr(
            std::runtime_error("Gemma 4 MTP decode failed: " + batch[k].error));
        continue;
      }
      decodes[i].sampler.get().CopyDrawStateFrom(working[k]);
      AppendTokens(results[i], batch[k].stop_at_eos, &step);
      Count(before[i], batch[k].session->Statistics(), &step);
      step.execution_plan = plan;
    }
    return steps;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const ext::TextRunnerState& state) const override {
    return Get(state).session().Position();
  }

private:
  static std::span<const ext::ChatTool> Tools(const ext::ChatRequest& request) {
    return request.tool_choice == ext::ChatRequest::ToolChoice::kNone
               ? std::span<const ext::ChatTool>{}
               : std::span<const ext::ChatTool>{request.tools};
  }
  static ChatOptions Options(const ext::ChatRequest& request) {
    return ResolveChatOptions(request.reasoning);
  }

  // Images extend the cache identity at their first token; a prefix ending
  // at or before an image holds only the earlier images.
  [[nodiscard]] ext::TextPreparedPrompt PrepareImagePrompt(
      const ext::ChatRequest& request) const {
    if (!vision_) {
      throw std::invalid_argument(
          "image input requires a matching --mmproj BF16 sidecar");
    }
    auto prompt = std::make_shared<const vision::Prompt>(vision::Prepare(
        model_->tokenizer(), request.messages, Tools(request), Options(request),
        vision_->identity(), max_context_, image_tokens_));
    auto context = std::make_shared<ImageContext>();
    context->cache_identity = prompt->cache_identity;
    for (const auto& image : prompt->images) {
      const auto identity = prompt->IdentityForPrefix(image.span.offset);
      context->cache_prefixes.push_back(
          {image.span.offset, {identity.begin(), identity.end()}});
    }
    context->prompt = prompt;
    return {{prompt->tokens.begin(), prompt->tokens.end()},
            std::move(context),
            prompt->stable_prefix_tokens};
  }

  // A step never ends inside an image or before its closing token: it stops
  // before the image, or takes the whole image when it starts there.
  [[nodiscard]] static std::size_t StepEnd(
      const State& gemma, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) {
    if (offset != gemma.session().Position() || offset >= prompt.size()) {
      throw std::logic_error("Gemma 4 prefill offset does not match state");
    }
    std::size_t end =
        offset + std::min(max_input_tokens, prompt.size() - offset);
    if (const auto* images = gemma.images()) {
      for (const auto& image : images->images) {
        const std::size_t first = image.span.offset;
        const std::size_t last = first + image.span.rows;  // closing token
        if (end > first && end <= last) {
          end = first > offset ? first : last + 1;
        }
      }
    }
    return end;
  }
  [[nodiscard]] static bool ImageAtOrAfter(const State& gemma,
                                           std::size_t position) {
    const auto* images = gemma.images();
    return images != nullptr &&
           std::ranges::any_of(images->images, [&](const auto& image) {
             return image.span.offset >= position;
           });
  }
  // The rows past the boundary ride in the boundary's forward only while
  // they are text and within the engine's lookahead limit.
  [[nodiscard]] static bool InPassBoundary(const State& gemma,
                                           std::size_t offset, std::size_t end,
                                           std::size_t boundary) {
    if (boundary <= offset || boundary >= end ||
        end - boundary > kLookaheadRows) {
      return false;
    }
    const auto* images = gemma.images();
    return images == nullptr ||
           std::ranges::none_of(images->images, [&](const auto& image) {
             const std::size_t first = image.span.offset;
             return first + image.span.rows >= boundary && first < end;
           });
  }

  // Syncs the session to `prefix` with the images that end before it.
  void Sync(State& gemma, std::span<const TokenId> prefix, std::size_t end,
            std::span<const TokenId> lookahead) const {
    std::vector<ImageSpan> spans;
    if (const auto* images = gemma.images()) {
      for (const auto& image : images->images) {
        if (std::size_t{image.span.offset} + image.span.rows < end) {
          spans.push_back(
              {image.span.offset, image.span.rows, image.prefix_identity});
        }
      }
    }
    const auto embed = [&](std::size_t index) {
      if (!vision_) {
        throw std::invalid_argument("Gemma 4 image input has no encoder");
      }
      return gemma.Embedding(*vision_, index);
    };
    std::string error;
    if (!gemma.session().Sync(prefix, spans, embed, &error, lookahead)) {
      gemma.Invalidate();
      throw std::runtime_error("Gemma 4 prefill failed: " + error);
    }
  }

  // Emits decoded tokens up to a stop token, whose logprobs become the
  // step's terminal entry.
  void AppendTokens(const Session::DecodeResult& decoded, bool stop_at_eos,
                    ext::TextDecodeStep* step) const {
    const auto logprobs = [&](std::size_t i) {
      return decoded.logprobs.empty() ? std::nullopt
                                      : std::optional(decoded.logprobs.at(i));
    };
    for (std::size_t i = 0; i < decoded.tokens.size(); ++i) {
      const TokenId token = decoded.tokens[i];
      if (stop_at_eos && model_->IsStopToken(token)) {
        step->stop = true;
        step->terminal_logprobs = logprobs(i);
        break;
      }
      step->selections.push_back({.stop = false,
                                  .token = static_cast<TextRunnerToken>(token),
                                  .piece = model_->TokenText(token),
                                  .logprobs = logprobs(i)});
    }
  }
  static void Count(const Session::SpeculativeStats& before,
                    const Session::SpeculativeStats& after,
                    ext::TextDecodeStep* step) {
    step->draft_rounds = after.verified - before.verified;
    step->draft_tokens = after.drafted - before.drafted;
    step->draft_accepted_tokens = after.accepted - before.accepted;
    step->lookup_tokens = after.copied - before.copied;
    step->lookup_accepted_tokens =
        after.copied_accepted - before.copied_accepted;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<vision::Encoder> vision_;
  std::uint32_t image_tokens_;
  std::uint32_t max_context_;
  cache::Identity component_identity_;
};

std::string Fingerprint(const ext::TextRunnerOptions& options,
                        std::string_view label, const std::string& path,
                        bool target, std::string* error) {
  try {
    std::shared_ptr<const core::GgufReader> reader =
        target ? options.reader : nullptr;
    if (!reader) {
      auto opened = core::GgufReader::OpenFile(path, error);
      if (!opened) {
        return {};
      }
      reader = std::move(opened);
    }
    return core::GgufIdentityHex(*reader);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = std::string(label) + " fingerprinting failed: " + e.what();
    }
    return {};
  }
}

// Hashes and uploads the sidecar at load, so the first image request pays
// neither.
void PrepareVision(vision::Encoder* encoder) {
  if (encoder == nullptr) {
    return;
  }
  using Clock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  const auto started = Clock::now();
  (void)encoder->identity();
  const auto hashed = Clock::now();
  try {
    encoder->Upload();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("could not upload the Gemma 4 vision sidecar: ") +
        e.what());
  }
  const auto uploaded = Clock::now();
  std::size_t heap_free = 0;
  std::size_t heap_total = 0;
  (void)hipMemGetInfo(&heap_free, &heap_total);
  server::Logger::Info(
      "loader", "event=vision_prepared resident_bytes=" +
                    std::to_string(encoder->ResidentBytes()) + " identity_ms=" +
                    std::to_string(Milliseconds(hashed - started).count()) +
                    " upload_ms=" +
                    std::to_string(Milliseconds(uploaded - hashed).count()) +
                    " device_heap_free_bytes=" + std::to_string(heap_free) +
                    " " + server::Logger::MemoryStatus());
}

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateGemma4TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  const auto& speculative = options.speculative;
  const bool mtp = speculative.backend == ext::TextSpeculativeBackend::kMtp;
  if (speculative.backend != ext::TextSpeculativeBackend::kDisabled && !mtp) {
    SetError(error,
             "Gemma 4 HTTP models support only MTP speculative decoding "
             "(--speculative mtp --mtp-model)");
    return nullptr;
  }
  if (mtp && (speculative.draft_model_path.empty() ||
              speculative.max_draft_tokens == 0)) {
    SetError(error,
             "Gemma 4 MTP requires --mtp-model and a positive draft limit");
    return nullptr;
  }
  // TODO(gemma4): --draft-policy, --draft-calibration, --draft-trace,
  // --prefill-weights and --image-tokens need serve plumbing (#979); they
  // keep their defaults. The host passes --draft-tokens without telling
  // whether it was given, so its value (default 7) is used as is.
  ModelOptions model_options{
      .max_context = options.max_context,
      .mtp_model_path = mtp ? speculative.draft_model_path : std::string{},
      .draft_tokens = speculative.max_draft_tokens,
      .min_draft_tokens = speculative.min_draft_tokens,
  };
  std::string load_error;
  auto model = Model::Load(options.model_path, model_options, &load_error);
  if (model == nullptr) {
    SetError(error, "Failed to create Gemma 4 model: " + load_error);
    return nullptr;
  }
  std::shared_ptr<vision::Encoder> vision;
  try {
    vision =
        vision::Encoder::Open(options.model_path, options.vision_model_path,
                              model->config().hidden_size);
  } catch (const std::exception& e) {
    SetError(error, std::string("Gemma 4 vision sidecar: ") + e.what());
    return nullptr;
  }
  const std::uint32_t image_tokens = vision::kDefaultSoftTokens;
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    max_context = model->MaxContext();
  }
  if (options.session_count == 0 || max_context == 0 ||
      max_context > model->MaxContext()) {
    SetError(error, "Gemma 4 HTTP sessions or context exceed the loaded model");
    return nullptr;
  }
  if (mtp != model->HasMtp()) {
    SetError(error, "Gemma 4 MTP requires a model loaded with its drafter");
    return nullptr;
  }
  std::string artifact = options.disk_cache.model_artifact_fingerprint;
  std::string draft = options.disk_cache.draft_model_artifact_fingerprint;
  if (!options.disk_cache.directory.empty()) {
    if (artifact.empty()) {
      artifact =
          Fingerprint(options, "Gemma 4", options.model_path, true, error);
      if (artifact.empty()) {
        return nullptr;
      }
    }
    if (mtp && draft.empty()) {
      draft = Fingerprint(options, "Gemma 4 MTP", speculative.draft_model_path,
                          false, error);
      if (draft.empty()) {
        return nullptr;
      }
    }
    if (!IsSha256Hex(artifact) || (mtp && !IsSha256Hex(draft)) ||
        options.disk_cache.capacity_bytes == 0) {
      SetError(error, "Gemma 4 persistent disk cache configuration is invalid");
      return nullptr;
    }
  }
  try {
    PrepareVision(vision.get());
    const auto ring = static_cast<std::uint32_t>(model->SessionRingSlots());
    return std::make_shared<Runner>(std::move(model), max_context, ring,
                                    artifact, draft, std::move(vision),
                                    image_tokens);
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return nullptr;
  }
}

}  // namespace gufo::models::gemma4
