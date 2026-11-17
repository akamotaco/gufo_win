#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/json.hpp>
#include <gufo/ext/text.hpp>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
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
#include "dspark_sampler.hpp"
#include "engine.hpp"
#include "provider_options.hpp"
#include "text_contract.hpp"
#include "vision/encoder.hpp"
#include "vision/prompt.hpp"

namespace gufo::models::deepseek_v41_flash {
namespace {

using namespace gufo::ext;

constexpr std::string_view kStateAbi = "deepseek-v4.1-flash-gfx1151-state-v1";

// The memory left to a streamed model's expert cache depends on the RAM cache,
// so its automatic budget is a fixed share, smaller with a disk cache.
constexpr std::size_t kStreamedRamShareBytes = std::size_t{3} << 30;
constexpr std::size_t kStreamedDiskRamShareBytes = std::size_t{1} << 30;
// What the host's backing commits before readiness.
constexpr std::size_t kBackingGrowAheadBytes = std::size_t{2} << 30;
// Prompt-cache transfer streams besides each session's own: one capture or
// restore batch leases at most this many. Gufo's default of 64 holds about
// 0.4 GiB more GPU memory, which streamed builds lose from the expert cache.
constexpr std::size_t kBatchTransferStreams = 16;

// The RAM cache's host limit. The host keeps a fixed reserve above it, which
// resident IQ1_M with the drafter and vision does not leave (about 4 GiB free
// after the sessions), so the host would refuse the load. The automatic budget,
// half the RAM still available, may use that reserve; it shrinks as the
// backing commits, so the host keeps at least as much as the cache takes.
std::size_t CacheCeilingBytes() {
  return std::max(HostSnapshotCeilingBytes(), HostSnapshotBudgetBytes());
}

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

// Prompt evaluation decides what a checkpoint holds. Each prefill pass replays
// the decoder layers over the last tokens up to its own end, a cache boundary
// or the prompt's end, so these checkpoints differ from a whole-prompt replay.
std::string_view PrefillPolicy(bool exact) {
  return exact ? "exact" : "decoder-swa-replay-140-per-pass";
}

std::vector<std::uint8_t> CompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view support_fingerprint,
    std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::string_view prefill_policy, std::string_view adapters) {
  if (!IsSha256Hex(artifact_fingerprint)) {
    throw std::invalid_argument(
        "DeepSeek disk cache requires an artifact fingerprint");
  }
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=deepseek41\n"
           << "artifact_id=" << ext::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=joyai-byte-bpe-v1\n"
           << "chat_template=" << ChatTemplateId() << '\n'
           << "chat_template_reference_sha256=" << EncoderReferenceSha256()
           << '\n'
           << "state_abi=" << kStateAbi << '\n'
           << "payload_layout=ds41-rocm-v1-window128-decoder\n"
           << "numerics=ds41-decoder-v1\n"
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "rope_window_policy=deepseek41-compiled-v1\n"
           << "adapters=" << adapters << '\n'
           << "prefill_policy=" << prefill_policy << '\n'
           << "support_id=" << ext::kGgufIdentityScheme << ':'
           << support_fingerprint << '\n'
           << "max_draft_tokens=" << max_draft_tokens << '\n'
           << "draft_policy=dspark-cost-v5-window128\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

// Without file fingerprints the loaded model is the identity; it is valid only
// inside this process, so it can never match a durable cache.
std::vector<std::uint8_t> ProcessIdentity(const void* model,
                                          std::uint32_t max_context,
                                          bool use_dspark,
                                          std::uint32_t max_draft_tokens,
                                          std::string_view prefill_policy) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "dspark=" << use_dspark << '\n'
           << "draft_max_tokens=" << max_draft_tokens << '\n'
           << "prefill_policy=" << prefill_policy << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

class DeepSeekTextRunnerState final : public TextRunnerState {
public:
  DeepSeekTextRunnerState(Session& session, cache::Adapter& adapter,
                          cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void SetCancellationCheck(const CancellationCheck& is_cancelled) override {
    session_->SetCancellationCheck(is_cancelled);
  }

  void Invalidate() noexcept override {
    session_->SetCancellationCheck({});
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
  }

  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    const std::uint64_t bytes = session_->PayloadBytes();
    if (bytes == 0 || bytes > static_cast<std::uint64_t>(
                                  std::numeric_limits<std::size_t>::max())) {
      return {};
    }
    return {
        .per_request_state_bytes = static_cast<std::size_t>(bytes),
        .temporary_scratch_bytes = std::nullopt,
    };
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

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
      const auto& image = images_->images.at(index);
      embedding = encoder.Encode(*image.pixels, image.grid);
    }
    return embedding->data();
  }

private:
  Session* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t position_{0};
  std::shared_ptr<const vision::Prompt> images_;
  std::map<std::size_t, std::shared_ptr<const vision::Encoder::Embedding>>
      embeddings_;
};

struct DeepSeekImageContext final : TextPromptContext {
  std::shared_ptr<const vision::Prompt> prompt;
};

DeepSeekTextRunnerState& RequireDeepSeekState(TextRunnerState& state) {
  auto* deepseek = dynamic_cast<DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

const DeepSeekTextRunnerState& RequireDeepSeekState(
    const TextRunnerState& state) {
  const auto* deepseek = dynamic_cast<const DeepSeekTextRunnerState*>(&state);
  if (deepseek == nullptr) {
    throw std::logic_error("text runner state is not DeepSeek");
  }
  return *deepseek;
}

std::vector<TextRunnerToken> DeepSeekRunnerTokens(std::span<const int> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const int token : tokens) {
    if (token < 0) {
      throw std::invalid_argument("DeepSeek token ID must not be negative");
    }
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<int> DeepSeekEngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<int> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    converted.push_back(static_cast<int>(token));
  }
  return converted;
}

class DeepSeekTextRunner final : public HipTextModelRunner {
public:
  [[nodiscard]] std::string ToolDialectId() const override {
    return std::string(kToolDialectId);
  }
  DeepSeekTextRunner(std::shared_ptr<Model> model, std::uint32_t max_context,
                     bool use_dspark, std::uint32_t max_draft_tokens,
                     std::string artifact_fingerprint,
                     std::string support_fingerprint,
                     std::shared_ptr<vision::Encoder> vision,
                     std::size_t snapshot_budget_cap,
                     std::size_t retained_ram_bytes, std::size_t sessions)
      : model_(std::move(model)),
        vision_(std::move(vision)),
        max_context_(max_context),
        use_dspark_(use_dspark),
        max_draft_tokens_(std::max(max_draft_tokens, 1u)),
        snapshot_budget_cap_(snapshot_budget_cap),
        retained_ram_bytes_(retained_ram_bytes),
        sessions_(sessions) {
    const auto policy = PrefillPolicy(model_->ExactPrefill());
    if (!artifact_fingerprint.empty()) {
      if (use_dspark_ && !IsSha256Hex(support_fingerprint)) {
        throw std::invalid_argument(
            "DSpark disk cache requires a support SHA-256 fingerprint");
      }
      component_identity_ = CompatibilityIdentity(
          artifact_fingerprint,
          use_dspark_ ? support_fingerprint : std::string{}, max_context_,
          max_draft_tokens_, policy, model_->AdapterIdentity());
    } else {
      component_identity_ = ProcessIdentity(
          model_.get(), max_context_, use_dspark_, max_draft_tokens_, policy);
    }
  }

  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t ram) const override {
    (void)ram;
    auto adapter = std::make_unique<ContinuationAdapter>(
        model_, max_context_, component_identity_, use_dspark_);
    auto resources = hip::CreateHipComponentCacheResources(
        std::move(adapter), ledger,
        {[] { return CacheCeilingBytes(); },
         [](std::string_view message) {
           server::Logger::Warn("cache", message);
         },
         [](std::string_view message) {
           server::Logger::Info("cache", message);
         },
         kBatchTransferStreams + sessions_});
    // The host measures free memory for the RAM cache and commits its backing
    // before readiness, so the expert cache takes what is left only then.
    if (model_->StreamsExperts() && resources->commit_backing) {
      resources->commit_backing =
          [this, commit = resources->commit_backing](std::size_t budget) {
            commit(budget);
            FitExpertCache();
          };
    }
    return resources;
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    // A server with no RAM or disk cache commits no backing.
    FitExpertCache();
    auto& deepseek = dynamic_cast<ContinuationAdapter&>(adapter);
    return std::make_unique<DeepSeekTextRunnerState>(deepseek.GetSession(slot),
                                                     adapter, slot);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.session().Position() != static_cast<int>(position)) {
      throw std::logic_error("DeepSeek component state position differs");
    }
    deepseek.set_position(position);
  }

  [[nodiscard]] bool CanContinue(const TextRunnerState& state,
                                 std::size_t prompt_tokens) const override {
    return RequireDeepSeekState(state).session().CanContinue(prompt_tokens);
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    // Speculative steps verify prompt-lookup copies, also without a drafter.
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .supports_vision = vision_ != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = true,
                .batched_multi_token_decode = true,
                .batched_multi_token_decode_max_width = 8u,
                .prefix_reuse = true,
            },
        .sampling_defaults = SamplingDefaults(),
        // Prompts beside decoders take one prefill batch per pass: larger
        // passes only queue more batches, smaller ones repeat each pass's
        // expert reads. Passes run unsplit.
        .scheduling = {.pass_tokens = model_->PrefillBatch(),
                       .max_layer_groups = 1u,
                       .max_yield_decode_width = 8u},
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
      capacity = free_bytes;
    }
    return {
        .resident_weights_bytes = std::nullopt,
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes =
            snapshot_budget_cap_
                ? std::min(HostSnapshotBudgetBytes(), snapshot_budget_cap_)
                : HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = CacheCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{
        {
            .kind = TextExecutionPlanKind::kSerial,
            .physical_width = 1,
        },
    };
    for (std::size_t width = 2; width <= 8; ++width) {
      plans.push_back({
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = width,
      });
    }
    return plans;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              model_->DecodeToken(static_cast<int>(id)),
              model_->IsStopToken(static_cast<int>(id))};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override {
    return static_cast<std::uint32_t>(model_->VocabSize());
  }

  // Raw prompts map control-token spellings to their special tokens, as the
  // chat template's render does and as llama-server's /completion does.
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return DeepSeekRunnerTokens(model_->Tokenize(text, true));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    const auto input = ConvertChat(request);
    auto tokens = DeepSeekRunnerTokens(
        model_->EncodeChat(input.messages, input.tools, input.options));
    if (tokens.empty()) {
      return std::nullopt;
    }
    return tokens;
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    const auto input = ConvertChat(request);
    return RenderChat(input.messages, input.tools, input.options);
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(
      const ChatRequest& request) const override {
    return ResolveDeepSeekChatOptions(request.reasoning).enable_thinking
               ? ext::InitialOutputState::kReasoning
               : ext::InitialOutputState::kContent;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    if (HasImages(request)) {
      return PrepareImagePrompt(request);
    }
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (prepared) {
      // DeepSeek joins adjacent user/tool messages into one user block.
      // A new user turn after a discarded assistant can therefore also change
      // the final BPE token before the generation suffix (e.g. ">\n\n").
      // Find the exact stable frontier using the renderer/tokenizer instead
      // of assuming the suffix alone is the only changed part of the prompt.
      auto continuation = request;
      continuation.messages.emplace_back(ext::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared->cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared->tokens.begin(), prepared->tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared->tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    if (!HasImages(request)) {
      return TextModelRunner::CountPromptTokens(request);
    }
    // Generation's CPU image preparation, without encoding or a context
    // limit, so oversized prompts can still be sized.
    const auto prompt =
        ExpandImages(request, std::numeric_limits<std::uint32_t>::max());
    if (!prompt) {
      return std::nullopt;
    }
    return prompt->tokens.size();
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    std::shared_ptr<const vision::Prompt> prompt;
    if (context) {
      const auto* image =
          dynamic_cast<const DeepSeekImageContext*>(context.get());
      if (image == nullptr) {
        throw std::invalid_argument("invalid DeepSeek prompt context");
      }
      prompt = image->prompt;
    }
    RequireDeepSeekState(state).SetImages(std::move(prompt));
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const TextRunnerToken token : tokens) {
      if (token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      text += model_->DecodeToken(static_cast<int>(token));
    }
    return text;
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() != prefix.size()) {
      throw std::logic_error(
          "DeepSeek reused prefix does not match checkpoint");
    }
    deepseek.session().BeginRequest();
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (offset != deepseek.position()) {
      throw std::logic_error(
          "DeepSeek prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) {
      throw std::logic_error("DeepSeek prefill has no remaining input");
    }
    // A prompt prefilled from its start is a new request, like one that
    // reuses a prefix (PreparePrefixReuse): its draft policy starts afresh.
    if (offset == 0) {
      deepseek.session().BeginRequest();
    }

    const std::size_t consumed =
        std::min<std::size_t>({max_input_tokens, prompt.size() - offset,
                               deepseek.session().PrefillCapacity()});
    const std::size_t next_position = offset + consumed;
    const auto prefix = DeepSeekEngineTokens(prompt.first(next_position));
    // The embeddings of the image rows this step evaluates; attention is
    // causal, so a step may end inside an image.
    std::vector<ds41_image_span> spans;
    if (const auto* images = deepseek.images()) {
      for (std::size_t i = 0; i < images->images.size(); ++i) {
        const auto& image = images->images[i];
        const std::size_t end = std::size_t{image.offset} + image.grid.Tokens();
        if (end <= offset || image.offset >= next_position) {
          continue;
        }
        if (!vision_) {
          throw std::invalid_argument(
              "DeepSeek V4.1 image input has no encoder");
        }
        spans.push_back({image.offset, image.grid.Tokens(),
                         deepseek.Embedding(*vision_, i)});
      }
    }
    std::string error;
    // The decoder layers run over the whole prompt's last tokens only, also
    // when this step ends at a cache boundary: a capture there continues only
    // prompts whose replay starts within the rows it holds (CanContinue).
    const std::size_t prompt_end =
        std::max(prompt.size(), state.prefill_prompt_end());
    if (!deepseek.session().Sync(prefix, &error, prompt_end, spans)) {
      deepseek.set_position(0);
      throw std::runtime_error("DeepSeek prefill failed: " + error);
    }
    deepseek.set_position(next_position);
    return {
        .consumed_tokens = consumed,
        .decode_ready = next_position == prompt.size(),
    };
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.stop = true, .piece = {}};
    // A speculative step's last token may be unevaluated.
    std::string error;
    if (!deepseek.session().EvaluatePending(&error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.session().Position());
    const auto logits = deepseek.session().CopyLogits(&error);
    if (logits.empty()) {
      throw std::runtime_error("DeepSeek token selection failed: " + error);
    }
    const int token = static_cast<int>(sampler.Sample(logits));
    const bool stop = deepseek.stop_at_eos() && model_->IsStopToken(token);
    return {.stop = stop,
            .token = static_cast<TextRunnerToken>(token),
            .piece = stop ? std::string{} : model_->DecodeToken(token),
            .logprobs = sampling::CollectLogprobs(logits, token,
                                                  sampler.config().logprobs)};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("DeepSeek token ID exceeds engine range");
    }
    auto& deepseek = RequireDeepSeekState(state);
    std::string error;
    if (!deepseek.session().Evaluate(static_cast<int>(token), &error)) {
      throw std::runtime_error("DeepSeek decode failed: " + error);
    }
    deepseek.set_position(deepseek.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return TextDecodeSelection{.stop = true, .piece = {}};
    std::string error;
    const auto logits = deepseek.session().CopyLogits(&error);
    if (logits.empty())
      throw std::runtime_error("DeepSeek token preview failed: " + error);
    const auto token = sampler.Sample(logits);
    return TextDecodeSelection{
        .stop = deepseek.stop_at_eos() &&
                model_->IsStopToken(static_cast<int>(token)),
        .token = token,
        .piece = model_->DecodeToken(static_cast<int>(token)),
        .logprobs =
            sampling::CollectLogprobs(logits, token, sampler.config().logprobs),
    };
  }

  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (max_tokens == 0) {
      throw std::invalid_argument(
          "DeepSeek DSpark decode budget must be at least one token");
    }

    auto& deepseek = RequireDeepSeekState(state);
    if (deepseek.position() >= max_context_)
      return {.selections = {}, .stop = true};
    const auto stats_before = deepseek.session().DsparkStatistics();
    std::optional<DsparkSamplerBridge> bridge;
    if (!sampler.config().can_use_unmodified_argmax() ||
        sampler.config().logprobs) {
      bridge.emplace(sampler);
    }
    std::vector<int> emitted;
    std::string error;
    if (!deepseek.session().DsparkStep(
            max_tokens, max_draft_tokens_, &emitted, &error,
            bridge ? bridge->hook() : nullptr, deepseek.stop_at_eos())) {
      throw std::runtime_error("DeepSeek DSpark decode failed: " + error);
    }
    if (emitted.empty()) {
      throw std::runtime_error("DeepSeek DSpark decode produced no tokens");
    }
    if (bridge) {
      sampler.SetRngState(bridge->rng_state());
    }

    TextDecodeStep step;
    deepseek.set_position(deepseek.session().Position());
    step.selections.reserve(emitted.size());
    for (const int token : emitted) {
      auto logprobs =
          sampler.config().logprobs
              ? std::optional(bridge->logprobs().at(step.selections.size()))
              : std::nullopt;
      if (deepseek.stop_at_eos() && model_->IsStopToken(token)) {
        step.terminal_logprobs = std::move(logprobs);
        step.stop = true;
        break;
      }
      step.selections.push_back({
          .stop = false,
          .token = static_cast<TextRunnerToken>(token),
          .piece = model_->DecodeToken(token),
          .logprobs = std::move(logprobs),
      });
    }
    const auto stats_after = deepseek.session().DsparkStatistics();
    step.draft_rounds = stats_after.steps - stats_before.steps;
    step.draft_tokens =
        stats_after.support_drafted - stats_before.support_drafted;
    step.draft_accepted_tokens =
        stats_after.support_accepted - stats_before.support_accepted;
    step.lookup_tokens = stats_after.lookup_copied - stats_before.lookup_copied;
    step.lookup_accepted_tokens =
        stats_after.lookup_accepted - stats_before.lookup_accepted;
    return step;
  }

  [[nodiscard]] std::vector<TextDecodeStep> DecodeBatch(
      std::span<const TextRunnerDecode> decodes) const override {
    if (decodes.size() < 2 || decodes.size() > 8) {
      return TextModelRunner::DecodeBatch(decodes);
    }
    if (std::any_of(decodes.begin(), decodes.end(), [&](const auto& item) {
          return RequireDeepSeekState(item.state.get()).position() >=
                 max_context_;
        })) {
      std::vector<TextRunnerDecode> active;
      std::vector<std::size_t> active_indices;
      std::vector<TextDecodeStep> steps(decodes.size());
      for (std::size_t i = 0; i < decodes.size(); ++i) {
        if (RequireDeepSeekState(decodes[i].state.get()).position() >=
            max_context_) {
          steps[i].stop = true;
        } else {
          active.push_back(decodes[i]);
          active_indices.push_back(i);
        }
      }
      if (!active.empty()) {
        auto active_steps = DecodeBatch(active);
        for (std::size_t i = 0; i < active.size(); ++i)
          steps[active_indices[i]] = std::move(active_steps[i]);
      }
      return steps;
    }
    // Greedy and sampled requests share one DSpark cohort; the runtime keeps
    // the greedy verifier for items without a sampler.
    std::array<SessionDsparkBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::array<Session::DsparkStats, 8> stats_before{};
    std::array<std::vector<int>, 8> emitted{};
    std::array<std::optional<DsparkSamplerBridge>, 8> bridges{};
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      auto& deepseek = RequireDeepSeekState(decodes[index].state.get());
      states[index] = &deepseek;
      stats_before[index] = deepseek.session().DsparkStatistics();
      auto& sampler = decodes[index].sampler.get();
      if (!sampler.config().can_use_unmodified_argmax() ||
          sampler.config().logprobs) {
        bridges[index].emplace(sampler);
      }
      items[index] = {
          .session = &deepseek.session(),
          .max_tokens = decodes[index].max_tokens,
          .max_draft_tokens = max_draft_tokens_,
          .emitted = &emitted[index],
          .sampler = bridges[index] ? bridges[index]->hook() : nullptr,
          .stop_at_eos = states[index]->stop_at_eos(),
      };
    }

    std::string error;
    bool interrupted = false;
    if (!model_->DsparkStepBatch(std::span<const SessionDsparkBatchItem>(
                                     items.data(), decodes.size()),
                                 &error, &interrupted)) {
      // Cancellation is rejected before sampling or device-state mutation.
      // Isolate the cancelled row rather than failing every client in its
      // batch. Ordinary cycles retain the shared physical batch.
      if (interrupted)
        return TextModelRunner::DecodeBatch(decodes);
      throw std::runtime_error("DeepSeek DSpark batch decode failed: " + error);
    }

    std::vector<TextDecodeStep> steps(decodes.size());
    for (std::size_t index = 0; index < decodes.size(); ++index) {
      if (emitted[index].empty()) {
        throw std::runtime_error("DeepSeek DSpark batch produced no tokens");
      }
      if (bridges[index]) {
        decodes[index].sampler.get().SetRngState(bridges[index]->rng_state());
      }
      auto& step = steps[index];
      step.execution_plan = {
          .kind = TextExecutionPlanKind::kBatched,
          .physical_width = decodes.size(),
      };
      states[index]->set_position(states[index]->session().Position());
      step.selections.reserve(emitted[index].size());
      for (const int token : emitted[index]) {
        auto logprobs = decodes[index].sampler.get().config().logprobs
                            ? std::optional(bridges[index]->logprobs().at(
                                  step.selections.size()))
                            : std::nullopt;
        if (states[index]->stop_at_eos() && model_->IsStopToken(token)) {
          step.terminal_logprobs = std::move(logprobs);
          step.stop = true;
          break;
        }
        step.selections.push_back({
            .stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->DecodeToken(token),
            .logprobs = std::move(logprobs),
        });
      }
      const auto stats_after = states[index]->session().DsparkStatistics();
      step.draft_rounds = stats_after.steps - stats_before[index].steps;
      step.draft_tokens =
          stats_after.support_drafted - stats_before[index].support_drafted;
      step.draft_accepted_tokens =
          stats_after.support_accepted - stats_before[index].support_accepted;
      step.lookup_tokens =
          stats_after.lookup_copied - stats_before[index].lookup_copied;
      step.lookup_accepted_tokens =
          stats_after.lookup_accepted - stats_before[index].lookup_accepted;
    }
    return steps;
  }

  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2 || advances.size() > 8) {
      throw std::invalid_argument(
          "DeepSeek batched decode requires two to eight sessions");
    }

    std::array<SessionBatchItem, 8> items{};
    std::array<DeepSeekTextRunnerState*, 8> states{};
    std::size_t item_count = 0;
    for (const auto& advance : advances) {
      if (advance.token >
          static_cast<TextRunnerToken>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("DeepSeek token ID exceeds engine range");
      }
      auto& deepseek = RequireDeepSeekState(advance.state.get());
      states[item_count] = &deepseek;
      items[item_count] = {
          .session = &deepseek.session(),
          .token = static_cast<int>(advance.token),
      };
      ++item_count;
    }

    std::string error;
    bool interrupted = false;
    if (!model_->EvaluateBatch(
            std::span<const SessionBatchItem>(items.data(), item_count), &error,
            &interrupted)) {
      if (interrupted) {
        TextModelRunner::AdvanceBatch(advances);
        return;
      }
      throw std::runtime_error("DeepSeek batch decode failed: " + error);
    }
    for (std::size_t index = 0; index < item_count; ++index) {
      states[index]->set_position(states[index]->position() + 1);
    }
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireDeepSeekState(state).position();
  }

private:
  struct DeepSeekChatInput {
    std::vector<ChatMessage> messages;
    std::vector<ChatTool> tools;
    ChatTemplateOptions options;
  };

  [[nodiscard]] static std::string_view ChatRoleName(ext::ChatRole role) {
    switch (role) {
      case ext::ChatRole::kSystem:
        return "system";
      case ext::ChatRole::kDeveloper:
        return "developer";
      case ext::ChatRole::kAssistant:
        return "assistant";
      case ext::ChatRole::kTool:
        return "tool";
      case ext::ChatRole::kUser:
        return "user";
    }
    return "user";
  }

  [[nodiscard]] static DeepSeekChatInput ConvertChat(
      const ChatRequest& request) {
    std::vector<ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      ChatMessage converted{
          .role = std::string(ChatRoleName(message.role)),
          .content = message.content + message.framing_suffix,
          .reasoning_content = message.thought,
          .tool_calls = {},
          .tool_call_id = message.tool_call_id,
      };
      // Each image renders as one placeholder token (vision::Expand).
      for (const auto& image : message.images) {
        converted.image_offsets.push_back(image.offset);
      }
      converted.tool_calls.reserve(message.tool_calls.size());
      for (const auto& call : message.tool_calls) {
        ChatMessage::ToolCall converted_call{
            .name = call.name,
            .arguments = {},
            .id = call.id,
        };
        converted_call.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          converted_call.arguments.push_back({
              .name = argument.name,
              .value = argument.value,
              .is_string = argument.is_string,
          });
        }
        converted.tool_calls.push_back(std::move(converted_call));
      }
      messages.push_back(std::move(converted));
    }

    std::vector<ChatTool> tools;
    if (request.tool_choice != ChatRequest::ToolChoice::kNone) {
      tools.reserve(request.tools.size());
      for (const auto& tool : request.tools) {
        tools.push_back({
            .name = tool.name,
            .description = tool.description,
            .parameters_json = tool.parameters_json,
            .definition_json =
                tool.definition_json.empty()
                    ? std::string{}
                    : json::parse(tool.definition_json)["function"].dump(),
        });
      }
    }
    auto options = ResolveDeepSeekChatOptions(request.reasoning);
    options.tools_present =
        !request.tools.empty() &&
        request.tool_choice != ChatRequest::ToolChoice::kNone;
    return {std::move(messages), std::move(tools), options};
  }

  // Sessions and the committed backing exist; the retained checkpoints past
  // the backing's grow-ahead still commit with captures.
  void FitExpertCache() const {
    if (model_->StreamsExperts()) {
      std::call_once(expert_fit_, [this] {
        model_->FitExpertCache(retained_ram_bytes_ > kBackingGrowAheadBytes
                                   ? retained_ram_bytes_ -
                                         kBackingGrowAheadBytes
                                   : 0);
      });
    }
  }

  static bool HasImages(const ChatRequest& request) {
    return std::ranges::any_of(request.messages,
                               [](const auto& m) { return !m.images.empty(); });
  }

  // The rendered prompt with each image placeholder expanded into its span.
  [[nodiscard]] std::optional<vision::Prompt> ExpandImages(
      const ChatRequest& request, std::uint32_t max_context) const {
    if (!vision_) {
      throw std::invalid_argument(
          "image input requires DeepSeek V4.1's --mmproj vision sidecar");
    }
    const auto rendered = RenderAndTokenize(request);
    if (!rendered) {
      return std::nullopt;
    }
    std::vector<std::shared_ptr<const std::vector<std::uint8_t>>> images;
    for (const auto& message : request.messages) {
      for (const auto& image : message.images) {
        images.push_back(image.bytes);
      }
    }
    return vision::Expand(DeepSeekEngineTokens(*rendered), images,
                          vision_->identity(), max_context);
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PrepareImagePrompt(
      const ChatRequest& request) const {
    auto expanded = ExpandImages(request, max_context_);
    if (!expanded) {
      return std::nullopt;
    }
    auto prompt = std::make_shared<const vision::Prompt>(std::move(*expanded));
    // The stable frontier as for text prompts, found among the rendered
    // tokens and moved past the images expanded before it.
    std::size_t cache_prefix = 0;
    auto continuation = request;
    continuation.messages.emplace_back(ext::ChatRole::kUser, "");
    const auto rendered = RenderAndTokenize(request);
    const auto continued = RenderAndTokenize(continuation);
    if (rendered && continued) {
      cache_prefix = prompt->Expanded(static_cast<std::size_t>(
          std::mismatch(rendered->begin(), rendered->end(), continued->begin(),
                        continued->end())
              .first -
          rendered->begin()));
    }
    auto context = std::make_shared<DeepSeekImageContext>();
    context->cache_identity = prompt->cache_identity;
    // A prefix ending at or before an image holds only the earlier images.
    for (const auto& image : prompt->images) {
      const auto identity = prompt->IdentityForPrefix(image.offset);
      context->cache_prefixes.push_back(
          {image.offset, {identity.begin(), identity.end()}});
    }
    context->prompt = prompt;
    return TextPreparedPrompt{DeepSeekRunnerTokens(prompt->tokens),
                              std::move(context), cache_prefix};
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<vision::Encoder> vision_;
  std::uint32_t max_context_;
  bool use_dspark_;
  std::uint32_t max_draft_tokens_;
  std::size_t snapshot_budget_cap_;
  std::size_t retained_ram_bytes_;
  std::size_t sessions_;
  mutable std::once_flag expert_fit_;
  cache::Identity component_identity_;
};

// Hashes and uploads the sidecar at load, so the first image request pays
// neither.
void PrepareVision(vision::Encoder& encoder) {
  using Clock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  const auto started = Clock::now();
  (void)encoder.identity();
  const auto hashed = Clock::now();
  try {
    encoder.Upload();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("could not upload the DeepSeek V4.1 vision sidecar: ") +
        e.what());
  }
  const auto uploaded = Clock::now();
  std::size_t heap_free = 0;
  std::size_t heap_total = 0;
  (void)hipMemGetInfo(&heap_free, &heap_total);
  server::Logger::Info(
      "loader", "event=vision_prepared resident_bytes=" +
                    std::to_string(encoder.ResidentBytes()) + " identity_ms=" +
                    std::to_string(Milliseconds(hashed - started).count()) +
                    " upload_ms=" +
                    std::to_string(Milliseconds(uploaded - hashed).count()) +
                    " device_heap_free_bytes=" + std::to_string(heap_free) +
                    " " + server::Logger::MemoryStatus());
}

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateDeepSeekTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  const auto set_error = [&](std::string message) {
    if (error != nullptr) {
      *error = std::move(message);
    }
  };
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string load_error;
    auto reader_owner = core::GgufReader::OpenFile(
        options.model_path, &load_error, {.allow_unsized_tensors = true});
    if (!reader_owner) {
      set_error("Failed to open GGUF: " + load_error);
      return nullptr;
    }
    reader = std::shared_ptr<const core::GgufReader>(std::move(reader_owner));
  }
  const bool use_dspark =
      options.speculative.backend == ext::TextSpeculativeBackend::kDSpark;
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled &&
      !use_dspark) {
    set_error(
        "DeepSeek V4.1 HTTP models support only DSpark speculative "
        "decoding");
    return nullptr;
  }
  if (use_dspark && options.speculative.draft_model_path.empty()) {
    set_error("DeepSeek V4.1 DSpark HTTP decoding requires --dspark-model");
    return nullptr;
  }
  if (use_dspark && (options.speculative.max_draft_tokens == 0 ||
                     options.speculative.min_draft_tokens != 1)) {
    set_error(
        "DSpark uses model-owned adaptive drafting; it needs a positive "
        "draft limit and no custom draft floor");
    return nullptr;
  }
  if (options.session_count == 0) {
    set_error("HTTP session count must be at least one");
    return nullptr;
  }
  // Vision is opt-in: only an explicit --mmproj loads it (about 0.9 GiB). The
  // sidecar is validated before the model loads.
  std::shared_ptr<vision::Encoder> vision;
  if (!options.vision_model_path.empty()) {
    try {
      vision = std::make_shared<vision::Encoder>(options.vision_model_path);
    } catch (const std::exception& e) {
      set_error(std::string("Invalid DeepSeek V4.1 --mmproj: ") + e.what());
      return nullptr;
    }
  }
  std::string load_error;
  std::string template_warning;
  if (!ValidateGgufTemplate(*reader, &load_error, &template_warning)) {
    set_error("Unsupported DeepSeek V4.1 chat template: " + load_error);
    return nullptr;
  }
  if (!template_warning.empty()) {
    server::Logger::Warn("loader", template_warning);
  }
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    const std::uint64_t native =
        reader->GetMetadataUint64("deepseek41.context_length").value_or(0);
    if (native < 2 || native > std::numeric_limits<std::uint32_t>::max()) {
      set_error("GGUF has no valid native context length; specify --context");
      return nullptr;
    }
    max_context = static_cast<std::uint32_t>(native);
  }
  ModelOptions model_options{
      .max_context = max_context,
      .dspark_model_path =
          use_dspark ? options.speculative.draft_model_path : std::string{},
  };
  try {
    ApplyProviderOptions(options.provider_options, model_options);
  } catch (const std::invalid_argument& e) {
    set_error(e.what());
    return nullptr;
  }
  auto model = Model::Load(options.model_path, model_options, &load_error);
  if (!model) {
    set_error("Failed to create DeepSeek V4.1 model: " + load_error);
    return nullptr;
  }
  const bool disk_cache = !options.disk_cache.directory.empty();
  std::string model_fingerprint = options.disk_cache.model_artifact_fingerprint;
  std::string support_fingerprint =
      options.disk_cache.draft_model_artifact_fingerprint;
  try {
    if (disk_cache && model_fingerprint.empty()) {
      model_fingerprint = core::GgufIdentityHex(*reader);
    }
    if (disk_cache && use_dspark && support_fingerprint.empty()) {
      std::string dspark_error;
      auto dspark_reader = core::GgufReader::OpenFile(
          options.speculative.draft_model_path, &dspark_error);
      if (!dspark_reader) {
        set_error("Failed to open DSpark model: " + dspark_error);
        return nullptr;
      }
      support_fingerprint = core::GgufIdentityHex(*dspark_reader);
    }
    if (vision) {
      std::string routing_error;
      if (!model->EnableImageRouting(&routing_error)) {
        set_error(routing_error);
        return nullptr;
      }
      PrepareVision(*vision);
    }
    const std::size_t ram_share = !model->StreamsExperts() ? 0
                                  : disk_cache ? kStreamedDiskRamShareBytes
                                               : kStreamedRamShareBytes;
    const std::size_t retained_ram = !options.ram_cache.enabled ? 0
                                     : options.ram_cache.capacity_bytes != 0
                                         ? options.ram_cache.capacity_bytes
                                         : ram_share;
    return std::make_shared<DeepSeekTextRunner>(
        std::move(model), max_context, use_dspark,
        options.speculative.max_draft_tokens, std::move(model_fingerprint),
        std::move(support_fingerprint), std::move(vision), ram_share,
        retained_ram, options.session_count);
  } catch (const std::exception& e) {
    set_error(std::string("DeepSeek V4.1 load failed: ") + e.what());
    return nullptr;
  }
}

}  // namespace gufo::models::deepseek_v41_flash
