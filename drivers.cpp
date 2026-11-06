// `gufo prompt`, `gufo chat` and `gufo bench` for Gemma 4 on one session,
// outside the serving pool.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/image.hpp>
#include <gufo/ext/text.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "chat_template.hpp"
#include "engine.hpp"
#include "provider_options.hpp"
#include "vision/encoder.hpp"
#include "vision/prompt.hpp"

namespace gufo::models::gemma4 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kDefaultContext = 4096;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void PrintModelLoadTime(Clock::time_point start, bool success = true) {
  auto& output = success ? std::cout : std::cerr;
  output << "[Model Load]: " << SecondsSince(start) << " s";
  if (!success) {
    output << " (failed)";
  }
  output << '\n';
}

std::optional<ReasoningEffort> ParseEffort(std::string_view name) {
  if (name == "minimal")
    return ReasoningEffort::kMinimal;
  if (name == "low")
    return ReasoningEffort::kLow;
  if (name == "medium")
    return ReasoningEffort::kMedium;
  if (name == "high")
    return ReasoningEffort::kHigh;
  if (name == "xhigh")
    return ReasoningEffort::kXHigh;
  if (name == "max")
    return ReasoningEffort::kMax;
  return std::nullopt;
}

// `--think`, `--reasoning-effort` and `--preserve-thinking` as the HTTP
// request fields would set them.
ReasoningOptions PromptReasoning(const ext::TextPromptOptions& opt) {
  ReasoningOptions reasoning;
  if (opt.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (opt.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (opt.reasoning_effort != "auto") {
    reasoning.effort = ParseEffort(opt.reasoning_effort);
    if (!reasoning.effort) {
      throw std::invalid_argument("unknown reasoning effort '" +
                                  opt.reasoning_effort + "'");
    }
    reasoning.enabled = true;
  }
  if (opt.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (opt.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

std::shared_ptr<Model> LoadModel(const ext::TextPromptOptions& opt,
                                 Clock::time_point start) {
  if (opt.force_cpu) {
    std::cerr << "Gemma 4 is supported only by the ROCm backend\n";
    return nullptr;
  }
  const bool mtp = opt.speculative_backend == "mtp";
  if (!opt.speculative_backend.empty() && !mtp) {
    std::cerr << "Gemma 4 supports only --speculative mtp\n";
    return nullptr;
  }
  if (mtp && opt.mtp_model_path.empty()) {
    std::cerr << "--speculative mtp requires --mtp-model\n";
    return nullptr;
  }
  ModelOptions options{.max_context = kDefaultContext,
                       .mtp_model_path = mtp ? opt.mtp_model_path : "",
                       .draft_tokens = opt.draft_tokens,
                       .min_draft_tokens = opt.min_draft_tokens};
  try {
    options.draft_policy = ParseDraftPolicy(opt.draft_policy);
    ApplyProviderOptions(opt.provider_options, options);
    (void)ImageTokens(opt.provider_options);
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return nullptr;
  }
  std::string error;
  auto model = Model::Load(opt.model_path, options, &error);
  if (model == nullptr) {
    std::cerr << "Error loading Gemma 4 model: " << error << '\n';
    PrintModelLoadTime(start, false);
    return nullptr;
  }
  PrintModelLoadTime(start);
  return model;
}

/// The vision sidecar, when images are attached or --mmproj is given.
std::shared_ptr<vision::Encoder> LoadVision(const ext::TextPromptOptions& opt,
                                            const Model& model) {
  if (opt.image_paths.empty() && opt.vision_model_path.empty()) {
    return {};
  }
  auto encoder = vision::Encoder::Open(opt.model_path, opt.vision_model_path,
                                       model.config().hidden_size);
  if (!encoder) {
    throw std::invalid_argument(
        "image input requires a matching --mmproj BF16 sidecar");
  }
  return encoder;
}

void AttachImages(const ext::TextPromptOptions& opt,
                  ext::ChatMessage& message) {
  for (const auto& path : opt.image_paths) {
    message.images.push_back(
        {0, std::make_shared<const std::vector<std::uint8_t>>(
                core::ReadImageFile(path))});
  }
}

/// A rendered conversation with its images, encoded on first use.
struct Prompt {
  std::vector<TokenId> tokens;
  std::vector<ImageSpan> spans;
  std::shared_ptr<const vision::Prompt> vision;
};

Prompt RenderPrompt(const Model& model, vision::Encoder* encoder,
                    std::span<const ext::ChatMessage> history,
                    const ChatOptions& options, std::uint32_t image_tokens) {
  auto prepared = std::make_shared<const vision::Prompt>(
      vision::Prepare(model.tokenizer(), history, {}, options,
                      encoder != nullptr ? encoder->identity() : "",
                      model.MaxContext(), image_tokens));
  Prompt prompt{prepared->tokens, {}, prepared};
  for (const auto& image : prepared->images) {
    prompt.spans.push_back(
        {image.span.offset, image.span.rows, image.prefix_identity});
  }
  return prompt;
}

/// Generates one reply after `prompt`; `reply` receives the decoded text.
int Generate(const ext::TextPromptOptions& opt, Model& model, Session& session,
             vision::Encoder* encoder, const Prompt& input,
             std::string* reply) {
  const std::span<const TokenId> prompt = input.tokens;
  if (prompt.empty()) {
    std::cerr << "Gemma 4 prompt produced no tokens\n";
    return 1;
  }
  if (prompt.size() + opt.max_tokens > session.ContextSize()) {
    std::cerr << "Gemma 4 prompt and output exceed the "
              << session.ContextSize() << "-token CLI context\n";
    return 1;
  }
  std::string error;
  const auto prefill_start = Clock::now();
  std::map<std::size_t, std::shared_ptr<const vision::Encoder::Embedding>>
      embeddings;
  const auto embed = [&](std::size_t index) -> const float* {
    auto& embedding = embeddings[index];
    if (!embedding) {
      embedding = encoder->Encode(input.vision->images.at(index).pixels);
    }
    return embedding->data();
  };
  if (!session.Sync(prompt, input.spans, embed, &error)) {
    std::cerr << "Gemma 4 prefill failed: " << error << '\n';
    return 1;
  }
  const double prefill_seconds = SecondsSince(prefill_start);
  if (opt.verbose) {
    std::cout << "[Engine]: Gemma 4 ROCm (gfx1151)\n"
              << "Model: " << model.ModelName() << '\n'
              << "Prompt tokens: " << prompt.size() << '\n'
              << "--- Generation Output ---\n";
  }
  std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(opt.sampling, history);
  const auto decode_start = Clock::now();
  std::size_t generated = 0;
  bool stop = false;
  while (generated < opt.max_tokens && !stop) {
    Session::DecodeResult step;
    if (!session.DecodeStep(opt.max_tokens - generated, sampler, &step,
                            &error)) {
      std::cerr << "\nGemma 4 decode failed: " << error << '\n';
      return 1;
    }
    for (const TokenId token : step.tokens) {
      if (model.IsStopToken(token)) {
        stop = true;
        break;
      }
      const std::string piece = model.TokenText(token);
      if (reply != nullptr) {
        reply->append(piece);
      }
      std::cout << piece << std::flush;
      ++generated;
    }
    stop = stop || step.stop || step.tokens.empty();
  }
  std::cout << '\n';
  if (opt.verbose && generated > 0) {
    const double seconds = SecondsSince(decode_start);
    std::cout << "Prefill " << prompt.size() << " tokens in " << prefill_seconds
              << " s (" << static_cast<double>(prompt.size()) / prefill_seconds
              << " tok/s)\nGenerated " << generated << " tokens in " << seconds
              << " s (" << static_cast<double>(generated) / seconds
              << " tok/s)\n";
    if (model.HasMtp()) {
      const auto& stats = session.Statistics();
      std::cout << "[Speculative]: cycles=" << stats.cycles
                << " drafted=" << stats.drafted
                << " accepted=" << stats.accepted << " acceptance="
                << (stats.drafted != 0 ? static_cast<double>(stats.accepted) /
                                             static_cast<double>(stats.drafted)
                                       : 0.0)
                << " lookup_proposed=" << stats.copied
                << " lookup_accepted=" << stats.copied_accepted
                << " siblings=" << stats.siblings
                << " siblings_accepted=" << stats.siblings_accepted << '\n';
    }
  }
  return 0;
}

struct BenchStats {
  double mean{0};
  double stddev{0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) {
    return {};
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  if (values.size() <= 1) {
    return {.mean = mean, .stddev = 0.0};
  }
  double sq_sum = 0.0;
  for (const double v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

}  // namespace

int RunGemma4Prompt(const ext::TextPromptOptions& opt, const core::GgufReader&,
                    Clock::time_point load_start) {
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma 4 session creation failed: " << error << '\n';
    return 1;
  }
  try {
    const auto encoder = LoadVision(opt, *model);
    Prompt prompt;
    if (opt.use_chat_template) {
      std::vector<ext::ChatMessage> messages;
      if (!opt.system_prompt.empty()) {
        messages.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
      }
      messages.emplace_back(ext::ChatRole::kUser, opt.prompt_text);
      AttachImages(opt, messages.back());
      prompt = RenderPrompt(*model, encoder.get(), messages,
                            ResolveChatOptions(PromptReasoning(opt)),
                            ImageTokens(opt.provider_options));
    } else if (!opt.image_paths.empty()) {
      throw std::invalid_argument("--image requires chat framing");
    } else {
      prompt.tokens = model->tokenizer().Encode(opt.prompt_text, true, true);
    }
    return Generate(opt, *model, *session, encoder.get(), prompt, nullptr);
  } catch (const std::exception& e) {
    std::cerr << "Gemma 4 prompt failed: " << e.what() << '\n';
    return 1;
  }
}

int RunGemma4Chat(const ext::TextPromptOptions& opt, const core::GgufReader&,
                  Clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Gemma 4 interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt, load_start);
  if (!model) {
    return 1;
  }
  std::string error;
  auto session = model->CreateSession(kDefaultContext, &error);
  if (!session) {
    std::cerr << "Gemma 4 session creation failed: " << error << '\n';
    return 1;
  }
  std::shared_ptr<vision::Encoder> encoder;
  ChatOptions chat_options;
  try {
    encoder = LoadVision(opt, *model);
    chat_options = ResolveChatOptions(PromptReasoning(opt));
  } catch (const std::exception& e) {
    std::cerr << "Gemma 4 chat failed: " << e.what() << '\n';
    return 1;
  }
  std::vector<ext::ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
  }
  std::cout << "=== Gufo Interactive Chat (Gemma 4) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit") {
      break;
    }
    if (input.empty()) {
      continue;
    }
    history.emplace_back(ext::ChatRole::kUser, input);
    if (history.size() == 1 ||
        (history.size() == 2 && history[0].role == ext::ChatRole::kSystem)) {
      try {
        AttachImages(opt, history.back());
      } catch (const std::exception& e) {
        std::cerr << "Gemma 4 image failed: " << e.what() << '\n';
        return 1;
      }
    }
    Prompt prompt;
    try {
      prompt = RenderPrompt(*model, encoder.get(), history, chat_options,
                            ImageTokens(opt.provider_options));
    } catch (const std::exception& e) {
      std::cerr << "Gemma 4 chat template failed: " << e.what() << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (Generate(opt, *model, *session, encoder.get(), prompt, &reply) != 0) {
      return 1;
    }
    // The template drops thinking channels from model history itself.
    history.emplace_back(ext::ChatRole::kAssistant, std::move(reply));
    std::cout << '\n';
  }
  return 0;
}

/// llama-bench style pp/tg: prefill of `n_prompts` tokens after `n_depths`
/// cached tokens, and generation of `n_gens` tokens.
int RunGemma4Bench(const ext::TextBenchOptions& options,
                   const std::shared_ptr<const core::GgufReader>& reader,
                   Clock::time_point model_load_start) {
  const bool mtp = options.speculative_backend == "mtp";
  if (!options.speculative_backend.empty() && !mtp) {
    std::cerr << "Error: Gemma 4 supports only --speculative mtp or off\n";
    return 1;
  }
  if (mtp && options.mtp_model_path.empty()) {
    std::cerr << "Error: --speculative mtp requires --mtp-model\n";
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Gemma 4 bench supports C1; use the serving benchmark "
                 "for concurrent requests\n";
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t required_context = std::max(
      {std::size_t{4096}, max_depth + max_or_zero(options.n_prompts) + 1,
       std::max<std::size_t>(max_depth, 16) + max_or_zero(options.n_gens) + 1});
  if (required_context > std::numeric_limits<std::uint32_t>::max()) {
    std::cerr << "Error: Gemma 4 benchmark context is out of range\n";
    return 1;
  }
  ModelOptions model_options{
      .max_context = static_cast<std::uint32_t>(required_context),
      .mtp_model_path = mtp ? options.mtp_model_path : "",
      .draft_tokens = options.draft_tokens,
      .min_draft_tokens = options.min_draft_tokens};
  try {
    model_options.draft_policy = ParseDraftPolicy(options.draft_policy);
    ApplyProviderOptions(options.provider_options, model_options);
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  std::string error;
  auto model = Model::Load(options.model_path, model_options, &error);
  if (model == nullptr) {
    std::cerr << "Error creating Gemma 4 model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  std::vector<TokenId> tokens;
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    tokens.resize(required_context);
    tokens[0] = model->tokenizer().BosToken();
    for (std::size_t i = 1; i < tokens.size(); ++i) {
      tokens[i] = pattern[(i - 1) % pattern.size()];
    }
  }

  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  const auto model_name = model->ModelName();
  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model_name << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };
  const auto context = static_cast<std::uint32_t>(required_context);
  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      std::vector<double> runs;
      // The first repetition is an untimed warm-up of the chunk shapes.
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(context, &error);
        if (!session ||
            (depth > 0 &&
             !session->Sync(std::span(tokens).first(depth), &error))) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = Clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length),
                           &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = SecondsSince(start);
        if (repetition > 0) {
          runs.push_back(static_cast<double>(prompt_length) / seconds);
        }
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }
    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(context, &error);
        if (!session ||
            !session->Sync(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        const std::vector<sampling::TokenId> history(
            tokens.begin(),
            tokens.begin() + static_cast<std::ptrdiff_t>(prefix_length));
        sampling::SamplerState sampler(options.sampling, history);
        std::vector<TokenId> generated;
        const auto start = Clock::now();
        while (generated.size() < generation_length) {
          Session::DecodeResult step;
          if (!session->DecodeStep(generation_length - generated.size(),
                                   sampler, &step, &error, false) ||
              step.tokens.empty()) {
            std::cerr << "Error running Gemma 4 decode: " << error << '\n';
            return 1;
          }
          generated.insert(generated.end(), step.tokens.begin(),
                           step.tokens.end());
        }
        const double seconds = SecondsSince(start);
        runs.push_back(static_cast<double>(generation_length) / seconds);
        if (options.verbose) {
          const auto& stats = session->Statistics();
          std::cerr << "Gemma 4 tg depth=" << depth
                    << " cycles=" << stats.cycles
                    << " drafted=" << stats.drafted
                    << " accepted=" << stats.accepted
                    << " lookup_proposed=" << stats.copied
                    << " lookup_accepted=" << stats.copied_accepted << " text="
                    << model->Decode(std::span(generated).first(
                           std::min<std::size_t>(generated.size(), 48)))
                    << '\n';
        }
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

}  // namespace gufo::models::gemma4
