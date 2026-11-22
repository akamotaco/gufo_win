#include <chrono>
#include <gufo/ext/generated/gemma4_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <memory>

#include "probe.hpp"
#include "runner.hpp"
#include "text_contract.hpp"

namespace gufo::models::gemma4 {

int RunGemma4Bench(const ext::TextBenchOptions& options,
                   const std::shared_ptr<const core::GgufReader>& reader,
                   std::chrono::steady_clock::time_point load_start);

int RunGemma4Prompt(const ext::TextPromptOptions& options,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point load_start);

int RunGemma4Chat(const ext::TextPromptOptions& options,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::gemma4

void gemma4_provider_register(gufo::ext::Registry& registry) {
  namespace g4 = gufo::models::gemma4;
  registry.AddProvider(
      gufo::ext::generated::extension_gemma4::kProvider_gemma4);
  registry.AddTextProvider({
      .id = "gemma4",
      .probe = &g4::ProbeGemma4,
      .probe_gguf = &g4::ProbeGemma4Gguf,
      .create = &g4::CreateGemma4TextRunner,
      .bench_driver = &g4::RunGemma4Bench,
      .prompt_driver = &g4::RunGemma4Prompt,
      .chat_driver = &g4::RunGemma4Chat,
      .sampling_defaults = &g4::ProbeSamplingDefaults,
      .speculative_defaults = {{.backend = "mtp",
                                .max_draft_tokens = 15,
                                .min_draft_tokens = 1,
                                .draft_policy = "calibrated"}},
  });
}
