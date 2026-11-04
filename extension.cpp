#include <gufo/ext/generated/gemma4_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"

// The engine libraries build here; the text runner and the prompt, chat and
// bench drivers are not ported yet, so every build registers the probes with
// unavailable execution factories.
void gemma4_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_gemma4::kProvider_gemma4);
  registry.AddTextProvider({
      .id = "gemma4",
      .probe = &gufo::models::gemma4::ProbeGemma4,
      .probe_gguf = &gufo::models::gemma4::ProbeGemma4Gguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
  });
}

void gemma4_register(gufo::ext::Registry& registry) {
  (void)registry;
}
