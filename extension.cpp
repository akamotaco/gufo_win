#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "text_contract.hpp"

// The tool dialect is host-side and registers in every build, before any
// runner is created. HIP builds register the provider from gufo_gemma4_text
// (provider.cpp); without HIP the probes register with no execution
// factories.
void gemma4_register(gufo::ext::Registry& registry) {
  gufo::ext::RegisterToolDialect(gufo::models::gemma4::ToolDialect());
#if !defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "gemma4",
      .probe = &gufo::models::gemma4::ProbeGemma4,
      .probe_gguf = &gufo::models::gemma4::ProbeGemma4Gguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
      .sampling_defaults = &gufo::models::gemma4::ProbeSamplingDefaults,
  });
#else
  (void)registry;
#endif
}
