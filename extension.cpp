#include <gufo/ext/generated/gemma4_constants.hpp>
#include <gufo/ext/registry.hpp>

// Metadata-only registration: the text provider, its probe and runner are
// added while the engine moves here from the fork's feat/gemma branch.
void gemma4_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_gemma4::kProvider_gemma4);
}

void gemma4_register(gufo::ext::Registry& registry) { (void)registry; }
