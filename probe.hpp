#ifndef GEMMA4_PROBE_HPP_
#define GEMMA4_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::gemma4 {

/// A `gemma4` target GGUF (31B, 31B QAT or 26B-A4B). The `gemma4-assistant`
/// MTP drafter and the vision sidecar are artifacts of their target, not
/// models of their own, so they do not match.
[[nodiscard]] bool ProbeGemma4Gguf(const core::GgufReader& reader);
/// `model_root` is a target GGUF file.
[[nodiscard]] bool ProbeGemma4(std::string_view model_root);

}  // namespace gufo::models::gemma4

#endif  // GEMMA4_PROBE_HPP_
