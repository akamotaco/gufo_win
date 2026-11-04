#ifndef GUFO_MODELS_GEMMA4_TEXT_CONTRACT_HPP_
#define GUFO_MODELS_GEMMA4_TEXT_CONTRACT_HPP_

#include <cstddef>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "vision/prompt.hpp"

namespace gufo::models::gemma4 {

/// Identity of the registered `<|tool_call>call:NAME{...}<tool_call|>`
/// dialect that the Gemma 4 runner selects.
inline constexpr std::string_view kToolDialectId = "gemma4";

/// `ToolDialect::scan_tool_calls`: appends every complete call block of
/// `text` (from the first marker on), with its span, and with `partial` the
/// one unfinished call that later text could complete. A complete block
/// whose body does not parse keeps its span but reports no call.
void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> markers,
                   std::vector<ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial);

/// The dialect the extension registers: Gemma's single call syntax with its
/// grammar, its own closer and the client envelope closers the host strips
/// after an accepted call.
[[nodiscard]] ext::ToolDialect ToolDialect();

/// The reasoning channel the model generates: `<|channel>thought\n` ...
/// `<channel|>`.
[[nodiscard]] ext::ReasoningMarkup ReasoningDelimiters();

/// The generation_config (and GGUF `general.sampling.*`) defaults: the same
/// values with and without thinking.
[[nodiscard]] sampling::TextSamplingDefaults SamplingDefaults();
/// `TextProvider::sampling_defaults`.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults>
ProbeSamplingDefaults(const core::GgufReader& reader);

/// A request's images. Their soft tokens attend each other, so prefill
/// stops before an image or after its closing token.
struct ImageContext final : ext::TextPromptContext {
  std::shared_ptr<const vision::Prompt> prompt;
  [[nodiscard]] std::size_t PrefillStop(std::size_t position,
                                        bool up) const override;
};

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_TEXT_CONTRACT_HPP_
