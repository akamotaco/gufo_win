#include "text_contract.hpp"

#include <string>

#include "chat_template.hpp"
#include "tool_syntax.hpp"

namespace gufo::models::gemma4 {
namespace {

void AddArguments(const json::Value& arguments,
                  std::vector<ext::ScannedToolArgument>* out) {
  // Non-string values keep the model's number spelling (Gufo writes tool-call
  // numbers as parsed).
  for (const auto& [key, value] : arguments.members()) {
    out->push_back({.name = key,
                    .value = value.is_string()
                                 ? value.str()
                                 : value.dump(json::Value::Numbers::kAsParsed),
                    .is_string = value.is_string()});
  }
}

}  // namespace

void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> markers,
                   std::vector<ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial) {
  // Gemma 4 has a single call syntax; its opener is the only marker.
  const std::string_view start =
      markers.empty() ? kToolCallStart : markers.front();
  std::size_t cursor = 0;
  while (true) {
    const std::size_t open = text.find(start, cursor);
    if (open == std::string_view::npos) {
      return;
    }
    const std::size_t body = open + start.size();
    const std::size_t close = text.find(kToolCallEnd, body);
    if (close == std::string_view::npos) {
      // Without a partial parse an interrupted call is never reported.
      if (!partial) {
        return;
      }
      const auto call = ParsePartialToolCall(text.substr(body));
      if (call) {
        ext::ScannedToolCall scanned{.name = call->name, .valid = true};
        AddArguments(call->arguments, &scanned.arguments);
        if (call->open_string) {
          scanned.unfinished =
              ext::ScannedToolArgument{.name = call->open_string->first,
                                       .value = call->open_string->second,
                                       .is_string = true};
        }
        calls->push_back(std::move(scanned));
      }
      return;
    }
    const std::size_t end = close + kToolCallEnd.size();
    if (spans != nullptr) {
      spans->emplace_back(open, end);
    }
    if (auto call = ParseToolCall(text.substr(body, close - body))) {
      ext::ScannedToolCall scanned{
          .name = std::move(call->name), .valid = true, .complete = true};
      AddArguments(call->arguments, &scanned.arguments);
      calls->push_back(std::move(scanned));
    }
    cursor = end;
  }
}

ext::ToolDialect ToolDialect() {
  return {
      .markers = {kToolCallStart},
      // The grammar has no reasoning header: a call opens after the thought
      // channel closes.
      .reasoning_markers = {},
      // Gemma's own closer and the client's envelope; Qwen and DSML closers
      // are prose under its single call syntax.
      .closers = {kToolCallEnd, "</invoke>", "</parameter>"},
      .scan_tool_calls = &ScanToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = ToolGrammarSyntax(),
  };
}

ext::ReasoningMarkup ReasoningDelimiters() {
  return {std::string(kThoughtStart), std::string(kThoughtEnd)};
}

sampling::TextSamplingDefaults SamplingDefaults() {
  sampling::TextSamplingValues values;
  values.temperature = 1.0F;
  values.top_k = 64;
  values.top_p = 0.95F;
  return {values, values, values};
}

std::optional<sampling::TextSamplingDefaults> ProbeSamplingDefaults(
    const core::GgufReader&) {
  return SamplingDefaults();
}

std::size_t ImageContext::PrefillStop(std::size_t position, bool up) const {
  if (prompt) {
    for (const auto& image : prompt->images) {
      const std::size_t first = image.span.offset;
      const std::size_t last = first + image.span.rows;  // closing token
      if (position > first && position <= last) {
        return up ? last + 1 : first;
      }
    }
  }
  return position;
}

}  // namespace gufo::models::gemma4
