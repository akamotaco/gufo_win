// Host contracts of the Gemma 4 text provider: the registered tool dialect
// (buffered and streamed parsing), the reasoning delimiters, the sampling
// defaults and the image prefill stops.

#include "text_contract.hpp"

#include <algorithm>
#include <cstddef>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "tests/check.hpp"
#include "tool_syntax.hpp"

namespace g4 = gufo::models::gemma4;
namespace ext = gufo::ext;
using gemma4_test::Require;

namespace {

struct Scan {
  std::vector<ext::ScannedToolCall> calls;
  std::vector<std::pair<std::size_t, std::size_t>> spans;
};

Scan RunScan(std::string_view text, bool partial) {
  const auto dialect = g4::ToolDialect();
  Scan scan;
  dialect.scan_tool_calls(text, dialect.markers, &scan.calls, &scan.spans,
                          partial);
  return scan;
}

const ext::ScannedToolArgument* Find(const ext::ScannedToolCall& call,
                                     std::string_view name) {
  const auto it =
      std::ranges::find(call.arguments, name, &ext::ScannedToolArgument::name);
  return it == call.arguments.end() ? nullptr : &*it;
}

void CheckDialect() {
  const auto dialect = g4::ToolDialect();
  Require(dialect.id == g4::kToolDialectId && dialect.id == "gemma4",
          "dialect identity");
  Require(dialect.markers.size() == 1 && dialect.markers[0] == "<|tool_call>",
          "one call marker");
  // RegisterToolDialect's conditions.
  Require(std::ranges::all_of(dialect.markers,
                              [](std::string_view marker) {
                                return marker.find('<') != marker.npos;
                              }),
          "markers contain '<'");
  Require(dialect.scan_tool_calls != nullptr, "scanner");
  Require(dialect.syntax.has_value() &&
              std::ranges::find(dialect.markers,
                                std::string_view(dialect.syntax->open)) !=
                  dialect.markers.end(),
          "the grammar opens with the marker");
  Require(dialect.syntax->open == g4::ToolGrammarSyntax().open &&
              dialect.syntax->close == g4::ToolGrammarSyntax().close,
          "grammar syntax");
  Require(dialect.reasoning_markers.empty(), "no reasoning header");
  Require(dialect.closers == std::vector<std::string_view>{"<tool_call|>",
                                                           "</invoke>",
                                                           "</parameter>"},
          "closers");
}

void CheckBuffered() {
  const std::string first =
      "<|tool_call>call:get_weather{days:3,location:<|\"|>Paris, "
      "FR<|\"|>,units:{c:true}}<tool_call|>";
  const std::string second = "<|tool_call>call:noop{}<tool_call|>";
  const std::string text = first + second + " trailing";
  const auto scan = RunScan(text, false);
  Require(scan.calls.size() == 2, "two calls");
  const auto& call = scan.calls[0];
  Require(call.name == "get_weather" && call.valid && call.complete &&
              !call.unfinished,
          "complete call");
  const auto* location = Find(call, "location");
  const auto* days = Find(call, "days");
  const auto* units = Find(call, "units");
  Require(location && location->is_string && location->value == "Paris, FR",
          "string argument");
  Require(days && !days->is_string && days->value == "3", "number argument");
  Require(units && !units->is_string && units->value == R"({"c":true})",
          "object argument as JSON");
  Require(scan.calls[1].name == "noop" && scan.calls[1].arguments.empty(),
          "call without arguments");
  // Numbers keep the model's spelling, nested ones too.
  const auto numbers = RunScan(
      "<|tool_call>call:tag{id:9007199254740993,w:1.0,v:[2.50]}<tool_call|>",
      false);
  Require(numbers.calls.size() == 1, "number call");
  const auto* id = Find(numbers.calls[0], "id");
  const auto* weight = Find(numbers.calls[0], "w");
  const auto* values = Find(numbers.calls[0], "v");
  Require(id && id->value == "9007199254740993" && weight &&
              weight->value == "1.0" && values && values->value == "[2.50]",
          "number spelling");
  Require(
      scan.spans ==
          std::vector<std::pair<std::size_t, std::size_t>>{
              {0, first.size()}, {first.size(), first.size() + second.size()}},
      "spans cover each call block");

  // A malformed block keeps its span and reports no call; later calls stay.
  const std::string broken = "<|tool_call>call:f{x:<tool_call|>";
  const auto mixed = RunScan(broken + second, false);
  Require(mixed.calls.size() == 1 && mixed.calls[0].name == "noop",
          "a malformed call is skipped");
  Require(mixed.spans.size() == 2 && mixed.spans[0].second == broken.size(),
          "a malformed call keeps its span");

  // Without a partial parse an unfinished call is never reported.
  const auto unfinished = RunScan(second + "<|tool_call>call:f{x:1", false);
  Require(unfinished.calls.size() == 1 && unfinished.spans.size() == 1,
          "an unfinished call needs a partial parse");
}

void CheckStreamed() {
  const std::string text =
      "<|tool_call>call:write{path:<|\"|>a.txt<|\"|>,mode:1,"
      "body:<|\"|>hello, world<|\"|>}<tool_call|>";
  const auto final_scan = RunScan(text, false);
  Require(final_scan.calls.size() == 1, "final call");
  std::size_t settled = 0;
  std::string open_text;
  bool named = false;
  for (std::size_t n = 1; n < text.size(); ++n) {
    const auto scan = RunScan(std::string_view(text).substr(0, n), true);
    Require(scan.calls.size() <= 1, "one call at a time");
    if (scan.calls.empty()) {
      Require(!named, "a named call stays reported");
      continue;
    }
    const auto& call = scan.calls[0];
    named = true;
    Require(call.name == "write" && call.valid, "partial call name");
    Require(call.arguments.size() >= settled, "settled arguments only grow");
    for (std::size_t i = 0; i < call.arguments.size(); ++i) {
      Require(
          call.arguments[i].name == final_scan.calls[0].arguments[i].name &&
              call.arguments[i].value == final_scan.calls[0].arguments[i].value,
          "a settled argument never changes");
    }
    if (call.arguments.size() != settled) {
      open_text.clear();
    }
    settled = call.arguments.size();
    if (!call.complete && call.unfinished) {
      Require(call.unfinished->is_string, "an open argument is a string");
      Require(call.unfinished->value.starts_with(open_text),
              "an open string only grows");
      open_text = call.unfinished->value;
      Require(std::string_view("hello, world").starts_with(open_text) ||
                  std::string_view("a.txt").starts_with(open_text),
              "an open string never shows its delimiter");
    }
    if (!call.complete) {
      Require(scan.spans.empty(), "an unfinished call has no span");
    }
  }
  Require(named && settled == 3, "every argument settled while streaming");
}

void CheckReasoning() {
  const auto markup = g4::ReasoningDelimiters();
  Require(markup.open == "<|channel>thought\n" && markup.close == "<channel|>",
          "reasoning delimiters");
  Require(markup.open == g4::kThoughtStart && markup.close == g4::kThoughtEnd,
          "delimiters follow the chat template");
}

void CheckSampling() {
  const auto defaults = g4::SamplingDefaults();
  for (const std::optional<bool> thinking :
       {std::optional<bool>{}, std::optional<bool>{true},
        std::optional<bool>{false}}) {
    const auto& values = defaults.For(thinking);
    Require(values.temperature == 1.0F && values.top_k == 64 &&
                values.top_p == 0.95F,
            "generation_config defaults");
    Require(!values.min_p && !values.min_keep && !values.repeat_penalty &&
                !values.repeat_last_n && !values.frequency_penalty &&
                !values.presence_penalty,
            "no other defaults");
    const auto resolved = gufo::sampling::ResolveTextSampling(
        defaults, thinking, gufo::sampling::SamplingConfig{}, {});
    Require(resolved.temperature == 1.0F && resolved.top_k == 64 &&
                resolved.top_p == 0.95F,
            "resolved defaults");
  }
  gufo::sampling::SamplingConfig configured;
  configured.top_k = 5;
  gufo::sampling::SamplingOverrides supplied;
  supplied.top_k = true;
  const auto resolved = gufo::sampling::ResolveTextSampling(
      defaults, std::nullopt, configured, supplied);
  Require(resolved.top_k == 5 && resolved.temperature == 1.0F,
          "explicit settings override the defaults");
}

void CheckPrefillStops() {
  auto prompt = std::make_shared<g4::vision::Prompt>();
  // Image soft tokens [10, 15) closed at 15, and [30, 33) closed at 33.
  for (const auto [offset, rows] : {std::pair{10U, 5U}, std::pair{30U, 3U}}) {
    g4::vision::PreparedImage image;
    image.span = {offset, rows};
    prompt->images.push_back(std::move(image));
  }
  g4::ImageContext context;
  Require(context.PrefillStop(12, false) == 12 &&
              context.PrefillStop(12, true) == 12,
          "no images declare no spans");
  context.prompt = prompt;
  struct Case {
    std::size_t position, down, up;
  };
  for (const auto& c : std::initializer_list<Case>{{0, 0, 0},
                                                   {9, 9, 9},
                                                   {10, 10, 10},
                                                   {11, 10, 16},
                                                   {14, 10, 16},
                                                   {15, 10, 16},
                                                   {16, 16, 16},
                                                   {29, 29, 29},
                                                   {30, 30, 30},
                                                   {31, 30, 34},
                                                   {33, 30, 34},
                                                   {34, 34, 34},
                                                   {100, 100, 100}}) {
    const auto down = context.PrefillStop(c.position, false);
    const auto up = context.PrefillStop(c.position, true);
    Require(down == c.down && up == c.up,
            "prefill stop at " + std::to_string(c.position));
    Require(context.PrefillStop(down, false) == down &&
                context.PrefillStop(down, true) == down &&
                context.PrefillStop(up, false) == up &&
                context.PrefillStop(up, true) == up,
            "stops are fixed points at " + std::to_string(c.position));
  }
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    CheckDialect();
    CheckBuffered();
    CheckStreamed();
    CheckReasoning();
    CheckSampling();
    CheckPrefillStops();
  });
}
