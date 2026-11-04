#ifndef GUFO_MODELS_GEMMA4_RUNNER_HPP_
#define GUFO_MODELS_GEMMA4_RUNNER_HPP_

#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

namespace gufo::models::gemma4 {

/// `TextProvider::create`: the serving runner over component-cache sessions,
/// with the MTP drafter for `--speculative mtp` and the vision sidecar when
/// one matches.
std::shared_ptr<ext::TextModelRunner> CreateGemma4TextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_RUNNER_HPP_
