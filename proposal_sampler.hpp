#ifndef GEMMA4_PROPOSAL_SAMPLER_HPP_
#define GEMMA4_PROPOSAL_SAMPLER_HPP_

#include <gufo/ext/sampling.hpp>
#include <utility>

namespace gufo::models::gemma4 {

/// The sampler with top-p disabled, for speculative proposals: rejection
/// sampling is exact for any proposal distribution. Built from the public
/// sampler state (config, repetition history and draw state), so the copy
/// keeps the request's temperature, top-k, min-p, repetition window and RNG.
/// The public state does not expose the generated-token counts or the
/// reasoning-budget phase, so frequency/presence penalties count the history
/// window only and the copy carries no reasoning budget. Only proposals see
/// the difference; the target's distribution is unchanged.
[[nodiscard]] inline sampling::SamplerState WithoutTopP(
    const sampling::SamplerState& sampler) {
  auto config = sampler.config();
  config.top_p = 1.0F;
  config.reasoning_budget.reset();
  sampling::SamplerState copy(std::move(config), sampler.history());
  copy.CopyDrawStateFrom(sampler);
  return copy;
}

}  // namespace gufo::models::gemma4

#endif  // GEMMA4_PROPOSAL_SAMPLER_HPP_
