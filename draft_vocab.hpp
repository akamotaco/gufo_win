#ifndef GUFO_MODELS_GEMMA4_DRAFT_VOCAB_HPP_
#define GUFO_MODELS_GEMMA4_DRAFT_VOCAB_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "tokenizer.hpp"

namespace gufo::models::gemma4 {

/// Drafter head rows an eligible draft step scores. The `gemma4-assistant`
/// head projects onto all 262,144 tokens: as Q4_K a 151 MB read, 0.65 ms on
/// gfx1151 and about a third of a drafter step. Gemma's token ids are not
/// ordered by frequency, but the merge that creates a token is a usable
/// proxy, so the device keeps the head rows in DraftHeadOrder. On 26B
/// outputs the first 98,304 rows (plus the request's own tokens, see
/// DraftVocab) serve 92-100% of code draft cycles and
/// 55-72% of English prose cycles; other scripts fall back to the full head.
///
/// Only proposals change. The target verifies every drafted token over the
/// full vocabulary, so greedy output is unchanged, and sampled proposals are
/// corrected against the top candidates they were drawn from.
///
/// Only the 26B-A4B uses it: its ~30 ms cycle gains 4-6% on code and up to
/// 3.5% on English prose. The dense 31B's ~130 ms cycle gains nothing
/// measurable and loses up to 5% to missed proposals in other languages.
inline constexpr std::uint32_t kDraftHeadRows = 98304;

/// Per-session head rows holding the request's tokens beyond kDraftHeadRows,
/// in first-use order.
inline constexpr std::uint32_t kDraftTableRows = 1024;

/// Committed tokens the head choice looks back over.
inline constexpr std::size_t kDraftVocabWindow = 32;

// DraftVocab is copied from Gufo src/families/qwen/draft_vocab/draft_vocab.hpp
// at upstream 1c75887b (gufo-org/gufo).

/// The request's draft vocabulary and when to use it. A token is a miss when
/// it is outside the prefix, not special and not yet in the request's table
/// (or the table is full). A draft cycle uses the small head when its anchor
/// is not a miss and at most one of the last kDraftVocabWindow committed
/// tokens was, so text that keeps bringing new tokens (a short reply in
/// another script) drafts over the full vocabulary as before, a lone new
/// word does not leave the small head, and a request that repeats its own
/// vocabulary (code and comments in the user's language) gets the small
/// head. Everything is a function of the request's own token history, so
/// batched, isolated and restored sessions draft the same way.
class DraftVocab {
public:
  /// `rare_rows`: table rows left after the special tokens.
  explicit DraftVocab(std::uint32_t rare_rows = 0) : capacity_(rare_rows) {}

  void Reset() {
    rare_.clear();
    known_.clear();
    scanned_ = 0;
    last_miss_.reset();
    prior_miss_.reset();
  }

  /// Takes in `committed[scanned()..]`. Appends the newly tabled tokens to
  /// `added` in table order. `common(id)` is true for prefix and special IDs.
  template<typename Common>
  void CatchUp(std::span<const std::int32_t> committed, Common common,
               std::vector<std::int32_t>* added) {
    for (; scanned_ < committed.size(); ++scanned_) {
      const std::int32_t id = committed[scanned_];
      if (common(id) || Known(id))
        continue;
      prior_miss_ = last_miss_;
      last_miss_ = scanned_;
      if (id < 0 || rare_.size() >= capacity_)
        continue;
      if (known_.size() <= static_cast<std::size_t>(id))
        known_.resize(static_cast<std::size_t>(id) + 1, false);
      known_[static_cast<std::size_t>(id)] = true;
      rare_.push_back(id);
      if (added != nullptr)
        added->push_back(id);
    }
  }

  /// Whether a cycle anchored on `anchor`, after the scanned tokens, drafts
  /// through the small head.
  template<typename Common>
  [[nodiscard]] bool Fits(std::int32_t anchor, Common common) const {
    return (common(anchor) || Known(anchor)) && Quiet();
  }

  /// Whether the window allows the small head; Fits also needs the anchor.
  [[nodiscard]] bool Quiet() const noexcept {
    return !prior_miss_ || *prior_miss_ + kDraftVocabWindow < scanned_;
  }

  [[nodiscard]] std::size_t scanned() const noexcept { return scanned_; }
  /// Tabled rare tokens, in table order.
  [[nodiscard]] std::span<const std::int32_t> rare() const noexcept {
    return rare_;
  }

private:
  [[nodiscard]] bool Known(std::int32_t id) const {
    return id >= 0 && static_cast<std::size_t>(id) < known_.size() &&
           known_[static_cast<std::size_t>(id)];
  }

  std::uint32_t capacity_;
  std::vector<std::int32_t> rare_;
  std::vector<bool> known_;
  std::size_t scanned_{0};
  std::optional<std::size_t> last_miss_;
  std::optional<std::size_t> prior_miss_;  ///< the miss before last_miss_
};

/// The token of each drafter head row in device order: control and
/// user-defined tokens, then tokens by the rank of the earliest merge that
/// creates them, with single Latin, punctuation and space symbols ranked
/// with the first merges; every other token follows by id.
[[nodiscard]] std::vector<TokenId> DraftHeadOrder(const Tokenizer& tokenizer);

}  // namespace gufo::models::gemma4

#endif  // GUFO_MODELS_GEMMA4_DRAFT_VOCAB_HPP_
