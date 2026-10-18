#ifndef GUFO_MODELS_GEMMA4_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_GEMMA4_CONTINUATION_ADAPTER_HPP_

#include <hip/hip_runtime.h>

#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/models/gemma4/engine.hpp"

namespace gufo::models::gemma4 {

// One continuation's state, by component (ID = index + 1).
enum class ContinuationPart : std::uint8_t {
  kTokens,
  kKey,
  kValue,
  kLogits,
  kHidden,
  kMetadata,
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};
struct ContinuationBackingClass {
  std::size_t block_bytes{}, blocks{}, metadata_bytes{};
};

// Serves Gemma 4 sessions to the component cache. Tokens and global-layer
// keys and values are append rows; sliding-layer keys and values are window
// rows held in the session's ring; the frontier logits, its hidden state (the
// MTP drafter's input) and metadata are private. The drafter reads the
// target's cache, so speculative decoding adds no components. The caller
// supplies the artifact and template identity; the adapter adds its layout.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model>, std::uint32_t context,
                      cache::Identity);
  ~ContinuationAdapter() override;
  [[nodiscard]] Session& GetSession(cache::Slot&);
  // Orders `stream` after the work queued so far on the model's executor, so
  // reads of rows and frontier state see completed writes.
  void AfterExecution(hipStream_t stream) const;
  [[nodiscard]] std::vector<ContinuationBackingClass> BackingPlan(
      std::size_t ram_bytes) const;
  [[nodiscard]] std::span<const ContinuationComponent> Layout() const {
    return components_;
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] std::vector<cache::ResidentRange> ResidentRows(
      const cache::Slot&,
      std::span<const cache::ComponentPosition>) const override;
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>,
                    std::span<const cache::ResidentRange> kept) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(
      const cache::Slot&, cache::ComponentId, cache::Rows,
      cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Piece {
    std::byte* pointer;
    std::size_t bytes;
  };
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  // Device memory of rows [first, end) of a key or value component: one
  // piece, or two where a window range wraps the session's ring.
  [[nodiscard]] std::vector<Piece> RowPieces(State&,
                                             const ContinuationComponent&,
                                             cache::Rows first,
                                             cache::Rows end) const;
  // The private component's whole representation: device memory, or host
  // memory (`host`) the adapter copies on the calling thread.
  [[nodiscard]] Piece PrivateRegion(State&, const ContinuationComponent&,
                                    bool loading, bool* host) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(hipStream_t)>&);
  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::uint32_t ring_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  mutable std::mutex executed_mutex_;
  hipEvent_t executed_{};
};
}  // namespace gufo::models::gemma4
#endif
