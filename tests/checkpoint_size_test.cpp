// The RAM cache budget one full-context checkpoint needs, computed from the
// continuation layout, against the backing plan and a measured checkpoint.
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "continuation_adapter.hpp"

namespace {

using gufo::models::deepseek_v41_flash::ContinuationAdapter;

// DeepSeek V4.1 Flash's vocabulary (the logits a checkpoint keeps).
constexpr std::size_t kVocab = 129280;

int failures = 0;

void Expect(bool ok, const char* what, std::uint32_t context, bool dspark) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s (context %u, dspark %d)\n", what, context,
                 dspark);
    ++failures;
  }
}

}  // namespace

int main() {
  for (const std::uint32_t context : {2048u, 32768u, 262144u, 1048576u}) {
    for (const bool dspark : {false, true}) {
      const auto blocks =
          ContinuationAdapter::CheckpointBlocks(context, dspark, kVocab);
      const auto bytes =
          ContinuationAdapter::FullContextBackingBytes(context, dspark, kVocab);
      std::printf("context %7u dspark %d: %zu blocks, budget %zu bytes\n",
                  context, dspark, blocks, bytes);
      Expect(ContinuationAdapter::BackingBlocks(bytes) >= blocks,
             "the budget's backing holds the checkpoint", context, dspark);
      Expect(ContinuationAdapter::BackingBlocks(bytes - 10) < blocks,
             "the budget is the smallest that holds it", context, dspark);
      if (context > 2048)
        Expect(blocks < 2 * ContinuationAdapter::CheckpointBlocks(
                                context / 2, dspark, kVocab),
               "rows grow with the context, private states do not", context,
               dspark);
    }
  }
  // Streamed UD-IQ2_XXS with DSpark published a 128,174,260-byte checkpoint
  // at 130,153 tokens. Whole blocks add slack: 2048-row chunks of tokens and
  // index scales hold 8 KiB in a 64 KiB block, about a tenth in all.
  constexpr std::size_t kMeasured = 128174260;
  const auto blocks =
      ContinuationAdapter::CheckpointBlocks(130153, true, kVocab);
  const auto paged = blocks * std::size_t{65536};
  std::printf("130153 tokens: %zu bytes in blocks, %zu measured\n", paged,
              kMeasured);
  Expect(paged >= kMeasured && paged <= kMeasured / 100 * 112,
         "blocks match the measured checkpoint", 130153, true);
  return failures ? 1 : 0;
}
