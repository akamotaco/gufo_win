// Copied from Gufo src/families/qwen/kernels/verify_gemm_27b.hpp at upstream
// 1c75887b (gufo-org/gufo), moved into the gemma4 extension namespace.
#ifndef GEMMA4_LIB_QWEN_KERNELS_VERIFY_GEMM_27B_HPP_
#define GEMMA4_LIB_QWEN_KERNELS_VERIFY_GEMM_27B_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::gemma4::qwen {

/// Qwen3.8-27B verification and DFlash2 draft projections at 2..8 rows:
/// launch geometries measured faster on gfx1151 for the UD-Q4_K_XL target and
/// the Q4_K_M DFlash2 draft shapes (tools/qwen27b/verify_sweep). Every
/// geometry is the production exact small-batch kernel with another block
/// shape; outputs are bit-identical to LaunchBatchedQuantGEMMFp32. Returns
/// false when no tuned geometry applies.
[[nodiscard]] bool TryLaunchVerifyGemm27(core::GgmlType type, const void* w,
                                         const float* x, float* y,
                                         std::size_t batch, std::size_t m,
                                         std::size_t k, hipStream_t stream);

}  // namespace gufo::models::gemma4::qwen

#endif  // GEMMA4_LIB_QWEN_KERNELS_VERIFY_GEMM_27B_HPP_
