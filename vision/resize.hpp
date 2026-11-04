// Antialiased bicubic RGB8 resize, copied from the Gufo fork's feat/gemma
// src/core/image.cpp (83b764b4), where Gemma 4 had moved it from
// src/models/qwen/vision/prompt.cpp.
#ifndef GEMMA4_VISION_RESIZE_HPP_
#define GEMMA4_VISION_RESIZE_HPP_

#include <cstdint>
#include <gufo/ext/image.hpp>

namespace gufo::models::gemma4::vision {

/// Antialiased bicubic RGB8 resize, bit-exact with PyTorch/torchvision uint8
/// `resize(..., BICUBIC, antialias=True)` (separable, horizontal first).
[[nodiscard]] core::Image ResizeBicubic(const core::Image& image,
                                        std::uint32_t width,
                                        std::uint32_t height);

}  // namespace gufo::models::gemma4::vision

#endif  // GEMMA4_VISION_RESIZE_HPP_
