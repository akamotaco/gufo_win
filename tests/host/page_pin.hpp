// Copied verbatim from gufo src/core/platform/page_pin.hpp at a8844cc6977b4c52349b1622f1b3f02e31fead25 (blob 8f3acd7c0f2a); only the includes are rewritten.
// Regenerate with tools/sync_test_host.py; do not edit.
#ifndef GUFO_CORE_PLATFORM_PAGE_PIN_HPP_
#define GUFO_CORE_PLATFORM_PAGE_PIN_HPP_

#include <cstddef>

namespace gufo::platform {

/// Pins host pages for DMA, reducing request-side page faults. Returns a
/// non-negative pin token, or -1 when pinning is unavailable. The pin owns
/// no memory; callers keep the allocation alive until UnpinHostPages().
int PinHostPages(void* data, std::size_t bytes) noexcept;

/// Releases a pin from PinHostPages(). A negative token is ignored.
void UnpinHostPages(int pin) noexcept;

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PAGE_PIN_HPP_
