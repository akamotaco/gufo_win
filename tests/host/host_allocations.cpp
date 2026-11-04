// Copied verbatim from gufo src/core/hip/host_allocations.cpp at a8844cc6977b4c52349b1622f1b3f02e31fead25 (blob 20ccf7930481); only the includes are rewritten.
// Regenerate with tools/sync_test_host.py; do not edit.
#include <gufo/ext/experimental/host_allocations.hpp>

#include "tests/host/page_pin.hpp"

namespace gufo::hip {

HostAllocations::~HostAllocations() {
  Reset();
}

hipError_t HostAllocations::Allocate(void** pointer, std::size_t bytes,
                                     unsigned flags) {
  if (!pointer)
    return hipErrorInvalidValue;
  *pointer = nullptr;
  void* data{};
  const auto status = hipHostMalloc(&data, bytes, flags);
  if (status != hipSuccess)
    return status;
  if (!data)
    return hipSuccess;
  const int pin = platform::PinHostPages(data, bytes);
  try {
    allocations_.push_back({data, bytes, pin});
  } catch (const std::bad_alloc&) {
    platform::UnpinHostPages(pin);
    (void)hipHostFree(data);
    return hipErrorOutOfMemory;
  }
  *pointer = data;
  return hipSuccess;
}

void HostAllocations::Reset() noexcept {
  for (const auto& allocation : allocations_) {
    platform::UnpinHostPages(allocation.pin);
    (void)hipHostFree(allocation.data);
  }
  allocations_.clear();
}

std::size_t HostAllocations::PinnedBytes() const noexcept {
  std::size_t bytes{};
  for (const auto& allocation : allocations_)
    if (allocation.pin >= 0)
      bytes += allocation.bytes;
  return bytes;
}

}  // namespace gufo::hip
