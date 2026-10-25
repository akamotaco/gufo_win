#ifndef GUFO_CORE_PLATFORM_DESCRIPTOR_HPP_
#define GUFO_CORE_PLATFORM_DESCRIPTOR_HPP_

// Bridges between PlatformFile and the int descriptors of code that still
// uses POSIX calls. On Linux both are the same fd and nothing is copied. On
// Windows PlatformFile is a HANDLE and descriptors come from compat/win32.

#include "src/core/platform/mapped_file.hpp"

#ifdef _WIN32
#include <unistd.h>
#endif

namespace gufo::platform {

/// The PlatformFile behind descriptor `fd`, borrowed: `fd` keeps owning it.
inline PlatformFile FileFromDescriptor(int fd) noexcept {
#ifdef _WIN32
  return fd < 0 ? kInvalidFile : gufo_handle_of_descriptor(fd);
#else
  return fd;
#endif
}

/// A descriptor for `file` that lives as long as this object. On Linux it is
/// `file` itself; on Windows a descriptor over a duplicate handle, closed on
/// destruction.
class DescriptorView {
public:
  explicit DescriptorView(PlatformFile file) noexcept
#ifdef _WIN32
      : fd_(gufo_descriptor_from_handle(file)) {
  }
  ~DescriptorView() {
    if (fd_ >= 0)
      ::close(fd_);
  }
#else
      : fd_(file) {
  }
#endif
  DescriptorView(const DescriptorView&) = delete;
  DescriptorView& operator=(const DescriptorView&) = delete;
  [[nodiscard]] int get() const noexcept { return fd_; }

private:
  int fd_;
};

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_DESCRIPTOR_HPP_
