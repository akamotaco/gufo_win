#include "ds41_platform.h"

#include <algorithm>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#endif

bool ds41_read_exact(ds41_file file, void* out, uint64_t bytes, uint64_t offset) {
  auto* dst = static_cast<uint8_t*>(out);
  uint64_t done = 0;
  while (done < bytes) {
    // Under 1 GiB per call: Windows ReadFile takes a 32-bit count.
    const size_t want = (size_t)std::min<uint64_t>(bytes - done, 1ull << 30);
    const size_t got =
        gufo::platform::ReadFileAt(file, dst + done, want, offset + done);
    if (got == 0)
      return false;
    done += got;
  }
  return true;
}

#if defined(_WIN32)

static uint64_t page_up(uint64_t v) {
  return (v + 4095) & ~uint64_t{4095};
}

/* Placeholder APIs (Windows 10 1803+) live in kernelbase.dll; the default
 * kernel32 import library does not export them, so they are looked up once. */
namespace {
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG,
                                       MEM_EXTENDED_PARAMETER*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T,
                                        ULONG, ULONG, MEM_EXTENDED_PARAMETER*,
                                        ULONG);
using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);
struct PlaceholderApi {
  VirtualAlloc2Fn virtual_alloc2 = nullptr;
  MapViewOfFile3Fn map_view3 = nullptr;
  UnmapViewOfFile2Fn unmap_view2 = nullptr;
  PlaceholderApi() {
    HMODULE k = GetModuleHandleW(L"kernelbase.dll");
    if (!k)
      return;
    virtual_alloc2 =
        reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(k, "VirtualAlloc2"));
    map_view3 = reinterpret_cast<MapViewOfFile3Fn>(
        GetProcAddress(k, "MapViewOfFile3"));
    unmap_view2 = reinterpret_cast<UnmapViewOfFile2Fn>(
        GetProcAddress(k, "UnmapViewOfFile2"));
  }
  bool ok() const { return virtual_alloc2 && map_view3 && unmap_view2; }
};
const PlaceholderApi& api() {
  static const PlaceholderApi a;
  return a;
}
}  // namespace

const uint8_t* ds41_map_shards(const ds41_file* files, const uint64_t* offsets,
                               const uint64_t* sizes, uint32_t count,
                               uint64_t total, const char** error) {
  if (!api().ok()) {
    *error = "VirtualAlloc2/MapViewOfFile3 are unavailable (Windows 10 1803+)";
    return nullptr;
  }
  auto* base = static_cast<uint8_t*>(api().virtual_alloc2(
      nullptr, nullptr, (SIZE_T)total, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
      PAGE_NOACCESS, nullptr, 0));
  if (!base) {
    *error = "cannot reserve the model address range";
    return nullptr;
  }
  uint32_t mapped = 0;
  for (; mapped < count; mapped++) {
    uint8_t* at = base + offsets[mapped];
    const uint64_t view = page_up(sizes[mapped]);
    // Split the placeholder so [at, at + view) is one placeholder of its own.
    if (!VirtualFree(at, (SIZE_T)view, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
      *error = "cannot split the model address range";
      break;
    }
    HANDLE section = CreateFileMappingW(static_cast<HANDLE>(files[mapped]),
                                        nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!section) {
      *error = "cannot create a file mapping for a model shard";
      break;
    }
    void* v = api().map_view3(section, GetCurrentProcess(), at, 0,
                             (SIZE_T)sizes[mapped], MEM_REPLACE_PLACEHOLDER,
                             PAGE_READONLY, nullptr, 0);
    CloseHandle(section);  // the view keeps the section alive
    if (v != at) {
      *error = "cannot map a model shard";
      break;
    }
  }
  if (mapped == count)
    return base;
  ds41_unmap_shards(base, offsets, sizes, mapped, total);
  return nullptr;
}

void ds41_unmap_shards(const uint8_t* base, const uint64_t* offsets,
                       const uint64_t* sizes, uint32_t count, uint64_t total) {
  if (!base)
    return;
  auto* b = const_cast<uint8_t*>(base);
  for (uint32_t i = 0; i < count; i++)
    (void)api().unmap_view2(GetCurrentProcess(), b + offsets[i],
                           MEM_PRESERVE_PLACEHOLDER);
  (void)sizes;
  // Merge the placeholders back into one, then release the reservation.
  (void)VirtualFree(b, (SIZE_T)total, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
  (void)VirtualFree(b, 0, MEM_RELEASE);
}

void ds41_advise_random(ds41_file, const uint8_t*, uint64_t) {}

#else

const uint8_t* ds41_map_shards(const ds41_file* files, const uint64_t* offsets,
                               const uint64_t* sizes, uint32_t count,
                               uint64_t total, const char** error) {
  void* region = mmap(nullptr, (size_t)total, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (region == MAP_FAILED) {
    *error = "cannot reserve the model address range";
    return nullptr;
  }
  for (uint32_t i = 0; i < count; i++) {
    void* at = (uint8_t*)region + offsets[i];
    void* mapped = mmap(at, (size_t)sizes[i], PROT_READ, MAP_SHARED | MAP_FIXED,
                        files[i], 0);
    if (mapped == MAP_FAILED) {
      *error = "cannot mmap model shard";
      munmap(region, (size_t)total);
      return nullptr;
    }
  }
  return static_cast<const uint8_t*>(region);
}

void ds41_unmap_shards(const uint8_t* base, const uint64_t*, const uint64_t*,
                       uint32_t, uint64_t total) {
  if (base)
    munmap((void*)base, (size_t)total);
}

void ds41_advise_random(ds41_file file, const uint8_t* map, uint64_t bytes) {
  (void)posix_fadvise(file, 0, 0, POSIX_FADV_RANDOM);
  (void)madvise((void*)map, (size_t)bytes, MADV_RANDOM);
}

#endif
