#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_DS41_PLATFORM_H_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_DS41_PLATFORM_H_

/* The few OS calls the V4.1 loader needs beyond Gufo's platform layer: shard
 * files are Gufo PlatformFiles (an fd on POSIX, a HANDLE on Windows), so the
 * expert store and the weight uploader take them as they are. */

#include <cstddef>
#include <cstdint>

#include <gufo/ext/platform/mapped_file.hpp>

using ds41_file = gufo::platform::PlatformFile;
inline constexpr ds41_file DS41_INVALID_FILE = gufo::platform::kInvalidFile;

/* Reads exactly `bytes` at `offset`; false on a short read or an error. */
bool ds41_read_exact(ds41_file file, void* out, uint64_t bytes, uint64_t offset);

/* Reserves `total` bytes of address space and maps file i read-only at
 * base + offsets[i] (offsets are 2 MiB aligned, files do not overlap). POSIX
 * reserves with mmap(PROT_NONE) and maps with MAP_FIXED; Windows splits a
 * VirtualAlloc2 placeholder and replaces each piece with MapViewOfFile3.
 * Returns the base, or nullptr with `error` set. */
const uint8_t* ds41_map_shards(const ds41_file* files, const uint64_t* offsets,
                               const uint64_t* sizes, uint32_t count,
                               uint64_t total, const char** error);
void ds41_unmap_shards(const uint8_t* base, const uint64_t* offsets,
                       const uint64_t* sizes, uint32_t count, uint64_t total);

/* Random-access hint for a mapped shard (POSIX fadvise + madvise); a no-op on
 * Windows. */
void ds41_advise_random(ds41_file file, const uint8_t* map, uint64_t bytes);

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_RUNTIME_DS41_PLATFORM_H_
