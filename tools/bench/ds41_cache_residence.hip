/* Where the streamed expert cache should live: the production table-indexed
 * IQ2_XXS expert GEMV over experts scattered in a large cache region, with
 * the region in three kinds of memory:
 *
 *   managed  hipMallocManaged + coarse grain, 128 MiB chunks (the fallback)
 *   malloc   hipMalloc, 128 MiB chunks (the device heap)
 *   vmm      hipMemCreate pages mapped into one reserved range (the store)
 *
 * Every variant holds the same bytes per slot, so outputs must match bit for
 * bit. Also times pinned-to-slot copies (cache admission) and, for vmm, page
 * creation, mapping and unmapping per GiB.
 *
 *   nix develop -c tools/bench/build.sh ds41_cache_residence
 *   /tmp/ds41_cache_residence [GiB]   # region size, default 32 */
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../kernels/rocm/ds41_expert_gemv.hip.cpp"

namespace {

#define CHECK(call)                                                                        \
  do {                                                                                     \
    const hipError_t e_ = (call);                                                          \
    if (e_ != hipSuccess) {                                                                \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e_)); \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

constexpr uint64_t kChunk = 128ull << 20;
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

/* Deterministic bytes per slot: the same content in every variant. IQ2_XXS
 * blocks start with an FP16 scale; keep it finite and small. */
__global__ void FillSlots(uint8_t* base, uint64_t slot_bytes, uint64_t slots, uint64_t first_slot) {
  const uint64_t words = slot_bytes / 4;
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < slots * words;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t slot = first_slot + i / words, w = i % words;
    uint64_t h = (slot * 0x9E3779B97F4A7C15ull) ^ (w * 0xBF58476D1CE4E5B9ull);
    h ^= h >> 31;
    h *= 0x94D049BB133111EBull;
    h ^= h >> 29;
    uint32_t v = static_cast<uint32_t>(h);
    if ((w * 4) % 66 == 0) v = (v & 0xFFFF0000u) | (0x2000u | (v & 0x3FFu));  // block scale
    reinterpret_cast<uint32_t*>(base + (i / words) * slot_bytes)[w] = v;
  }
}

struct Region {
  std::string kind;
  std::vector<void*> chunks;  // managed / malloc
  void* va = nullptr;         // vmm
  uint64_t bytes = 0;
  std::vector<hipMemGenericAllocationHandle_t> pages;
  uint64_t page = 0;
  /* Slot i's address; slots never straddle a chunk. */
  uint64_t slot_bytes = 0, per_chunk = 0;
  uint8_t* Slot(uint64_t i) const {
    if (kind == "vmm") return static_cast<uint8_t*>(va) + i * slot_bytes;
    return static_cast<uint8_t*>(chunks[i / per_chunk]) + (i % per_chunk) * slot_bytes;
  }
};

Region Allocate(const std::string& kind, uint64_t bytes, uint64_t slot_bytes, double* alloc_s) {
  Region r;
  r.kind = kind;
  r.slot_bytes = slot_bytes;
  r.per_chunk = kChunk / slot_bytes;
  const auto t = Clock::now();
  if (kind == "vmm") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    size_t gran = 0;
    CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
    r.page = gran;
    r.bytes = (bytes + gran - 1) / gran * gran;
    CHECK(hipMemAddressReserve(&r.va, r.bytes, gran, nullptr, 0));
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    for (uint64_t off = 0; off < r.bytes; off += gran) {
      hipMemGenericAllocationHandle_t h;
      CHECK(hipMemCreate(&h, gran, &prop, 0));
      CHECK(hipMemMap(static_cast<uint8_t*>(r.va) + off, gran, 0, h, 0));
      CHECK(hipMemSetAccess(static_cast<uint8_t*>(r.va) + off, gran, &access, 1));
      r.pages.push_back(h);
    }
  } else {
    for (uint64_t got = 0; got < bytes; got += kChunk) {
      void* p = nullptr;
      if (kind == "managed") {
        CHECK(hipMallocManaged(&p, kChunk, hipMemAttachGlobal));
        (void)hipMemAdvise(p, kChunk, hipMemAdviseSetCoarseGrain, 0);
        (void)hipGetLastError();
      } else {
        CHECK(hipMalloc(&p, kChunk));
      }
      r.chunks.push_back(p);
    }
    r.bytes = r.chunks.size() * kChunk;
  }
  *alloc_s = Seconds(t);
  return r;
}

double Free(Region& r) {
  CHECK(hipDeviceSynchronize());
  const auto t = Clock::now();
  if (r.kind == "vmm") {
    CHECK(hipMemUnmap(r.va, r.bytes));
    for (auto h : r.pages) CHECK(hipMemRelease(h));
    CHECK(hipMemAddressFree(r.va, r.bytes));
  } else {
    for (void* p : r.chunks) CHECK(hipFree(p));
  }
  return Seconds(t);
}

}  // namespace

int main(int argc, char** argv) {
  setbuf(stdout, nullptr);
  const uint64_t gib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 32;
  /* IQ2_XXS gate/up and down of one DeepSeek V4.1 Flash expert. */
  constexpr int kType = 16, kBlock = 66;
  constexpr uint32_t kM = 2304, kK = 5120;
  const uint64_t matrix = uint64_t(kM) * kK / 256 * kBlock;  // gate, up or down: same size
  const uint64_t slot_bytes = (matrix + 4095) / 4096 * 4096;
  const uint64_t region_bytes = gib << 30;
  const uint64_t slots_per_chunk = kChunk / slot_bytes;
  const uint64_t n_slots = region_bytes / kChunk * slots_per_chunk;
  constexpr int kRounds = 2000, kTop = 6, kBatch = 48;

  /* Random slots per round: gate, up and down tables of kTop (decode) and a
   * batch of kBatch distinct slots (DSpark verification-like width). */
  std::mt19937_64 rng(41);
  std::vector<uint64_t> pick(uint64_t(kRounds) * (3 * kTop + kBatch));
  for (auto& s : pick) s = rng() % n_slots;
  std::vector<float> xh(uint64_t(kK) * kBatch);
  std::normal_distribution<float> normal;
  for (auto& v : xh) v = normal(rng);

  float *x = nullptr, *y = nullptr;
  int32_t* ids = nullptr;
  const void** tables = nullptr;
  CHECK(hipMalloc(&x, xh.size() * sizeof(float)));
  CHECK(hipMalloc(&y, uint64_t(kK) * kBatch * sizeof(float)));
  CHECK(hipMalloc(&ids, kBatch * sizeof(int32_t)));
  CHECK(hipMalloc(&tables, pick.size() * sizeof(void*)));
  CHECK(hipMemcpy(x, xh.data(), xh.size() * sizeof(float), hipMemcpyHostToDevice));
  std::vector<int32_t> idh(kBatch);
  for (int i = 0; i < kBatch; ++i) idh[i] = i;
  CHECK(hipMemcpy(ids, idh.data(), idh.size() * sizeof(int32_t), hipMemcpyHostToDevice));
  void* pinned = nullptr;
  CHECK(hipHostMalloc(&pinned, slot_bytes * 16));
  std::memset(pinned, 0x5a, slot_bytes * 16);

  std::printf("region %llu GiB, %llu slots of %.2f MiB, %d rounds\n", (unsigned long long)gib,
              (unsigned long long)n_slots, slot_bytes / 1048576.0, kRounds);
  std::printf("%-8s %9s %9s %10s %10s %10s %9s %s\n", "kind", "alloc s", "fill s", "gu GB/s", "down GB/s",
              "b48 GB/s", "copy GB/s", "free s");
  std::vector<float> reference;
  const char* kinds[] = {"managed", "malloc", "vmm"};
  for (const char* kind : kinds) {
    double alloc_s = 0;
    Region r = Allocate(kind, region_bytes, slot_bytes, &alloc_s);
    auto t = Clock::now();
    if (r.kind == "vmm") {
      FillSlots<<<4096, 256>>>(r.Slot(0), slot_bytes, n_slots, 0);
    } else {
      for (size_t c = 0; c < r.chunks.size(); ++c)
        FillSlots<<<1024, 256>>>(static_cast<uint8_t*>(r.chunks[c]), slot_bytes, slots_per_chunk,
                                 c * slots_per_chunk);
    }
    CHECK(hipDeviceSynchronize());
    const double fill_s = Seconds(t);
    std::vector<const void*> th(pick.size());
    for (size_t i = 0; i < pick.size(); ++i) th[i] = r.Slot(pick[i]);
    CHECK(hipMemcpy(tables, th.data(), th.size() * sizeof(void*), hipMemcpyHostToDevice));
    const uint64_t per_round = 3 * kTop + kBatch;
    auto gu = [&](int round) {
      const void* const* tb = tables + uint64_t(round) * per_round;
      return ds41_v41_gemv_experts_swiglu(kType, r.Slot(0), r.Slot(0), slot_bytes, ids, kTop, x, 0, 1, nullptr, 0.0f,
                                          y, kM, kK, tb, tb + kTop);
    };
    auto down = [&](int round) {
      const void* const* tb = tables + uint64_t(round) * per_round + 2 * kTop;
      return ds41_v41_gemv_experts(kType, r.Slot(0), slot_bytes, ids, kTop, x, kM, 1, y, kK, kM, tb);
    };
    auto batch = [&](int round) {
      const void* const* tb = tables + uint64_t(round) * per_round + 3 * kTop;
      return ds41_v41_gemv_experts(kType, r.Slot(0), slot_bytes, ids, kBatch, x, kM, 1, y, kK, kM, tb);
    };
    /* Bit-identity: the first round of each launch kind. */
    std::vector<float> out;
    auto capture = [&](auto&& launch, uint64_t n) {
      CHECK(hipMemset(y, 0, uint64_t(kK) * kBatch * sizeof(float)));
      if (launch(0)) std::exit(2);
      CHECK(hipDeviceSynchronize());
      std::vector<float> o(n);
      CHECK(hipMemcpy(o.data(), y, n * sizeof(float), hipMemcpyDeviceToHost));
      out.insert(out.end(), o.begin(), o.end());
    };
    capture(gu, uint64_t(kM) * kTop);
    capture(down, uint64_t(kK) * kTop);
    capture(batch, uint64_t(kK) * kBatch);
    const bool same = reference.empty() || std::memcmp(reference.data(), out.data(), out.size() * 4) == 0;
    if (reference.empty()) reference = out;
    auto rate = [&](auto&& launch, double bytes_per_round) {
      for (int i = 0; i < 16; ++i) (void)launch(i);
      hipEvent_t a, b;
      CHECK(hipEventCreate(&a));
      CHECK(hipEventCreate(&b));
      CHECK(hipEventRecord(a, 0));
      for (int i = 0; i < kRounds; ++i) (void)launch(i);
      CHECK(hipEventRecord(b, 0));
      CHECK(hipEventSynchronize(b));
      float ms = 0;
      CHECK(hipEventElapsedTime(&ms, a, b));
      return bytes_per_round * kRounds / (ms * 1e-3) / 1e9;
    };
    const double gu_gbs = rate(gu, 2.0 * kTop * matrix);
    const double down_gbs = rate(down, 1.0 * kTop * matrix);
    const double b48_gbs = rate(batch, 1.0 * kBatch * matrix);
    /* Admission: pinned staging rows into random slots. */
    t = Clock::now();
    constexpr int kCopies = 2000;
    for (int i = 0; i < kCopies; ++i)
      CHECK(hipMemcpyAsync(r.Slot(pick[i]), static_cast<uint8_t*>(pinned) + (i % 16) * slot_bytes, matrix,
                           hipMemcpyHostToDevice, 0));
    CHECK(hipDeviceSynchronize());
    const double copy_gbs = double(kCopies) * matrix / Seconds(t) / 1e9;
    const double free_s = Free(r);
    std::printf("%-8s %9.2f %9.2f %10.1f %10.1f %10.1f %9.1f %.2f%s\n", kind, alloc_s, fill_s, gu_gbs, down_gbs,
                b48_gbs, copy_gbs, free_s, same ? "" : "  OUTPUTS DIFFER");
  }
  return 0;
}
