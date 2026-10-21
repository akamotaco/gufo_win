// Standalone timing of Gemma 4's dense prefill projection kernel
// (LaunchDenseHalfGemm) on production shapes with production-format weights.
//
//   nix develop -c tools/bench/build.sh gemma4_prefill_bench
//   /tmp/gemma4_prefill_bench [iterations]
//
// Cold-weight reads: every shape rotates over enough copies of its matrix to
// exceed the 32 MiB MALL, so a repeat does not read a cached tensor (the small
// attention projections sit entirely in the MALL otherwise). Outputs follow
// production: FP32 rows for a plain projection, binary16 GeGLU rows for the
// fused [gate | up] shape.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The production translation unit itself: the bench builds the exact kernel and
// flags the model ships, so a change is measured here before it lands.
#include "src/models/gemma4/kernels/rocm/moe_prefill.hip.cpp"

using gufo::models::gemma4::rocm::ExpertFormat;
using gufo::models::gemma4::rocm::LaunchDenseHalfGemm;

namespace {

constexpr std::uint32_t kThreads = 256;

__global__ void Fill(std::uint8_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n;
       i += std::size_t{gridDim.x} * blockDim.x) {
    std::uint32_t v = static_cast<std::uint32_t>(i) * 2654435761U + seed;
    v ^= v >> 13;
    p[i] = static_cast<std::uint8_t>(v);
  }
}

void FillBuffer(void* p, std::size_t n, std::uint32_t seed) {
  const unsigned blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  Fill<<<blocks, kThreads>>>(static_cast<std::uint8_t*>(p), n, seed);
}

void Alloc(void** p, std::size_t bytes, const char* what) {
  if (hipMalloc(p, bytes) != hipSuccess) {
    std::printf("alloc failed (%s, %zu bytes)\n", what, bytes);
    std::exit(1);
  }
}

std::uint32_t BlockBytes(ExpertFormat f) {
  switch (f) {
    case ExpertFormat::kQ4_K:
      return 144;
    case ExpertFormat::kQ5_K:
      return 176;
    case ExpertFormat::kQ6_K:
      return 210;
    default:
      return 0;
  }
}

const char* Name(ExpertFormat f) {
  switch (f) {
    case ExpertFormat::kQ4_K:
      return "Q4_K";
    case ExpertFormat::kQ5_K:
      return "Q5_K";
    case ExpertFormat::kQ6_K:
      return "Q6_K";
    default:
      return "?";
  }
}

struct Shape {
  const char* name;
  ExpertFormat format;
  std::uint32_t m;
  std::uint32_t k;
  std::uint32_t rows;
  bool geglu;
  int count;  // launches per 2048-token chunk, for the predicted total
};

double Bench(const Shape& s, int iterations, double* tflops) {
  const std::uint32_t block = BlockBytes(s.format);
  const std::size_t row_bytes = std::size_t{s.k} / 256 * block;
  const std::size_t weight_bytes = std::size_t{s.m} * row_bytes;
  const std::size_t x_bytes = std::size_t{s.rows} * s.k * sizeof(std::uint16_t);
  const std::uint32_t out_rows = s.geglu ? s.m / 2 : s.m;
  const std::size_t out_half_bytes = std::size_t{s.rows} * out_rows * 2;
  const std::size_t out_f32_bytes = std::size_t{s.rows} * out_rows * 4;
  // Enough copies to exceed the MALL, at least one.
  const int copies = std::max(
      1, static_cast<int>(std::ceil(96.0e6 / static_cast<double>(weight_bytes))));
  std::vector<void*> w(static_cast<std::size_t>(copies));
  for (auto& p : w) {
    Alloc(&p, weight_bytes, s.name);
    FillBuffer(p, weight_bytes, 7);
  }
  void* x = nullptr;
  Alloc(&x, x_bytes, "x");
  FillBuffer(x, x_bytes, 11);
  void* out_half = nullptr;
  void* out_f32 = nullptr;
  if (s.geglu) {
    Alloc(&out_half, out_half_bytes, "out_half");
    FillBuffer(out_half, out_half_bytes, 13);
  } else {
    Alloc(&out_f32, out_f32_bytes, "out");
    FillBuffer(out_f32, out_f32_bytes, 13);
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    std::printf("warmup failed for %s\n", s.name);
    std::exit(1);
  }

  auto run = [&](int i) {
    LaunchDenseHalfGemm(s.format, w[static_cast<std::size_t>(i) % copies], x,
                        static_cast<float*>(out_f32), out_half, s.rows, s.m,
                        s.k, nullptr, s.geglu);
  };
  for (int i = 0; i < 3; ++i) {
    run(i);
  }
  hipDeviceSynchronize();
  hipEvent_t a = nullptr;
  hipEvent_t b = nullptr;
  hipEventCreate(&a);
  hipEventCreate(&b);
  std::vector<double> ms;
  ms.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    hipEventRecord(a, nullptr);
    run(i);
    hipEventRecord(b, nullptr);
    hipEventSynchronize(b);
    float elapsed = 0.0F;
    hipEventElapsedTime(&elapsed, a, b);
    ms.push_back(elapsed);
  }
  hipEventDestroy(a);
  hipEventDestroy(b);
  std::sort(ms.begin(), ms.end());
  const double best = ms.front();
  *tflops = 2.0 * static_cast<double>(s.rows) * static_cast<double>(s.m) *
            static_cast<double>(s.k) / (best * 1e-3) / 1e12;
  for (auto& p : w) {
    hipFree(p);
  }
  hipFree(x);
  hipFree(out_half);
  hipFree(out_f32);
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 5;
  hipDeviceProp_t prop{};
  hipGetDeviceProperties(&prop, 0);
  std::printf("device: %s  CUs=%d  clock=%.2f GHz  iterations=%d\n", prop.name,
              prop.multiProcessorCount, prop.clockRate / 1e6, iterations);
  // One 2048-token chunk of UD-Q4_K_XL: the per-layer mix of formats, rows and
  // shapes (54 Q4_K layers and 6 Q5_K; 10 global layers with an 18432-row fused
  // q/k and a 16384-wide o, 50 sliding with 12288 and 8192).
  const Shape shapes[] = {
      {"gate|up Q4_K [43008x5376]", ExpertFormat::kQ4_K, 43008, 5376, 2048, true, 54},
      {"gate|up Q5_K [43008x5376]", ExpertFormat::kQ5_K, 43008, 5376, 2048, true, 6},
      {"down Q4_K [5376x21504]", ExpertFormat::kQ4_K, 5376, 21504, 2048, false, 34},
      {"down Q6_K [5376x21504]", ExpertFormat::kQ6_K, 5376, 21504, 2048, false, 26},
      {"q|k sliding [12288x5376]", ExpertFormat::kQ4_K, 12288, 5376, 2048, false, 46},
      {"q|k global [18432x5376]", ExpertFormat::kQ4_K, 18432, 5376, 2048, false, 8},
      {"q|k global Q5_K [18432x5376]", ExpertFormat::kQ5_K, 18432, 5376, 2048, false, 2},
      {"o sliding [5376x8192]", ExpertFormat::kQ4_K, 5376, 8192, 2048, false, 50},
      {"o global [5376x16384]", ExpertFormat::kQ4_K, 5376, 16384, 2048, false, 9},
      {"o global Q5_K [5376x16384]", ExpertFormat::kQ5_K, 5376, 16384, 2048, false, 1},
      {"v sliding [4096x5376]", ExpertFormat::kQ6_K, 4096, 5376, 2048, false, 50},
      {"v global [2048x5376]", ExpertFormat::kQ4_K, 2048, 5376, 2048, false, 10},
  };
  std::printf("\n%-30s %10s %10s %6s %9s\n", "shape", "ms", "TFLOP/s", "count", "chunk ms");
  double chunk_ms = 0.0;
  double chunk_flops = 0.0;
  for (const Shape& s : shapes) {
    double tflops = 0.0;
    const double ms = Bench(s, iterations, &tflops);
    chunk_ms += ms * s.count;
    chunk_flops += 2.0 * static_cast<double>(s.rows) * static_cast<double>(s.m) *
                   static_cast<double>(s.k) * s.count;
    std::printf("%-30s %10.3f %10.2f %6d %9.1f\n", s.name, ms, tflops, s.count,
                ms * s.count);
  }
  std::printf("\npredicted chunk projections: %.0f ms, %.2f TFLOP/s\n", chunk_ms,
              chunk_flops / (chunk_ms * 1e-3) / 1e12);
  std::printf("production (d0 chunk, profiled): 3246 ms, 37.0 TFLOP/s\n");
  return 0;
}
