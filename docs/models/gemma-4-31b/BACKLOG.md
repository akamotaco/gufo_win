# Gemma 4 performance backlog

Open performance work for the Gemma 4 engine (31B, 31B QAT, 26B-A4B), as of
2026-10-17 on branch `feat/gemma-perf`. Measured results, including every
rejected attempt, are in [EXPERIMENTS.md](EXPERIMENTS.md) and the
[26B log](../gemma-4-26b-a4b/EXPERIMENTS.md); read the matching rows before
retrying anything listed here. Impact figures are estimates on 31B
UD-Q4_K_XL tok/s unless stated.

## Where the time goes

Prefill, one 2048-token chunk (31B UD-Q4_K_XL, rocprof):

| Depth | Projections | Global attention | Sliding attention | Other kernels | Chunk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.30 s (91%) | 0.05 s | 0.10 s | 0.17 s | 3.6 s |
| 16K | 3.30 s (~77%) | ~0.73 s (~17%) | 0.14 s | 0.17 s | ~4.35 s |
| 32K | 3.40 s (~66%) | ~1.48 s (~28%) | 0.15 s | 0.17 s | ~5.2 s |
| 128K | ~3.4 s (~35%) | ~6 s (~60%) | 0.15 s | 0.17 s | ~9.6 s |

Projections cost the same at every depth; global attention grows linearly
(~46 ms per 1K of depth). Projections run Q4_K at ~38–39 TFLOP/s and global
attention at ~35, against a measured 55.4 TFLOP/s binary16 WMMA ceiling
(`tools/bench/gfx1151_peak.hip`).

Decode, autoregressive (31B Q4, per token): ~84 ms of weight reads (~94% of
the measured 239 GB/s), ~0.6 ms of attention at d0 growing to ~36 ms at 128K
(global-layer KV, 50 KiB per cached token), ~6 ms of small kernels and launch
gaps. MTP verify cycles at d0 (~101 ms GPU): target verification projections
~93%, drafter ~3–5%, attention ~2%.

## Open items

| # | Item | Area | Effort | Est. impact | Confidence |
| ---: | --- | --- | --- | --- | --- |
| 1 | MTP verification projections: close the 5–13% per-byte gap of the WMMA verification GEMV to the AR GEMV | Decode MTP, all depths | Hard | MTP +5–10% | Low |
| 2 | 8-bit KV cache | Decode at depth, memory | Very hard, quality risk | AR +5% at 32K, +14% at 128K; MTP more; about half the KV memory | Speed medium, quality unknown |
| 3 | MTP verification attention at five or more rows | Decode MTP at depth | Hard | +2–5% at 32K, more at 128K | Low |
| 4 | Prefill projections past ~70% of the WMMA ceiling | Prefill, all depths | Hard | +2–5% | Low |
| 5 | Global prefill attention redesign: two heads per wave over one shared LDS arena | Prefill at depth | Hard | +1.5–3% at 32K, +3–6% at 128K | Medium |
| 6 | AR decode GEMV | Decode AR | Hard | +2–4% | Low |

### 1. MTP verification projections

`wmma_gemv.hip.cpp`. The kernel stays 5–13% behind the AR Gemma GEMV on equal
bytes (cold, one row: 5376×16384 247 vs 220 µs). Load-only probes cap it at
207–217 GB/s regardless of barriers, prefetch depth, request size or DRAM
locality; occupancy is 9–24 against the AR GEMV's 47–62. Rejected so far:
split half-wave loads, deeper weight prefetch, software-pipelined groups. The
remaining idea is cooperative full-line weight loads redistributed to the WMMA
lanes (today each load instruction touches 16 cache lines). Verification must
keep one arithmetic per format so it still equals decode bit for bit
(`gemma4.projection_ops`).

### 2. 8-bit KV cache

The only large lever left for long contexts: decode streams the global-layer
KV once per token (or verify cycle, plus once per draft step in the drafter's
global layer). It changes numerics, so it needs a quality decision first:
target KL against the scalar oracle (`gemma4.target`), long-context and
functional suites. The format touches decode and verification attention
(`RowSplitAttentionKernel`), prefill attention (`attention_wmma.hip.cpp`),
QkvPost's cache writes, snapshots and the disk cache. Prefill attention is
compute-bound, so it gains little there.

### 3. MTP verification attention, five or more rows

`kernels.hip.cpp`, `RowSplitAttentionKernel` and `LaunchSplitAttention`. Two
to four rows now run on one wave per head (−28% for two rows at 32K). From
about four rows the kernel is compute-bound at ~270–300 µs per added row and
layer at 32K, ~50% of the unpaired FP32 FMA ceiling. Paired FMAs (VOPD
through opaque converted floats) were neutral, so the cost is the per-row score
reductions (`ReduceScatterN`), softmax statistics (`WaveMax`, `WaveSum`) and
the per-key `readlane` broadcasts. Rows must keep the arithmetic of
single-row decode (the op test compares every verification row with its
single-row evaluation bit for bit). The draft cost tables in
`draft_policy.cpp` hold verification costs by rows and depth: re-measure the
affected entries after any change (a kernel A/B at each table depth, times the
number of global layers).

### 4. Prefill projections

`moe_prefill.hip.cpp`, `RoutedHalfKQuantKernel` (dense and routed). Drop-work
probes on the Q4_K gate/up (25 ms): without WMMAs 9.5 ms; without the weight
decode −12.5%; without the global → LDS staging −8%; without barriers −3%.
The decode only costs where it shares the SIMD issue with the WMMAs, and the
WMMA A operand must be replicated in both half-waves, so each row is decoded
twice. Rejected: 128-token tiles (three blocks per WGP), unpadded XOR-swizzled
activation rows (registers), 16-wave 256-row blocks, a cross-half decode swap,
scale-conversion splits for Q5_K and Q6_K (registers). A real gain probably
needs a different decode scheme.

### 5. Global prefill attention redesign

`attention_wmma.hip.cpp`. After today's rows the tile loop issues ~350
instructions per wave and 16-key tile, 16 of them WMMAs; VALU and WMMA share
the SIMD issue, so cutting instructions maps almost one to one to time. Probes
cap the obvious redesign: loading every K/Vᵀ fragment once per two products
saves only 7%, removing all barriers 5%. Two heads per wave would halve the
fragment reads but needs eight score partials, beyond 64 KB of LDS without
reusing one arena for K, the partials and Vᵀ. Larger key tiles do not fit in
64 KB. The kernel's outputs have been kept bit-identical; keep doing so or
revalidate quality.

### 6. AR decode GEMV

Already at ~94% of the measured DRAM read bandwidth; past work is in the
decode GEMV rows of EXPERIMENTS.md.

## Measured, no change (do not retry without a new idea)

- Fusing the prefill norms and Q/K post-processing: they move 154–230 MB per
  layer at 185–203 GB/s against a 210 GB/s read-plus-write ceiling.
- Decode launch gaps through HIP graphs: a 733-kernel dependent chain takes
  7.9 µs per kernel launched on a stream and the same replayed as a graph; the
  gaps are GPU-side. A removed launch saves ~1.4 µs.
- Prefill attention LDS padding sweeps, packed probability stores, interleaved
  row scales, 128-bit Vᵀ stores, DPP-swapped score writes (see EXPERIMENTS.md).

## Housekeeping

- Combined prefill benchmark of all 2026-10-17 rows against c06dc7db: running
  at the end of that session; record it in EXPERIMENTS.md if it is missing.
- Model card refresh: BENCHMARKS.md is from 2026-10-07 and holds none of the
  2026-10-17 gains; one full sweep (the `benchmark-model` skill) at the end of
  the round.
- `gemma4.target` on 26B QAT fails "shared sampled MTP verified or accepted no
  siblings" with output identical to 2026-10-16's head: pre-existing,
  uninvestigated.
- The branch is not pushed; its pull request title follows
  [the release policy](../../RELEASING.md).

## Methods that worked

- Kernel A/B in one binary: compile HEAD's kernel file with its namespace
  renamed (`git show HEAD:<file> | sed 's/^namespace X {$/namespace X::base {/'`)
  next to the working copy, run both on production shapes, compare outputs
  bitwise and time them in interleaved rounds. Widen the inputs (for example
  queries ×10–30) to exercise underflow and large softmax maximum jumps.
- Drop-work probes: remove one mechanism at a time (results wrong, timing
  only) to see what a kernel's time is made of before redesigning it.
- Instruction counts of a kernel's hottest loop from `--cuda-device-only -S`
  (`tools/prof/isa_mix.py` for whole kernels); on RDNA3 VALU instructions and
  WMMAs share the issue, so the loop's VALU count predicts time.
- `gemma4.attention_ops` outputs hashed to prove bit identity across all its
  cases (a temporary local hash print; not committed).
- End to end: `tools/prof/prof.py` chunk profiles compared at matched
  projection times, because long prefills here drift 2–4% between runs with
  power and temperature; `gufo bench` in ABBA order for tok/s; an HTTP
  ~31K-token document with prose answers for MTP decode at depth (`gufo
  bench`'s synthetic prompt is repetitive and drafts 15 tokens every cycle).
