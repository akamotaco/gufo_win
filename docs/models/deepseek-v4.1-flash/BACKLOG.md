# DeepSeek V4.1 Flash backlog

The single list of what to do next. Items keep their numbers as stable IDs
(other documents cite "item N"); sections are in rank order. Results are in
[EXPERIMENTS.md](EXPERIMENTS.md) (`E<n>`; data recorded here first is in
[its last section](EXPERIMENTS.md#backlog-measurements-without-an-e-number)).
The streaming design is in [STREAMING.md](STREAMING.md), memory use in
[the README](../../../README.md#memory). Gains are
estimates unless marked measured.

## Where the time goes

Streamed `UD-IQ2_XXS`, 2026-11-02; detail in [DECODE.md](DECODE.md) and
[PREFILL.md](PREFILL.md).

- **Decode.** ~10 GB read per token, ~42 ms at the 240 GB/s ceiling
  (~24 tok/s); steady AR runs 53-58 ms. The gap: expert read stalls
  4.2-6.5 ms per AR token (8-13 ms per DSpark cycle), kernel bandwidth
  ~3.9 ms, launch gaps ~2.9 ms traced. Each verification row past the first
  adds ~12 ms of GPU time and ~4.5 ms of read stall.
- **Prefill.** A fresh 19,927-token served document: 20.59 s profiled,
  19.2 s after E82. GPU time 17.0 s: routed FP16 46%, dense FP16 ~27%,
  attention 11%, hyper-connections 8%, BF16 5%. Idle: encoder expert reads
  (819 ms over two documents after E84), the decoder replay's cold experts
  (0.91 s, drive-bound, E86), Engram rows (first batch only after E85).

## Open, ranked

Speed items that keep outputs unchanged come first, by expected value;
items that change rounding come last. Output-changing options and features
follow in their own sections.

| Rank | # | Item | Expected gain (evidence) | Next step |
| ---: | ---: | --- | --- | --- |
| 1 | 30 | More memory for the expert cache | Replays: +3.7 GiB −28% decode misses; the decoder replay's cold experts (0.91 s) are capacity-bound (E86). E93 gave one session +1.9 GiB, the 64-row demand stage +3.8 GiB (E119: decode −1.1 to −1.6%) | The 2 GiB headroom (swap bursts first) |
| 2 | 37 | Cheap prefetch adoption, then every-layer decode prediction | +2.3-2.8% decode simulated once adoption is free (E89); Belady would halve decode misses (E92) | Stop resolving a running job on hit layers |
| 3 | 31 | Eviction scoring and prompt admission | Frequency classes retained (E105); admitting every expert routed twice cut follow-up misses 32% but slowed prefill | Re-measure admission after direct-to-slot uploads |
| 4 | 9 | Stable-boundary prefill pass per chat request | 64.6 ms per chat request (measured pass cost) | Snapshot or skip the N−3 capture |
| 5 | 4 | Launch gaps between decode kernels | ~2.5 µs dispatch latency before each of ~975 launches, ~2.4 ms/token (E44, E60); fusions 0.2-0.5% each | Event-timed untraced share, then fusions |
| 6 | 3 | Expert reads for rejected verification rows | Each row past the first adds ~4.5 ms of read stall (DECODE.md); unmeasured | Charge a row its own miss bytes in the width choice |
| 7 | 6 | Routed FP16 kernel: weight decode | 46% of served prefill GPU time; decode ~4.4 of ~21 ms per layer (E25); low confidence after E81 | E25's ablation on served counts for the UD types |
| 8 | 7 | Remaining prefill glue | ~25-50 ms per 2K batch at 128K; router at ~5 TFLOP/s | Split-K router; FP16 Engram rows |
| 9 | 40 | FP16 Q for prefill attention | ~1% of prefill (E53 estimate) | Write Q in FP16 from the `q_b` epilogue |
| 10 | 22 | Decode misses after a long prompt | First 64 tokens after a 24K prompt read 2-3× the steady rate | Admission variants over the first 64-256 steps |
| 11 | 39 | Expert cache seed from served text | Fresh server: 74-78% hits and ~24 ms/token of read wait for the first 64 tokens | Histogram from served chat and code traces |
| 12 | 38 | Drafter pass | 9.15 ms per drafting cycle (E102), ~0.5 ms less after E102/E103; request-vocabulary head +1.1% greedy in replay (E64) | Drafter expert bytes; vocabulary head |
| 13 | 19 | Host work and round trips per step | ~0.7 ms per DSpark step (2026-10-29), 0.43-0.49 ms per AR greedy step | Logits download only when needed; device embedding |
| 14 | 29 | Zero-copy expert reads | ≤ ~1.5% of kernel time at an 88 GiB cap; first version slower | Slack from `STATX_DIOALIGN`, contiguous stage |
| 15 | 17 | Single-row routed expert GEMV bandwidth | Routed GEMVs at 205-220 GB/s against 240 | ISA mix and spill check |
| 16 | 28 | Split-read head share from measured drive speeds | None on matched drives; the slower drive's excess on mismatched ones | Per-drive "landed last" attribution |
| 17 | 35 | Rank-ordered expert layout | Agent session: ~0.5 s of 214 s (E70) | Replay prompt-heavy sessions |
| 18 | 5 | Compact selection for `--exact-prefill` rescoring | ~85 ms per masked layer per 2K chunk at 128K, `--exact-prefill` only | Compact WMMA rescoring |
| 19 | 36 | Decoder rows for short continuations of mid-prompt checkpoints | 1.9 s fallback instead of 0.18 s when hit; no hit in recorded sessions | Keep layer-19 rows per checkpoint |
| 20 | 15 | Smaller items | Small | See the list |
| 21 | 33 | Attention partials written back during the dense GEMVs | ~0.6 ms/AR token, ~3 ms per 6-row verification; 10 splits +0.5% AR, not retained under the rounding bar (E65) | Only with an accepted rounding change |
| 22 | 34 | Multi-token BF16 matmuls in verification | 0.2-0.35 ms per cycle (~0.3-0.4%); not retained under the rounding bar (E66) | Only with an accepted rounding change |

### 30. More memory for the expert cache

The cache gets what unified memory leaves after the dense weights, the
drafter, the sessions, the RAM prompt cache and the store's buffers; more
cache cuts misses without changing outputs, and is the only lever left for
the decoder replay's drive-bound stall (E86). Replay data:
[EXPERIMENTS.md](EXPERIMENTS.md#item-30-expert-cache-memory-replays). Parts:

- **Sized after the sessions** (done, E93): the server fits the cache to
  the memory its sessions leave, minus 2 GiB, shrinking it by whole chunks
  when needed; the automatic RAM prompt cache of a streamed model is 1 GiB
  with a disk cache. One session +1.9 GiB of cache; 4 sessions at 1M no
  longer swap. Context depth costs little (~0.06 GiB from 20K to 110K).
  Open: what is left of the 2 GiB headroom is ~1.5 GiB at the floor; a
  smaller margin needs the swap bursts below explained first.
- **Admitted misses straight into their slot** (rejected, E120): removes
  the stage-to-slot copy, but decode got 1.2-1.8% slower with the same
  misses. Only batches with more misses than stage rows do it (E119).
- **Compact demand stage** (done, E119): 64 rows; a batch with more misses
  reads admitted experts into their slots and lends slots to the rest.
  +3.8 GiB of cache, decode −1.1 to −1.6%, prefill −2.4 to −2.7%.
- **Prefetch stage** (done, E104): 96 rows, truncated wide jobs keep the
  most-routed experts; ~0.9 GiB to the cache. 48 rows slow a 3.5K prompt
  on a cold cache by 5-9%. The prompt cache keeps 16 transfer streams plus one
  per session instead of 64 (gufo-org/gufo#1108): +0.3 GiB of cache.
- **Startup seed stops early** (done, E104): once the server has fitted the
  cache, it grows toward its limit while 1 GiB stays free, instead of
  stopping at the 2 GiB reserve the limit already holds back. 3 of 4 starts
  after long prefills had left 0.4-0.6 GiB unused for the whole run.
- **Swap bursts in long single-session runs** (E93): ~20M pages out around
  the 11th-15th fresh 50-64K prompt in both builds, prefill down to
  450-640 tok/s for those prompts; the server's own swap stays 9-20 MB.

The drafter's size is settled (E62). Gate: outputs at a 24 GiB cap.

### 37. Cheap prefetch adoption, then every-layer decode prediction

Decode predicts the next layer only after a layer waited on reads
([design](STREAMING.md#prefetch-from-router-predictions)). Predicting at
every layer is +2.3-2.8% in `prefetch_sim` but measured slower (E89),
because every prepare then resolves a running job (stop handshake, read
remainder, bookkeeping: ~0.1 ms per layer, ~4 ms per cycle). Two layers
ahead predicts fewer misses than one (E89). This is also the route toward
Belady's optimum (−47.9% / −53.5% decode misses, E92), which past-use
eviction rules cannot approach. E100 made the hit layers' resolution free
and re-enabled `predict_hot`: read wait fell ~1 ms per token, but the
extra router passes and 60% more SSD reads cancelled it. What remains is
the prediction's own cost: a cheaper predictor (fewer router rows, or
only the layers that miss most) or a more selective trigger.

### 31. Eviction scoring and prompt admission

The scan is gone (E91/E92). **Eviction:** frequency classes replaced the
decayed score (E105): −6% reads on a replayed agent session, −10% on a
code prompt after a chat, +1-2% on a cold first prompt. The seeded
experts' starting weight sets that trade (none: cold 15K +4.6%; full: chat
decode +3.2%); a better start would separate prompt-serving experts from
decode's, e.g. seed from prefill and decode rankings apart. Belady still
halves decode misses (E92). **Admission:** admitting every expert
routed twice cut follow-up misses 32% but slowed prefill through
stage-to-slot copies ([runs](EXPERIMENTS.md#item-31-prompt-admission-served-runs));
re-measure after item 30's direct-to-slot uploads. More admission evicts
the replay's experts (E86) and interacts with item 22.

### 9. Stable-boundary prefill pass per chat request

The prompt cache captures a checkpoint only at a pass end, so every chat
request runs a 3-row pass for its stable boundary three tokens before the
end ([cost](EXPERIMENTS.md#item-9-stable-boundary-pass)); since E73 it no
longer replays the decoder. The checkpoint serves turns whose rendering
diverges after the user message (thinking mode, regenerated or edited
replies). Capture it only when the next turn can need it, or snapshot row
N−3 inside the final pass (invasive: compressor accumulators, SWA windows,
bounded replay). Gemma's generation-prompt lookahead is the precedent; the
policy is shared (`src/cache/retention.cpp`), so other models must hold.

### 4. Launch gaps between decode kernels

The gap is the device's dispatch latency between dependent launches; graph
replay keeps it (E44), and traced gaps after long GEMVs overstate it by
~4.5 µs (E60). Untraced, a dependent launch costs 1.2-1.9 µs (E101): all
~986 launches of a step are 2-3.4% of it, one fusion ~0.1%. [Traces](EXPERIMENTS.md#item-4-decode-launch-gaps-2026-10-29-2026-10-31).
Only fewer launches help:

- fusions: q rope into `q_b` (needs an epilogue pairing rows 8 lanes
  apart), route into the router matvec, the window copy (cannot precede
  attention, which reads the slot it overwrites); mix-collapse fusion was
  slower (E49);
- batching the per-session launches of a joint step;
- a persistent per-layer kernel, the larger step.

### 3. Expert reads for rejected verification rows

The distinct-expert count per verification layer is already at its
minimum (E89), so only fewer or better-chosen rows help. The verified
prefix prices misses only through average cycle times (E18, E87);
verification batches already predict after a missed layer (E31). Charge
each row the miss bytes of its own routes when choosing the width. See
[DECODE.md lever 3](DECODE.md#3-verification-row-cost-on-the-streamed-build).
Sampled cycles cannot choose better from q alone (E117: the 0.6 cut is
within 0.2% of the best prefix rule; ~7% of decode goes to rows past the
first rejection), and a linear cost model for greedy widths lost 1.5% on
code and documents (E116).

### 6. Routed FP16 kernel: weight decode

After E25, IQ1_M gate+up spends ~4.4 of ~21 ms per 2,048-row layer on
weight decode and ~14.7 ms on the matrix-core skeleton; IQ2_XXS's 16-byte
codebook reads cost ~2 ms (8-byte entries need ~12 more VALU per decode);
IQ2_S and IQ2_XS downs run 5-10% slower than IQ2_XXS. Tile, staging and
fragment variants are closed (E8, E35, E76-E81). Use served route counts
(`route_occupancy.py`, `ds41_mmq_bench --f16 1 --counts`), not the bench
prompt. See [PREFILL.md lever 3](PREFILL.md#3-routed-fp16-kernels-78-s-busy-46).

### 7. Remaining prefill glue

At 128K per 2K chunk: Q8_0 projections still on MMQ plus its Q8_1 pass,
the BF16 router at ~5 TFLOP/s, closing expands, SwiGLU, FP16 conversion,
~4.5 ms of embedding staging, FP32 Engram rows (FP16 would save 0.8 ms per
call) ([profile](EXPERIMENTS.md#item-7-prefill-glue-at-128k-2026-10-28)).
Router: E75's narrow kernel is not yet reprofiled served; split K with an
FP32 reduction in the same order keeps the sums bit-identical (E71's
32-value K steps, part order and per-step FP32 sums must stay).

### 40. FP16 Q for prefill attention

Prefill attention reads FP32 Q (0.2-0.5 ms per call); FP16 Q from the
`q_b` epilogue, with rope and rounding moved there, would save about half
(E53, not attempted). Attention grows with depth: check
`ds41_attention_test --bench` at 16K and 32K first, and keep the attention
input's rounding identical.

### 22. Decode misses after a long prompt

Once the cache is full a large prefill admits only its final 16 rows and
its frequent experts, and the next decode tokens pay for it
([DECODE.md lever 2](DECODE.md#2-expert-read-stalls-42-65-mstoken-ar-8-13-ms-per-dspark-cycle)).
mako found leaving prefill out of eviction scores raised steady decode
misses 20% ([external](EXPERIMENTS.md#items-13-22-26-external-measurements-and-reference-data)).
Measure hits and tok/s over the first 64-256 steps after 4K and 16K
prompts: current rule, admit every prompt expert, a looser threshold
(shared with item 31).

### 39. Expert cache seed from served text

The built-in ranking is a calibration trace and does not match chat text.
The store already loads a `MODEL.gguf.expert-histogram` sidecar
([STREAMING.md](STREAMING.md#expert-cache)); build one from served chat and
code route traces and measure the first 64 tokens of the DECODE.md chat
request. Matters for benchmarks and short sessions only.

### 38. Drafter pass

Per drafting cycle (DECODE.md lever 7, table in E102): 9.15 ms, of it the
target head over the drafted rows 2.58 ms at 210 GB/s, `markov_select`
1.48 ms at 223 GB/s (E47), the stage Q4_K GEMVs 1.76 ms at 113-181 GB/s
(~1.25 ms in the bench after E103), drafter experts 1.49 ms (bytes not
traced) and attention 3 × 41 µs isolated after E102 (not yet re-traced).
Next: an expert count for the routed share. A
request-vocabulary head (as Qwen's `draft_vocab.hpp`, with a fallback to
the full vocabulary after misses; the verifier keeps it) gains +1.1% only
with the specials and a request table; the id prefix alone loses (E64). It
needs gathered head and Markov W2 rows, index mapping for argmax, Markov
and sampled candidates, and per-session tables in batched drafting.

### 19. Host work and round trips per step

DSpark: verification argmax is on the device (E50); the remaining gaps sit
around the logits copies after the output head
([trace](EXPERIMENTS.md#item-19-dspark-host-round-trips-2026-10-29));
download full rows only when sampling needs them. AR: the host argmax, the
Engram hash, per-evaluation `std::async`/`engram_read` threads and the
embedding dequant and uploads follow each download (DECODE.md lever 8).
Device argmax for one-row steps, a device embedding lookup and a
persistent Engram reader pool would take most of it.

### 29. Zero-copy expert reads

The SSD can DMA straight into managed memory, which kernels read at full
speed; a first version (tag `ds41-zero-copy-wip`, reverted) read every
projection in place and got slower, because a page of slack per projection
cost cache capacity and per-expert reads lost merging
([data](EXPERIMENTS.md#item-29-zero-copy-expert-reads-first-version)).
To pay: slack sized from `STATX_DIOALIGN` (512 B on XFS) and a page only
where needed (btrfs, 4 KiB sectors); stage rows contiguous in file order;
or in place only into the stage. Measure at 64 GiB and on ~150 GB builds.

### 17. Single-row routed expert GEMV bandwidth

The routed GEMVs read 205-220 GB/s against 197-236 for the dense ones
([DECODE.md](DECODE.md#per-kernel-position-ar-single-token)); IQ1 downs
keep a fixed per-launch floor, while a plain read of the same experts
streams at 230-238 GB/s
([resident data](EXPERIMENTS.md#item-17-resident-single-row-routed-gemvs-2026-10-29)). More tiles,
deeper unrolls and LDS group sums were slower (E46, E59, E60). Check the
VALU share with `tools/prof/isa_mix.py`, and whether `expert_group_kernel`
variants still spill (80-540 bytes per lane before E37).

### 28. Split-read head share from measured drive speeds

Each read is split at the share packed into the tails file (`--head`,
default 0.5; 0.50-0.60 measured the same on matched drives, E61). On
unequal drives a miss finishes at the slower one's pace: pack a wider tail
(e.g. 0.3), time cold reads per drive at load and split at the measured
ratio. First add a per-drive "landed last, by how much" count, which
ds4-argodrive used to tune decode and prefill splits separately
([external](EXPERIMENTS.md#item-28-ds4-argodrive-split-reads-external)).

### 35. Rank-ordered expert layout

Per-expert contiguity is worth ~1%, long runs +36% (E30); in a served agent
session decode misses are single experts and prompt misses form runs for at
most 15% of their bytes (E70). Replay prompt-heavy sessions (long
documents, many fresh prompts) against a usage-rank order per layer before
converting any file; a second copy or a rank-ordered tails file is an
acceptable disk cost.

### 5. Compact selection for `--exact-prefill` rescoring

With bounded replay the masked decoder index layers (24, 28, 32, 36) see
only 140 rows, ~25 ms per 128K prompt (E34); under `--exact-prefill` they
score every chunk. Verification already rescores compactly (E1). Keep
absolute position tie order, the pinned newest block, causal partial
blocks and the emitted row order.

### 36. Decoder rows for short continuations of mid-prompt checkpoints

Grid checkpoints hold no decoder rows near their end (E73), so a prompt
ending within 140 tokens past one falls back to the previous checkpoint
([data](EXPERIMENTS.md#item-36-fallback-for-short-continuations-of-mid-prompt-checkpoints)).
Keep the layer-19 output of the 140 rows before each checkpoint as a
component and run only the decoder layers on restore, without storing
layer 20's compressed rows or advancing its pool again. Check bit for bit
against an uncached prompt.

### 15. Smaller items

- **Device top-k sampling.** After E88 and E131, top-k and penalties
  still build the full distribution on the host (~0.85 ms per row); a
  device top-k filter with the host's uniforms and tie rules keeps seeded
  outputs identical. Constrained (tool-call) draws now run on the device
  through the grammar's packed mask (E131); mask misses cost the agent
  replay ~2.6 s, which a larger mask cache did not show in decode time
  (E118).
- **Tokenizer merges** rescan adjacent pairs (`bpe_rank`); Qwen's heap
  merge is the precedent. **`gufo diagnose`** lacks streaming statistics.
- **Encoder SWA replay on a frontier miss.** Approximate; needs its own
  cache and quality contract.
- **Streamed `-j 8` at the native 1M context** sometimes fails at startup
  ("decoder allocation failed"): the sessions are created after the load
  seed, which leaves 4 GiB plus a sixteenth, and eight 1M sessions need
  ~8.6 GiB; the cache is fitted only after they exist (E93). `-c 4096`
  starts reliably.

### 42. 4,096-token prefill batches for streamed builds

`--prefill-batch 4096` keeps its prompt cache (E113) and gives the logits
of 2,048 (E114), and is 5-7% faster on fresh 14-24K-token prompts, but as
the streamed default it slowed a 2.3K prompt by 3% and decode by 0.2-2.1%
through 1.2-1.7 GiB less expert cache (E115): an option. It becomes a
default candidate again if the batch scratch stops costing cache, e.g.
allocated for the first long prompt and returned after it.

### 43. Dense FP16 kernel after the operand swap

Since E114 `attn_q_b`, `attn_output_b`, `attn_output_a` and the shared
down are 3-5% slower in `ds41_dense_f16_cache` (FP16 inputs; the FP32-input
shapes are unchanged), with the same registers and occupancy. It is not
the stores: turning each wave's results through LDS so that a lane stores
one row again was bit-identical and no faster (E114). Next: compare the K
loop's instruction stream before and after with `isa_mix.py`.

### 33. Attention partials written back during the dense GEMVs

Split attention writes 2.64 MB of partials per row and layer; their
writeback slows the following `attn_output_b` (135-193 GB/s in the model
against 198-216 alone, E65). FP16 partials or a merge without the buffer
could reach ~1.5% AR and ~3% at 6 rows, also with changed rounding.
Item 34 (E66) has the same constraint: retain only with reassociation-free
kernels for both AR and verification.


## Quality-affecting and opt-in options

These change outputs and stay opt-in, gated by teacher-forced KL and top-1
against the oracle plus perplexity at 3K and 18.5K
([QUALITY.md](QUALITY.md)). External data:
[EXPERIMENTS.md](EXPERIMENTS.md#items-13-22-26-external-measurements-and-reference-data).

| # | Item | Expected gain (evidence) | Next step |
| ---: | --- | --- | --- |
| 24 | Selection-only miss penalty | Up to the ~5 ms/token of miss-read idle warm; more cold, after long prompts and in verification (mako) | Residency bitmap in the router top-k |
| 13 | Route bias toward cached experts | Xolotl: 13.7 → 30.3 tok/s on a 4090 at ~1 point of top-1; scales logits, so it changes mixing weights too | Same gates and runs as item 24 |
| 23 | Streamed `IQ2_XS` with cache-aware routing | Lower KL than `IQ2_XXS`; slower at ~3/4 of experts cached | Item 27 first |

### 24. Selection-only miss penalty

mako's `--miss-penalty F` subtracts F from the selection score of each
non-resident expert while the mixing weights keep the unpenalized scores,
so it only swaps near-ties (F 0.1-0.2 kept perplexity within noise there).
Here it goes into the router's top-k with a per-layer residency bitmap.
Outputs then depend on cache contents: not bit-identical across cache
sizes or sessions, and a replayed prefix may route differently. Measure
cold, at reduced `--expert-cache-bytes`, after long prompts, on short
continuations and in verification (item 3).

### 23. Streamed `IQ2_XS` with cache-aware routing

ISTA's `IQ2_XS` is closer to the reference than `IQ2_XXS` in mako's
measurements, but here the cache would hold about three quarters of its
expert bytes on a slower SSD. Measure whether item 13 or 24 makes it a
usable higher-quality streamed option once item 27 runs it.

## Features, model support and multi-user serving

| # | Item | Value (evidence) | Next step |
| ---: | --- | --- | --- |
| 16 | Multi-user AR decode | Distinct-prompt `IQ1_M` C8 63.0 tok/s against a ~93 tok/s read ceiling | Card refresh with distinct prompts, then a C8 trace |
| 10 | Batch-level draft width | DSpark trails AR with distinct prompts at C4 (53.6 against 55.5) and C8 (59.3 against 63.0) | Re-measure E27's batch plan after E42 |
| 27 | Streamed kernels for the `IQ2_XS` build | Prerequisite for item 23 | IQ3_XXS down, IQ2_S gate/up paths |

### 16. Multi-user AR decode below the card

The card's multi-user tables give every user the same prompt, which keeps
sessions in lockstep and shares experts (E27). With distinct prompts IQ1_M
AR decodes C1 / C4 / C8 at 24.5 / 55.5 / 63.0 tok/s (2026-10-29); a C8
step routes ~42 distinct experts per layer (E37), and E37/E38/E40/E58
raised the grouped and multi-token GEMVs since
([data](EXPERIMENTS.md#item-16-distinct-prompt-c8-step-2026-10-29)).
Refresh the card with distinct prompts, then trace a C8 step.

### 10. Batch-level draft width

Concurrent sessions share one drafter pass (E42) and one 16-row
verification (E27). E27's v3 batch plan (narrow or skip drafting when the
joint step does not pay) tied on distinct prompts while every session ran
its own drafter forward; E42 removed most of that cost, so re-measure it.
Handing joint rows to the most confident drafts lost (E43).

### 27. Streamed kernels for the `IQ2_XS` build

Two expert placements of the ISTA `IQ2_XS` build have no streamed path: the
IQ3_XXS down in layer 23 (no table-indexed expert GEMV or FP16 tile; E67's
IQ3_XXS work for `UD-IQ2_XXS` may be reusable) and IQ2_S gate/up in
layers 17, 25 and 30 (`fused_pair` and `routed_f16_type` exclude it)
([survey](EXPERIMENTS.md#item-27-iq2_xs-streamed-path-survey-code-trace-not-run)).
Check each against the resident MMQ/MMVQ outputs (`ds41.mmq_typed`,
`ds41_gemv_bench`); existing builds stay bit-identical.

### Adapter follow-ups

Per-request adapter scale (the prompt cache would key on it). LoRA decode
launches: the up terms join the hc expand since E106 (L22 LoRA 240 → 120
launches per token, AR −0.4%); each matrix still adds its down launch.
One-launch kernels lost (E98), so the remaining option is the A dot as an
epilogue of the GEMV producing its input, or one launch for the shared and
routed FFN downs (the shared one already overlaps the streamed host gap).

## Tried and closed

Kernel micro-variants are listed per kernel in
[PREFILL.md lever 3](PREFILL.md#3-routed-fp16-kernels-78-s-busy-46) and
[DECODE.md lever 5](DECODE.md#5-kernel-bandwidth-39-mstoken-on-the-critical-path-7).

| Item | What was tried | Outcome | Ref |
| --- | --- | --- | --- |
| 21. Two-layer-ahead decode prediction | Lead 2 and every-layer lead 1 | Lead 2 predicts fewer misses (74.0% against 79.3%); every-layer lead 1 slower on the 3K probe (+7.1%) and in `--spec-bench`. Continues as item 37 | E89, E60 |
| 32. GPU idle during read waits | Clock sensors across streamed DSpark decode | No clock drops during waits | [data](EXPERIMENTS.md#item-32-gpu-clocks-during-read-waits) |
| 18. Small decode kernels | hc mix variants, fusions, multi-token mixes; router loads | Mix hidden on a side stream (E57), fusions and multi-token mixes no gain (E49, E55, E63); router at 211 GB/s. Fusions left in item 4 | E40, E49, E55, E57, E59, E63 |
| 20. `markov_select` | Exact pruning bound | Still scores 99.2% of rows; at ~92% of the read ceiling. Item 38 remains | E47 |
| 11. Engram reads under a full cache | Skipping every read (upper bound) | +0.8% warm | E28 |
| 12. Expert-contiguous repack | Read benchmark on real spans | ~1% | E30, E70 |
| Graph replay | Dependent launch chains | ~0.13 µs per launch | E44 |
| Q8_0 matrix cores at verification widths | Split GEMV vs one token's time, 1-16 rows | 1.03-1.10× at 4-8 rows; ceiling ~0.5% of decode over the 1-6-row cycle mix; rounding change | E133 |
| Dense decode GEMV slices, formats, nontemporal loads | DRAM-cold sweep; loads in the model | No Q8_0 gap; slices worth 0.1-0.4%, under the rounding bar; nontemporal no gain | E59, E65, E69, E90 |
| Verification grouped-GEMV redundancy | Distinct experts per layer | At the arithmetic minimum | E89 |
| Decoder replay read-ahead and protection | Simulated policies | Read-ahead recovers 14-15% (drive-bound); protection costs ~4 GiB of prompt reads per GiB | E86 |
| Layer-major prefill | `wip/ds41-layer-major-prefill` | Reads −32%, GPU +4-6 s, known race; shelved | [data](EXPERIMENTS.md#layer-major-prefill-shelved) |
| Routed FP16 tiles, stages, sign masks, grid packing, scheduler | 32/128-slot, cross-expert, wave64, 32/128-value stages | None faster; tile direction closed by arithmetic | E8, E35, E76-E81 |
| Dense FP16 prefill tiles and loads; BF16 64-row tiles | Fragment prefetch, LDS fragments, paired loads, rotated copies; wide BF16 tiles | Exhausted; in-situ `q_b` gap is contention; BF16 tiles slower | E52, E53, E74 |
| Prefill attention fixed costs | Ablations | Nothing cheap; FP16 Q is item 40 | E53 |
| 4,096-token prefill batches as default | Native pp | +2-3%, memory cost; option | E16 |
| Short prompt passes: earlier prediction, undrained adoption | Served follow-ups | Within noise | E72 |
| Joint rows by confidence, draft trimming (E27 v2), pre-draft gates | Concurrent and single DSpark | C8 −1.1%; within noise; real gates lose | E27, E43, E64 |
| DSpark drafter size and quantization | Smaller and larger drafters | Current drafter kept | E62 |
| Prefill prediction lead time | Two or three layers ahead | One ahead already lands early enough | PREFILL.md |
| Dense rescoring on the bounded replay | Indexer bench | ~25 ms per 128K prompt; item 5 for `--exact-prefill` | E34 |

## Done

Retained work that closed or changed an item or lever; everything else is
in EXPERIMENTS.md.

| Item | Result | Ref |
| --- | --- | --- |
| Streamed expert-pruned builds (item 41) | Coder streams with the kept experts' ranking; scores byte-identical to resident; default cache at resident speed, 30 GiB cache tg128 14.5 → 21.8 tok/s as it warms | E96 |
| Higher-precision Engram tables (item 14) | MXFP8 reads cost the same as IQ4_NL (one random read per row); host dequantization 9× faster, bit-identical; served gaps follow the text; Q5_K/Q6_K tables untested | E107 |
| LoRA adapters (item 26): `--adapter` on `attn_output_b`, `ffn_down_shexp`, `ffn_down_exps` | Bit-identical without one; L22 LoRA 3K KL 0.012, heretic 0.018; decode −1.6 to −3.2% | E98 |
| Direction adapters (item 25): `--adapter`, mako files and rank-1 LoRAs | Bit-identical without one; 3K perplexity ±0.002 nats, KL 0.012-0.015; decode −0.5 to −0.9% | E97 |
| Greedy DSpark width policy (DECODE.md lever 1) | Chat 16.5-16.9 → 17.6-18.9, code 26.6-27.4 → 28.3-28.7, long 24.7-26.2 → 26.4-26.7 tok/s | E87 |
| Grammar-constrained draws on the device (item 15, DECODE.md lever 4) | Agent replay decode −1.2%, tool-call replies −1.9%; identical replies | E131 |
| Top-p softmax totals on the device (DECODE.md lever 4) | Sampled DSpark +0.3 to +1.1 tok/s, AR sampled +0.1 to +0.2 | E88 |
| Eviction victim lists, exact pops (item 31) | Scan removed; served speed unchanged (scan ≤ ~0.15 ms per admission) | E91, E92 |
| Eviction by frequency class (item 31) | Agent replay −0.9% total, −6% reads; code prompt after a chat −10%; cold 2K/15K +1-2% | E105 |
| Dense FP16 workgroups in cache-sized chunks; BF16 operands without rocWMMA | `attn_q_b` 5.54 → 4.56 ms, router 1.15 → 0.90 ms; fresh 20-24K prompts −1.6 to −2.1% | E108, E109 |
| Two-wave prefill, gate+up then down | pp2048 1,109 → 1,160 tok/s; served −1.7 s / −1.2 s, then −1.2% | E82, E84 |
| Layer-0 prediction, Engram rows in layer order | Encoder read idle 1.53 → 0.96 s over two documents | E83 |
| Engram lookahead | Engram waits 1.21 → 0.50 s; −2.1% on a 58K prompt | E85 |
| One decoder replay per served prompt | −2.4 to −2.7 s per fresh 14-24K-token document | E73 |
| Next-layer prediction for mid-size prefill batches | +3.5% warm pp2048, +11-15% first pp512 | E68 |
| Narrow and grouped BF16 tiles | pp20-pp64 +6-7%; router 1.27 → 1.16 ms per call | E71, E75 |
| Expert reads split across two drives | tg +7.6%, pp512 +16% | E61 |
| 3 (part). Prediction for verification batches | +3.5-6.7% cold, neutral warm | E31 |
| 10 (part). One drafter pass, one joint verification | DSpark C4 +4.3%, C8 +7.3% (E42) | E27, E42 |
| 1, 2. 6-column GEMVs; fused SwiGLU and Q8_0 pairs in the split GEMV | 5-6-row verification +0.5-0.8%; joint 16-row pass 177.7 → 174.4 ms | E29, E36 |
| 7 (part), 8. Pinned embedding staging; batched 7-8-token prompts | ~6.0 → ~4.5 ms per batch; pp8 121.8 → ~99 ms | E32, E33 |
| Decode GEMV and launch work | `q_b` four slices, hc side stream, shared expert during the route read, global expert loads, router loads | E54-E59, E69 |
| Routed decode experts | Paired IQ1 downs, single-slot IQ1 experts, IQ3_XXS/Q2_K downs, IQ2_S fused gate/up | E46, E51, E67 |
| Verification argmax on the device | Host scans behind ~680 µs of idle per step removed | E50 |
| Routed FP16 staging and LDS lookups | IQ2_XXS gate+up 24.5 → 20.4 ms (bench) | E25 |
| Cost-aware draft prefix | Streamed greedy +3.76% | E18 |
