# Gemma 4 continuation adapter (design)

**Status:** design and prototyping on `feat/gemma-cache`; nothing below is
wired into serving yet. Gemma 4 serves through the legacy `ContinuationCache`
with its own device-resident `SessionSnapshot`. Upstream's switch-over
([card 19](../../cache-redesign/cards/19-switch-over.md)) deletes that cache
and the runner snapshot hooks, so Gemma needs a `cache::Adapter` for the
component cache before that change merges. The component cache also brings
unique-byte accounting (the legacy budget charges shared snapshot blocks once
per snapshot), cost-aware retention, the chunked crash-safe disk tier and
per-request `component_cache` reporting.

Three shared-layer changes come first. Each is fork-local until it has
served Gemma, then a candidate for upstream: the window-rows kind is open for
upstream's DeepSeek adapter ([card 17](../../cache-redesign/cards/17-deepseek-adapter.md)
lists sliding windows as undecided), and in-place same-slot restore is an open
item in [todo.md](../../cache-redesign/cards/todo.md) (latency flags row) and
card 19 ("needs an adapter protocol change and is not implemented").

## State and sizes

Both targets use a 1,024-position sliding window and global layers whose keys
keep only the 128 rotated dimensions per KV head (the rest are derived from V).
Sizes per position, binary16:

| Component | 26B-A4B | 31B | Kind |
| --- | --- | --- | --- |
| Global K / V, per layer | 512 / 2,048 B (5 layers) | 1,024 / 4,096 B (10 layers) | append rows, 256-row chunks |
| Global total | 12,800 B (12.5 MiB per 1K positions) | 51,200 B (50 MiB per 1K) | |
| Sliding K / V, per layer | 4,096 / 4,096 B (25 layers) | 8,192 / 8,192 B (50 layers) | window rows, 1,023 rows |
| Sliding window total | 200 MiB | 800 MiB | |
| Tokens | 4 B | 4 B | append rows |
| Frontier hidden (FP32, MTP drafter input) | 11,264 B | 21,504 B | private |
| Logits (FP32) | 1 MiB | 1 MiB | private |
| Image records (offset, rows, prefix identity) | 40 B per image | 40 B per image | private metadata |

The formula reproduces a 26B v3 restore of 22 tokens to the byte (5,847,164).
A 31B checkpoint at 100K tokens holds 4.8 GiB of global rows and 0.8 GiB of
window rows; consecutive checkpoints of one history share all but their new
chunks and tails.

Components are per layer (31B: 120 KV components; 26B: 60) because KV is
layer-major and a borrowed range must be one contiguous span. Not captured:
the prompt-lookup index and draft-vocabulary table (rebuilt from tokens), the
draft-policy calibration (model or request owned), lookahead rows, sibling
spare keys and the pending token (request owned or past the frontier). The
drafter reads the target's KV, so MTP adds no draft components; only the
frontier hidden state. Sampler state stays in the request, as for Flash-Next.

## Shared-layer changes

### Window rows

`ComponentKind::kWindowRows` with `ComponentDescriptor::window_rows` (1,023 for
Gemma). A checkpoint at position *n* references only the chunks from
`ChunkStart = floor(max(0, n − window_rows) / rows_per_chunk) · rows_per_chunk`
to the last complete chunk, plus a private tail. `ComponentPosition` stays
`{id, n}`; the first row is derived, never stored. Window chunks are shared by
identity between consecutive checkpoints of a branch, and chunks below every
retained checkpoint's window are released.

About 25 sites in 10 shared files count chunks from row 0 or align chunks by
vector position. Two would misbehave badly if missed:
`Checkpoint::NewPayloadBytes` would count every row below the window and push
long-context captures to the disk-only path, and the three retention byte
functions (`UniqueBytes`, `UniquePayloadBytes`, `RetainedPayloadBytes`) align
peers by index; the second feeds a running total that could underflow. They
must align by absolute chunk index. The disk manifest gains the window size
only for windowed components: write version 3 when one is present, keep reading
version 2, so existing Flash-Next directories stay valid. The slot's
`borrowed_end` fast path never skips ring evictions (they are always below
it), so the guard needs a per-component lower bound of live registrations.

Chunk size for window components is open: 64 rows bound each checkpoint's
private tail to 63 rows (31B 52 MB, 26B 13 MB; v3's 256-position sliding
blocks left about 210 MB per 31B snapshot to copy), 256 rows mean a quarter of
the chunk files on disk.

### In-place same-slot restore

A swipe or regenerate re-sends a prompt equal to a checkpoint the slot that
produced it still holds; it has only appended the reply. Today the component
cache preserves the slot's borrowed rows to host memory, resets the slot and
loads everything back (or restores into another slot). The design:

- The slot tracks an intact interval per component from guard calls, updated
  before `LeaseState::Guard`'s early exit: a write at or below the interval's
  start raises it (ring eviction), any other overlapping write lowers its end.
  Reset, release and destruction clear it; capture and successful restore
  declare it.
- `ExecutionHistory::Holds(chunk)` proves chunk identity; a bounded record of
  the branch's checkpoints proves tails. A chunk is resident when the slot
  generation matches, the history holds it and it lies in the intact interval.
  Rewritten rows always produce new chunk objects, so identical rewrites are
  never mistaken for intact rows.
- `LeasedSlot::AcquireInPlace(generation)` claims the slot without
  `PrepareReassignment` or `Invalidate`, and falls back to a cold acquire when
  the generation moved. Borrowed ranges at or past the new frontier (the
  discarded reply) are preserved at acquire, not in the first decode step.
- `Capabilities::in_place_restore` and `BeginRestore(slot, positions, kept)`:
  the adapter guards and loads only the complement of the kept ranges and
  counts kept ranges as covered in `Validate`. Private components always load.
- Selection prefers the slot holding the rows when the checkpoint's tokens are
  a prefix of that slot's live tokens (a same-conversation sibling) and the
  slot is idle; live-frontier reuse stays first.

Gemma's ring holds 3,328 slots (window + 2,048-row prefill chunk + 16, rounded
to 256), so a checkpoint's window survives a reply of up to about 2,300 tokens
in place; longer replies reload only the evicted window chunks. Estimated size:
about 650 production lines and 800 test lines, shared layer plus adapter.

### Batched gather transfers

Measured on gfx1151 with 1 GiB per run, host memory prepared as
`CommittedBackingPool` does (mmap, no THP, populated, `hipHostRegister`), GB/s
([cache_transfer_bench](../../../tools/bench/cache_transfer_bench.hip)):

| Piece | `hipMemcpyAsync` H2D | D2H | D2D, 16 streams | Gather kernel D2D | from mapped host | to mapped host |
| --- | --- | --- | --- | --- | --- | --- |
| 128 KiB | 21.0 | 22.1 | 91.8 | 108.2 | 103.0 | 105.2 |
| 256 KiB | 33.7 | 34.5 | 104.6 | 109.1 | 103.7 | |
| 1 MiB | 61.1 | 60.6 | 108.1 | 109.7 | 104.0 | 107.1 |
| 2 MiB | 70.6 | 69.2 | 107.1 | 108.2 | 103.1 | |
| 8 MiB | 79.7 | 77.5 | 105.9 | 107.1 | 102.1 | 104.5 |

Host copies through `hipMemcpyAsync` run at the same rate on 1, 4 or 16
streams. Gemma's chunks are 128 KiB–2 MiB, where one gather kernel over
registered host pages is 1.5–5× faster and as fast as device memory. The
adapter therefore queues `CopyRowsOut`/`CopyRowsIn` per cache stream and
submits one gather kernel when a completion is waited on or the stream is
synchronized. No shared caller polls `Completion::Ready` without waiting, so
deferred submission is safe with the current code. The checkpoint backing
stays host-addressable `CommittedBackingPool` memory, which the disk tier and
CRCs require.

Upstream's 64-stream `hip::TransferPool`, created after the model has loaded,
does not change Gemma decode: 26B AR over HTTP measured 55.63–55.66 tok/s with
and without it in three alternations (an extra stream created during model load
had cost 3.4% before).

## Gemma adapter

- **Positions:** tokens, global and window components at the target frontier;
  private hidden, logits and metadata at the same boundary.
- **Guards:** `Executor::Forward` after its capacity check and `MoveKey`, where
  `KvCache::written_from/to` are set today, plus `Session::Sync` rewinds,
  restore, reset and destruction. A write of positions [a, b) also guards
  [a − ring, b − ring) of every window component (the rows it evicts). Guards
  run under the model mutex, so preservation copies must not take it.
- **Capture ordering:** record an event on the executor stream after the
  session's last write and make the transfer stream wait on it; borrowed views
  need the same before `copy_observed_rows`. Logits come from the device buffer
  while `logits_on_device_` is set.
- **Prefill planning:** `PlanPrefill` mirrors `Session::Extend` (2,048-row
  chunks, remainders of 16 rows or fewer absorbed) and never places a boundary
  inside an image span. In-pass checkpoints generalize today's prompt
  lookahead: rows below a boundary survive the rest of the pass, so a
  checkpoint needs only that row's logits and hidden state.
- **Identity:** today's compatibility identity plus a component layout
  version, prefill chunk width and the derived-key scheme. Images stay in the
  input identity (`cache_prefixes`); a restore rebuilds the session's image
  records so `Sync` does not cut reuse at the first image.

## Open decisions

- Swiping back after an in-place restore: the discarded reply's rows are gone,
  so continuing from it re-prefills the reply unless a checkpoint covered it.
  The legacy path behaves the same unless it froze the live frontier.
- How window bytes count in retention's "cheap checkpoint" rule and in
  `GridLimit`, which today consider only private-state bytes.
- Window chunk size (64 or 256 rows).

## Plan

1. Window rows in the shared layer, with an opt-in ring-backed component in
   the fake adapter; `tests/cache` and the Flash-Next adapter and serving tests
   unchanged.
2. Gemma adapter and runner glue for AR and MTP with batched gather transfers;
   a native test that a component restore reproduces a v3 restore's logits,
   hidden state and next tokens, including ring wrap and images.
3. In-place same-slot restore, with a swipe replay.
4. In-pass checkpoints and the disk tier.
5. Functional cache, continuation and image suites on 26B and 31B, AR and
   MTP, against the legacy path; then remove Gemma's legacy snapshot hooks.
