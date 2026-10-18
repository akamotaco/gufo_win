# Gemma 4 continuation adapter (design)

**Status:** prototype on `feat/gemma-cache`. Steps 1–3 are done: window
rows in the shared layer, Gemma serving through the component cache,
in-place restore, batched gather transfers and the capture and restore
timing fixes (results below). Before
this, Gemma 4 served through the legacy `ContinuationCache` with its own
device-resident `SessionSnapshot`. Upstream's switch-over
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
layer-major and each component's rows must be one contiguous span. Not captured:
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
- **Capture ordering:** the executor reports every call that writes a
  session's rows, hidden state or logits (`ContinuationFences`). Before it
  queues one, the adapter records an event for each other session written
  since its last event; a copy of that session waits for the event, not for
  the work queued after it (another session's prefill). Copies of the session
  that wrote last wait for everything queued. Logits come from the device
  buffer while `logits_on_device_` is set.
- **Copy stream:** the adapter gathers on its own high-priority stream and
  completes transfers with events on it. The cache's pooled streams carry no
  work: some share a hardware queue with the executor stream, and a
  completion recorded there waited behind a whole queued prefill chunk.
- **Rows are copied, not lent:** a lent row is copied anyway before the slot
  is reassigned or restored in place, one chunk per wait (8–12 ms for a
  22K-token prompt's global rows), while a capture gathers it at about
  100 GB/s.
- **Prefill planning:** `PlanPrefill` mirrors `Session::Extend` (2,048-row
  chunks, remainders of 16 rows or fewer absorbed) and never places a boundary
  inside an image span. In-pass checkpoints generalize today's prompt
  lookahead: rows below a boundary survive the rest of the pass, so a
  checkpoint needs only that row's logits and hidden state.
- **Identity:** today's compatibility identity plus a component layout
  version, prefill chunk width and the derived-key scheme. Images stay in the
  input identity (`cache_prefixes`); a restore rebuilds the session's image
  records so `Sync` does not cut reuse at the first image.

## Decisions

- **Swiped-away replies:** a same-conversation retry restores in place and
  discards the reply the slot holds; no checkpoint keeps that reply. Gemma's
  template never lets a later chat turn reuse a reply: history drops the
  reasoning before the last user message, and with thinking off the
  generation prompt's empty reasoning block is not rendered in history. On
  26B, a second chat turn reused 20 of the first turn's 27 prompt tokens
  (thinking off; the reply ended at 96) and 30 of 30 (thinking on). Only a
  tool-call step with thinking on continues past its reply (158 of 158), and
  a retry of that step replaces the reply. A reply-end checkpoint would copy
  1–8 ms per retry and never be restored.
- **Window chunks of 64 rows**, global chunks of 256. Persistence time with
  four times as many chunk files is measured before it is kept.
- **Window bytes count as fixed per-checkpoint state** in retention's "cheap
  checkpoint" rule and in `GridLimit`, beside private state. Gemma checkpoints
  below about 2,050 tokens become cheap (window bytes exceed eight times their
  row bytes), and the grid limit is 3 for 31B and 4 for 26B at a 32 GiB budget
  (31B: none at 8 GiB, where each grid checkpoint would take a tenth of it).
  Counted as rows instead, Gemma checkpoints would never be cheap and every
  cold prompt would take four grid checkpoints (up to 3.2 GiB of window rows
  on 31B).

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

## Results so far

- **Native:** `gemma4.continuation` restores a 1,980-token checkpoint (ring
  1,280 slots, so its window has wrapped) into another slot and into its own
  slot after the session moved a whole ring further. A capture of the
  restored slot equals the original checkpoint byte for byte, and greedy
  decoding reproduces the original logits and tokens; with the MTP drafter,
  each fixed-length cycle drafts and accepts as before the restore. Passes on
  26B-A4B and 31B UD-Q4_K_XL. A window row misplaced only on load, or a
  corrupted hidden state, fails it.
- **Functional, 26B-A4B MTP (`--sessions 2 --context 32768`):** `cache`,
  `cache-edits`, `cache-growth`, `cache-depth`, `cache-rotation`,
  `cache-concurrency`, `cache-shared-prefix`, `long-context` and
  `state-edges` pass.
- **Against the legacy cache** (the production build of `1b7b154e` as the
  control, one run each): 2,479 of 2,952 timing measures pass. The other 473
  are mostly restores, queueing and first tokens 5–8 ms slower per request
  (the prefix is copied where the legacy cache skipped blocks the session
  still held) and shared-prefix captures about 8 ms slower (window chunks are
  copied at capture). Three requests re-prefill 3–7 tokens the legacy cache
  served: under a full 32 GiB budget (both caches ran full in this run) the
  component cache evicted their full-prompt retry copies first.
- **In-place restore:** a branch records the checkpoints it captured or was
  restored from, and the adapter reports which of their rows the slot still
  holds (`Adapter::ResidentRows`: global rows below the branch's first
  rewrite, window rows whose ring slots still hold them). A swipe of a
  16,159-token 26B prompt loads only the private state (1.06 MB): restore
  0.64–0.92 ms and first token 1.2–1.4 ms, against 0.19–4.4 ms and 0.6–4.8 ms
  for the legacy cache. Against the same legacy control, 2,678 of 2,952
  timing measures pass (2,479 before); the remaining flags are mostly
  shared-prefix captures (about 8 ms, window chunks copied) and queueing
  (3–4 ms), and one request (`long_json_stop`) still loses its retry copy to
  byte pressure. `gemma4.continuation` covers both in-place cases: after a
  short reply only the private state loads; after a ring wrap the global rows
  stay and the window reloads.
- **Batched transfers:** the adapter queues the device copies of every
  transfer and launches them as one gather kernel at the first wait (the
  other queued streams wait on its event); unaligned or pageable buffers
  copy singly. Against the legacy control 2,709 of 2,952 timing measures
  pass (2,678 before); shared-prefix captures are about 5 ms slower (8 ms
  before). Still open: queueing on shared-prefix requests (+4.7 ms), decode
  beside concurrent captures in `cache-concurrency` (+9.5 ms, unchanged by
  batching), and `long_json_stop`'s retry copy.
- **Capture and restore timing:** three changes, against the same legacy
  control. Rows are copied at capture (above), which removes the 8–12 ms of
  one-chunk-per-wait preservation from slot reuse and in-place restores:
  `cache-shared-prefix` restores of 200 MB take 3.2–3.6 ms (9.5–16.7 ms
  before; legacy 2.4–2.9 ms) and cold queueing 0.9–2.4 ms (up to 14.6 ms).
  Copies order after their own session's writes on the adapter's stream: a
  capture during another session's prefill (`cache-concurrency`
  `unrelated`) takes 4–5 ms instead of 270–480 ms (legacy 430 ms), and both
  requests reach their first token about 42 ms sooner than legacy. Without
  the per-session events the copy stream alone still waited 612 ms. A
  restore picks the slot it would have picked anyway, and keeps that slot's
  rows when its live frontier extends the checkpoint's tokens and inputs.
  26B AR decode stays at 55.64–55.66 tok/s. Against the legacy control,
  2,804 of 2,952 timing measures pass (2,709 before). The remaining flags:
  - Prefills split once more where a prompt shares more with a stored prompt
    than with its restored checkpoint (`cache-edits` edits, two
    `cache-concurrency` requests: 1,187 + 4 + 7 tokens instead of
    1,191 + 7, +37–44 ms on 26B). Upstream's shared-prefix checkpoint stops
    the pass at the divergence point. Fixed below (prompt lookahead).
  - Decode order: when two requests become ready together they decode
    batched (45 ms per step for both) instead of one after the other
    (35 ms each); which happens varies between runs on both paths.
  - Legacy restores report 0.3 ms for 263 MB, less than the copy takes; its
    time to first token for those requests is within 1% of the component
    path.
  - Under a full 32 GiB budget a capture spends about 2.4 ms per eviction in
    retention (up to five per capture); the same capture took 11.8 ms in one
    run and 33 ms in another (legacy 13 ms).
  - Acquisition runs three to four prefix-index lookups, 0.4–1.4 ms more
    queueing than legacy on cold requests.
  - `long_json_stop` still loses its retry copy under the full budget.
- **Prompt lookahead at later stops:** a prefill step that stops at a
  checkpoint evaluates the tokens after it in the same forward, as far as
  the forward has rows left (whole, or up to a later stop) and the step's
  budget allows; it used to ride only a tail of at most 16 tokens, adopted
  only whole. Later steps adopt it at each stop the runner announces (the
  logits and hidden state of those rows are kept, one stash slot each), and
  a step between stops keeps the shared rows and evaluates only its last
  token. The ridden rows stop short of the frontier's window and its first
  64-row chunk, so a capture there still reads them. The `cache-edits` edits
  now take one forward again: 334.5, 399.7 and 335.5 ms prefill against
  legacy's 342.4, 409.6 and 339.5 ms (+37–44 ms before). `gemma4.target`
  checks each adopted frontier against one that evaluated its prefix, and
  `gemma4.continuation` the adopted hidden state; both fail with a wrong
  logits row or stash slot. Paired runs against the previous build match on
  long prompts (7,004.7 / 7,008.5 ms, then 7,233.1 / 7,249.4 ms: the host
  drifts 3–4% between pairs, so long-prefill flags against an older legacy
  run are not from this). 26B AR decode stays at 55.63 tok/s.
- **Images (26B MTP with its sidecar):** `image-inputs`, `tool-images`,
  `conversation` and the image cancellation history pass. The component
  path had placed a grid checkpoint inside an image span, which Gemma
  prefills whole (500 "invalid prefill token count"); component-planned
  checkpoints now stop before such inputs, as the legacy grid did.
  `image-count` fails on its low-effort thinking check with the legacy build
  too: the reply's 256-token budget ends inside the reasoning, leaving no
  JSON text.

