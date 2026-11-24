# DeepSeek V4.1 Flash SSD expert streaming

How the engine runs a backbone whose routed experts do not fit in memory:
the dense weights stay resident and the routed experts are read from the
SSD into a bounded cache. The cache, reads and prefetch are Gufo's
experimental expert store (`gufo/ext/experimental/expert_stream.hpp`).
`runtime/expert_store.{h,cpp}` describes DeepSeek's expert tensors to it and
holds the DeepSeek policy (headroom without serving, built-in ranking,
pruned builds, histograms); the decoder's expert paths are in `runtime/v41_decode.cpp`.
Measured results, including the
rejected designs, are in [EXPERIMENTS.md](EXPERIMENTS.md#ssd-expert-streaming-m3);
open work is in [BACKLOG.md](BACKLOG.md).

## Why and when

| Build | Size | Fits resident? |
| --- | ---: | --- |
| `IQ1_M` backbone, 4 shards | 112.4 GiB | yes, 112.27 GiB GTT |
| `IQ2_XXS` backbone, 4 shards | 128.7 GiB | no: 5.31 GiB dense weights resident, experts streamed |
| Unsloth `UD-IQ2_XXS` backbone, 5 shards | 153.9 GiB | no: 7.04 GiB dense weights resident, 146.1 GiB of experts streamed |
| `Engram-IQ4_NL.gguf` | 103.0 GiB, 144 B rows | always read from disk |

MemTotal is 124.4 GiB (128 GiB minus the 1 GiB UMA carve-out and about
2.6 GiB of firmware and kernel reservations); `amdgpu.gttsize=126976`
already covers it. The model has 40 layers × 384 routed experts (top 6 plus
one shared expert). Expert sizes differ by layer, because the builds mix
block types: `IQ1_M` has six sizes (6.59–8.26 MiB, mean 7.11) and `IQ2_XXS`
eight (7.38–9.76 MiB, mean 8.20); Unsloth's `UD-IQ2_XXS` seven
(7.29–11.51 MiB, mean 9.74), with IQ3_XXS and Q2_K downs. A decode token reads six experts in each of
40 layers, so streamed decode depends almost entirely on the route-hit rate:
keeping about 90% of resident speed needs about 99% of routes to hit.

Streaming is opt-in. `--stream-experts` (`serve`, `bench`, `prompt`) keeps
dense weights resident and streams the routed experts; `--expert-cache-bytes
N` caps the expert cache (default: the memory the server's sessions leave,
minus 2 GiB) and requires `--stream-experts`. Without the flag, a model that does not fit
the available memory (free CMA excluded) fails to load and states what it
needs, rather than running slower. While streaming, weight placement counts
only the resident tensors, so the dense weights go through the device heap.
`--stream-experts` also runs `IQ1_M` in less memory, more slowly.

## Expert cache

- **Slots by size class.** Each distinct per-layer expert size (gate, up and
  down together, rounded to 256 B) is a class. The cache lives in 2 MiB
  device pages (`hipMemCreate`), which the kernel does not swap; each class's
  slots lie back to back in one reserved address range, mapped in 128 MiB
  chunks as it grows, while the cap, the grant and 1 GiB of free host and
  device-heap memory allow. An expert takes a free slot of its own class, or
  the smallest larger free slot once its class can no longer grow.
- **Admission and eviction.** Each expert counts its uses (its route count,
  at most 4 per call), decaying with a half-life of 2,048 tokens of its
  layer's clock, which a prompt chunk advances by at most 16; at each use it
  takes frequency class floor(log2(uses)), at most 7. Seeded experts start
  at half their ranking weight. While free capacity remains, every read
  expert is admitted. Once the cache is full, a large prefill admits only
  its final 16 rows and its unusually frequent experts (count ≥ max(4,
  tokens / 32)), so a scan does not displace the decode working set. The
  victim is the slot of the same or a larger class with the largest
  age / 2^class, so an expert used twice as often may stay unused twice as
  long (E105); a larger expert never takes a smaller slot. A batch's
  own hits are never victims: its weight table points at their slots, and
  the admission copies land before its kernels run.
- **Ranking.** The cache is filled from the built-in ranking of
  all 15,360 experts (`expert_rank.inc`, imported from Xolotl's calibration
  trace by `tools/ds41/import_expert_rank.py`), or from
  `MODEL.gguf.expert-histogram` (`DS41_EXPERT_HISTOGRAM_V1`, built by
  `tools/ds41/expert_histogram.py` from oracle `routes.npz` files) when that
  sidecar exists; the store gets it when it is created. Experts the ranking
  omits follow its ranked ones, interleaved across layers. Nothing is cached
  until the cache is sized; filling it does not count in the statistics.
- **Fit after the sessions.** The runner declares an expert-cache claim
  (Gufo's `TextRunnerExpertCacheClaim`): its store, 2 GiB of headroom, and
  the RAM prompt cache's automatic share per session. Its total is the
  backing one checkpoint of a whole context takes, from the continuation
  layout (`ContinuationAdapter::FullContextBackingBytes`), once per session,
  or once in all with `--cache-disk`, whose checkpoints restore the
  conversations the RAM cache evicts (16-30 ms for about 38K tokens). With
  DSpark one checkpoint is 0.29 GiB at a 256K context and 1.11 GiB at 1M.
  With `--cache-disk` the total is at least 1 GiB: Gufo charges the disk
  checkpoints' token lists to the RAM cache's metadata, about a tenth of its
  budget, and a smaller budget fills with them after one or two long agent
  sessions, after which checkpoints are refused (E128, gufo-org/gufo#1329).
  Once `gufo serve` has created its sessions, it commits the prompt cache's
  whole backing and grants the store the cache plus the free memory, minus
  the headroom. The store allocates the prefetch stage out of the grant, takes
  the rest as its limit and fills it once from the ranking (about 25 s for
  101 GiB), so the first prompt does not grow the cache. `gufo prompt`,
  `gufo bench` and the load probe size it themselves right after loading
  (`ds41_expert_store_fit_available`), leaving 4 GiB free. The RAM prompt
  cache keeps what its backing committed by then: growing it later needs
  4 GiB of device heap free, which the fitted expert cache does not leave
  (E126); the disk cache serves the rest.
- **Host pressure.** The fitted cache leaves about 2 GiB free on purpose, so
  while experts stream Gufo measures host pressure against the store's
  1 GiB floor rather than its 4 GiB host reserve. Below the floor, the idle
  server first frees the cache's newest chunks, moving the experts in use
  out of them into the slots of the coldest, then prompt-cache memory, and
  refuses new requests below half the floor. Once the floor, the
  released bytes and 256 MiB are available again, the cache may regrow to
  its grant (E127).
- **Wide prefill.** A prompt with at least 8,192 uncached tokens runs in
  4,096-token passes: before it, the server releases from the cache what the
  rows past 2,048 need (1.47 GB with their pinned host rows) beyond the
  floor and a 512 MiB margin, and restores it when the prompt ends,
  fails or is cancelled, or at idle; the cache regrows as experts miss.
  One prompt holds the rows at a time, and one that cannot get them stays
  narrow without releasing more of the cache (E134).

## A layer's prepare

`ds41_expert_store_prepare` runs once per MoE layer and evaluation (decode,
prefill batch, DSpark verification or a joint evaluation):

1. Count the routes, update the scores and find the misses.
2. Resolve the background job (below): adopt it if it targets this layer,
   otherwise drop or park it.
3. Point each expert at its weights through **per-layer device tables**
   (gate, up, down; 40 × 384 pointers each): its cache slot when resident,
   else the demand stage row or slot its read fills, or its row in the
   adopted job's stage. The kernels
   index the table, so resident experts are read in place with no staging
   copy. A host mirror records the entries; a layer's table is uploaded
   asynchronously only when an entry changed, which a warm cache rarely
   does.
4. For batches of up to 96 route rows (16 tokens: decode, verification and
   joint rows) that both hit and miss, upload **hit and miss row arrays**.
   Each keeps the route positions with −1 for the other's experts, so the
   launches are split: while the reads stream, an overlap callback computes
   the shared expert and the resident experts' gate+up and down; after the
   reads join, a second launch completes the read rows' gate+up and down.
   Every row keeps its single-launch arithmetic, so outputs are bit-identical
   to one launch. Other batches launch once over all routes.
5. Run the phase's reads, then copy each admitted expert from its stage row
   into a slot.

The routes copy that starts the next prepare orders every upload before
its pinned mirror is rewritten.

## Reads

- **ReadPool.** Eight worker threads read with `O_DIRECT` into their own
  pinned buffers (falling back to buffered reads when direct I/O is not
  available; the last read may end at an unaligned EOF) and upload through
  their own non-blocking streams, so staging does not queue behind the
  layer's kernels. Each upload is awaited before its buffer is reused.
- **Tasks.** Missing experts of one projection that are adjacent in the file
  merge into reads of at most 32 MiB, ordered by shard and offset. The GGUF
  stores each layer's experts as three projection-major blocks, so one
  expert is three reads about 1 GiB apart.
- **Demand stage.** 64 rows (0.6 GiB), assigned per call in expert order at
  the layer's own row size, so experts adjacent in the file stay adjacent
  in the stage. One buffer suffices: the routes copy orders a layer's
  demand reads behind the previous layer's compute. A batch with more
  misses than rows (a cold prefill batch) reads the experts it admits
  straight into their slots and gives the rows to the others; those past
  the rows fill free slots or take a lent one, a spare slot or the lowest
  keep value's. Lent slots stay unoccupied, return to the spare slots at
  the next call, and are the first an admitted expert takes, so the
  admission policy does not change (E119).
- **Prefetch stage.** A compact 96-row buffer in two 48-row halves,
  allocated out of the grant before the cache is filled. Each prepare opens
  the next prediction window, and windows alternate halves, so the layer
  in compute and the one being prefetched never share one, also where the
  last layer wraps to the first (E130). A job's experts take rows in its
  window's half, addressed with the allocation stride (the stride of the
  largest expert). A job beyond 48 experts keeps the experts the most
  predicted routes use (E104); the excess stays a demand read. A job starts
  only while its window is the next one to prepare, checked under the read
  pool's lock: an overtaken job's half may hold the rows of the layer in
  compute.

## Split reads across two drives

A second NVMe can serve part of every miss. `tools/ds41/pack_expert_tails.py`
copies the tail of each routed expert projection (by default the second
half: `--head 0.5` is the share still read from the GGUF) into one file:

```sh
python3 tools/ds41/pack_expert_tails.py \
  --model /models/IQ2_XXS-00001-of-00006.gguf \
  --out /second-drive/IQ2_XXS.expert-tails
ln -s /second-drive/IQ2_XXS.expert-tails \
  /models/IQ2_XXS-00001-of-00006.gguf.expert-tails
```

When `MODEL.gguf.expert-tails` exists beside the first shard (the file or a
symlink to it), every projection read becomes a head read from the GGUF and
a tail read from the file, queued together, so both drives work on each
miss; prefetch jobs split the same way. The symlink is resolved once when
the file is opened. The file is about half the expert bytes (61.6 GiB for
`IQ2_XXS`, 61.7 GiB for UD-`IQ1_M`, 73.0 GiB for UD-`IQ2_XXS`). An expert's three tails are adjacent
and 4 KiB-aligned for direct I/O; on btrfs the packer creates the file
without copy-on-write, which also turns off compression, as direct reads
need. At load the engine checks the header against the model's expert
tensors (sizes, offsets, shard sizes) and compares sampled tails with the
GGUF; on any mismatch it logs why and reads only the GGUF. Results do not
change.

On the 990 PRO plus a KIOXIA EXCERIA BASIC (both PCIe 4.0 x4), one cold
miss takes 0.87 ms instead of 1.32-1.47 ms, and six misses 4.35 ms
(10.7 GB/s). Served workloads gain most where misses are frequent: with
the full ~100 GiB cache, short-request prefill gains 5-17% and decode
1-3%. Capped caches stand in for larger quantizations: at 88 GiB (the
expert share a ~150 GB backbone would keep cached) prefill gains 18-27% and
decode 2-5%, at 64 GiB prefill 47-65% and decode 12-23%
([E61](EXPERIMENTS.md#e61--expert-reads-split-across-two-drives-retained)).

## Prefetch from router predictions

During a layer's read overlap the decoder enqueues the **next** layer's
router on this layer's FFN input (one layer stale), copies the ids to pinned
memory and records an event. A store thread waits for the event and
submits a background read job for the predicted uncached experts. The
workers run it whenever no demand phase is active; a demand phase always
preempts it.

- Prefill batches of at least 1,024 tokens predict every layer; their jobs
  are "wide" and only a batch of that size adopts them.
- Single tokens, verification and joint batches (up to 16 rows) predict
  only when the current layer waited on reads: misses cluster in
  neighbouring layers, and a prediction without a read stall to hide its
  matvec costs more than the read it saves.
- The next prepare of the job's layer adopts only the experts it routes,
  including their not yet started tasks; a wrong prediction costs bandwidth,
  never phase time. A prepare for the layer after parks the job; any other
  drops it once its in-flight tasks drain.

A router-free sweep of the next layer's uncached experts was rejected: this
model's routing is skewed enough that it read 3.6× the demand volume.

## Engram reads

The Engram layers (1 and 14) read 24 rows of 144 B per token from the
103 GiB table file with buffered `pread` on up to 32 threads, started with
the evaluation so they overlap the layers before them. The rows are random,
so a pp2048 batch issues about 98,000 of them: under 10% of the batch's
bytes but about 99% of its read operations. Once the expert cache has grown
the page cache keeps none of them. The measured wait stays small (about
0.1 ms per pp2048 batch when warm).

## Storage background

Measured on the Samsung 990 PRO (XFS, `noatime`, scheduler `none`):

- Expert reads reach about 5.7 GiB/s effective, against 8.2 GB/s for purely
  sequential reads; per-layer bursts reached 5.7–5.9 GiB/s with four direct
  readers, and eight or twelve readers did not help.
- `io_uring` and polling `io_uring` matched `pread` threads (about 4.9 GB/s
  sustained at queue depth four).
- Moving whole shards or the Engram to a second NVMe did not help: the
  moved layers read at that drive's pace, and the Engram move cost more in
  latency than the contention it removed. That drive (KIOXIA EXCERIA BASIC
  on btrfs) was measured at 2.3 GB/s through btrfs compression; with
  compression off for the file it reads 5.15 GB/s direct, and splitting
  every expert read across both drives does help (above).

The runs are in [EXPERIMENTS.md](EXPERIMENTS.md#storage-reads).

## Statistics

The engine logs route hits, MiB read per evaluated token (and the share
prefetched), host wait per token and the allocated cache, per interval of at
least 64 tokens (with its token count) and cumulatively. Host wait is split
into the wait for the layer's routes, which includes GPU work queued before
them, and the wait on demand reads after the overlapped work; the rest is
host bookkeeping. `gufo diagnose` does not report them yet.

## Other implementations

Published rates are workload references, not matched comparisons:
quantization, hardware, context and speculative decoding differ.

- [ds4](https://github.com/antirez/ds4) and its
  [Strix Halo PR](https://github.com/antirez/ds4/pull/1036) (inspected at
  [`815e8fb`](https://github.com/kyuz0/ds4/tree/815e8fb6194ffa3160f1bd5f80df0e55efd190fe)):
  aligned direct reads into staging buffers with a buffered fallback, and
  whole-layer staging ahead during prefill; 10.18 tok/s after 2K prefill
  with a 76 GiB cache on its Q2 build, without DSpark.
- [Kyojin](https://github.com/Yamz-Labs/kyojin/blob/0c5b91a0686acc8129190f753e032715f0b0dcc2/doc/DS41.md):
  a 172 GiB EXL3 pack on 128 GB Strix Halo with direct I/O, 12 readers and
  4 MiB chunks, later-layer reads during prefill, a cache that protects
  frequent and recent-prompt experts, and row-blocked 32K prefill; roughly
  5–11 tok/s.
- [Spark recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe/blob/63220db0a7cc62dd8b783f71c9cf28f909c9b09b/README.md):
  a 1.62 bpw EXL3 quant on NVIDIA GB10 with cold experts read through an ATS
  mapping; 31–34 tok/s with DSpark, 13.4 without.
- [Project Maya](https://github.com/mw00/project-maya/tree/8b4b39f84e35393833b31b80e82bee92e69bb646):
  GPU, RAM and SSD tiers with threaded direct reads and CPU expert
  execution, measured on multi-GPU/CPU machines.
- [Xolotl](https://github.com/Indras-Mirror/Xolotl/tree/961d390fea6d848431822f7c6e94a6e5c0775133):
  VRAM, pinned RAM and NVMe tiers with ranking and layer read-ahead (62–70%
  one-layer router recall); the source of the built-in expert ranking. Its
  optional route bias toward resident experts is in the backlog.
- [mako](https://github.com/jhanssen/mako/tree/8567de3) (inspected at
  `8567de3`): C++/Metal on a 128 GB M5 Max with ~55–66% of experts resident;
  an mlock'ed arena with decayed-count eviction and a calibrated start set,
  CPU `pread` misses overlapped with the resident experts, two-layer-ahead
  gated read-ahead, and large prefill groups staged through a 3 GB ring
  without admission. Its `--miss-penalty` (cache-aware routing) gives ~25 →
  ~29 tok/s on Maya-S and 34 tok/s on ISTA `IQ2_XS`; backlog items 21–24.
