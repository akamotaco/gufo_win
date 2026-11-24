# Changelog

Notable user-facing changes are recorded here. Versions follow `version` in
[extension.json](extension.json).

## Unreleased

### Performance

* **sampling:** tool calls and other output under a JSON grammar sample on
  the GPU, like unconstrained text, instead of building the distribution
  on the host; outputs are unchanged. On a coding-agent replay decode is
  1.2% faster, 1.9% in the replies that call tools
  ([E131](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e131--masked-device-draws-under-a-grammar-retained)).
* **prefill:** a streamed build runs prompts with at least 8,192 uncached
  tokens in 4,096-token passes: the expert cache gives up the memory for the
  extra rows while the prompt runs and gets it back afterwards. Fresh
  17.7K- and 21K-token prompts prefill 5.1% and 4.8% faster; shorter prompts,
  decode and the expert cache size are unchanged. `--prefill-batch` is
  removed: commands that pass it fail with an unknown option
  ([E134](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e134--wide-prefill-passes-on-demand-retained)).

## 0.5.0 (2026-11-24)

Built on upstream Gufo `60c48924`; needs extension API 0.3.43 or later (Gufo's
experimental expert store, gufo-org/gufo#1318).

### Features

* **expert streaming:** Gufo's serving sizes the streamed expert cache
  together with the RAM prompt cache, and the automatic prompt-cache share
  is now per session
  ([E126](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e126--serving-sizes-the-expert-cache-with-the-prompt-cache-retained)).
* **expert streaming:** the automatic RAM prompt cache follows the context:
  room for one checkpoint of a whole context per session (0.29 GiB at 256K),
  or one in all with `--cache-disk` and at least 1 GiB, instead of a fixed
  3 GiB or 1 GiB per session, so the expert cache gets the rest
  ([E128](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e128--the-automatic-prompt-cache-follows-the-context-retained)).
* **expert streaming:** under host pressure the streamed expert cache gives
  memory back first, keeping the experts in use, and regrows once it is free
  again; pressure is measured against the store's 1 GiB floor instead of
  Gufo's 4 GiB host reserve, which the cache leaves unused on purpose.
  Without this, Gufo's idle relief released the RAM prompt cache at startup.
  The expert cache gets back the 3.6-3.9 GiB that 0.4.0 left free for that
  reason ([E127](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e127--the-expert-store-as-proposed-to-gufo-retained)).
* **expert streaming:** the streamed expert cache lives in device pages the
  host cannot swap and is filled once, when its size is known: it starts at
  the same size every time, and gives memory back under host pressure in
  about a second instead of up to a minute
  ([E127](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e127--the-expert-store-as-proposed-to-gufo-retained)).

### Fixes

* **security:** the GGUF loader rejects a tensor or metadata count that
  cannot fit in the file before it sizes a table. Before, a crafted model file
  could overflow the tensor table's size and write past it on the heap, which
  can lead to code execution. Reported by eversinc33.

## 0.4.0 (2026-11-23)

Built on upstream Gufo `e7e38411`; needs extension API 0.3.3 or later.

This release fixes #13 but gives a streamed build's expert cache about
3.6-3.9 GiB less memory, until the expert store proposed to Gufo
(gufo-org/gufo#1318) can give it back. On a coding-agent replay decode was
unchanged and prefill 2-4% slower, but workloads that miss the expert cache
more may lose some speed.

### Fixes

* **prompt cache:** with `--cache-disk`, a long session no longer fails
  every turn with `std::bad_alloc` once the disk checkpoints' bookkeeping
  fills the RAM prompt cache's budget (#13). Gufo now refuses that
  bookkeeping instead of failing the reply (gufo-org/gufo#1326) and keeps
  each disk manifest once (gufo-org/gufo#1327), so `--cache-ram-bytes off`
  is no longer needed. Past that point turns recompute more until the
  bookkeeping itself shrinks (gufo-org/gufo#1329).
* **expert streaming:** the expert cache stops growing at 4.5 GiB of free
  memory instead of 1 GiB (5.5 GiB instead of 2 GiB at startup). Gufo now
  gives the RAM prompt cache's memory back whenever less than 4 GiB is
  available, and below that the prompt cache was lost at the first idle
  moment, so every turn recomputed its whole prompt
  ([E129](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e129--keep-the-expert-cache-above-gufos-host-reserve-retained)).

## 0.3.1 (2026-11-21)

Built on upstream Gufo `a27bbf98`, the same pin as 0.3.0; needs extension API
0.3.3 or later.

### Fixes

* **dspark:** a DSpark drafter whose tensors include types Gufo's GGUF
  reader cannot size, such as the MXFP4 routed experts of unsloth's
  `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, loads with `--cache-disk`.
  It failed with "Invalid GGUF tensor shape or unsupported storage type"
  when the disk cache fingerprinted it (#11).
* **tokenization:** control-token spellings in request text stay text.
  Message content, reasoning, tool schemas, call names and arguments, and
  tool results used to turn `</think>`, `<｜User｜>`, `<｜end▁of▁sentence｜>`
  and the other control spellings into control tokens, so a leaked tag or a
  file a tool returned could close a turn or a reasoning block. Prompts
  without such spellings tokenize as before, as does `/completion` text. The
  renderer records the request text's spans, as Gufo's `deepseek_v4_flash`
  does.

### Performance

* **tokenization:** each request tokenizes its prompt once, not twice (three
  times with images). The renderer reports the prefix a following user turn
  keeps, as Gufo's Qwen renderer does, and only the final pretokenizer piece
  of a user block that turn would join is tokenized again. Preparing a
  74K-token conversation takes 14.2 ms instead of 29.8 ms (870K tokens: 164
  instead of 340 ms), with the same tokens and cache prefix.

### Build

* **releases:** each release is tagged `vX.Y.Z` and published as a GitHub
  release with its changelog section, so a flake can pin a validated release.

## 0.3.0 (2026-11-17)

Built on upstream Gufo `a27bbf98` (0.34.0 and later commits, included in 0.35.0);
needs extension API 0.3.3 or later.

### Dependencies

* **Gufo:** pinned to `a27bbf98`. The extension now needs Gufo's extension
  API 0.3.3 or later, whose text runners must declare a scheduling profile.

### Bug Fixes

* **prompt cache:** a prompt-cache hit into a session that held another prompt
  keeps its restored state. Before, up to 6 in 10 such hits gave a greedy reply
  that parted from the uncached reply at its second token
  ([E121](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e121--wait-for-the-decoder-reset-before-a-prompt-cache-restore-retained)).
* **dspark:** the prompt-cache checkpoint 3 tokens before the end of each prompt
  keeps the drafter's rings. Before, it saved them before the last batch's
  writes finished, so a later hit would draft from stale rows. Replies did not
  change
  ([E122](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e122--wait-for-queued-writes-before-a-prompt-cache-capture-retained)).

### Performance

* **scheduling:** a prompt that arrives while another session decodes is
  prefilled one prefill batch per pass (2,048 tokens by default, following
  `--prefill-batch`) instead of 512 tokens. On streamed `UD-IQ2_XXS` a 9K-token
  prompt beside a decoding session prefills 36% faster (736 → 1,001 tok/s),
  and the decoding session stalls 25% less in total, though its longest stall
  grows from 1.9 to 2.3 s
  ([E123](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e123--one-prefill-batch-per-pass-beside-decoders-retained)).

## 0.2.0 (2026-11-11)

Built on upstream Gufo `0e2de43e` (0.33.0); needs extension API 0.2.17 or
later.

### Known limitations

* Resident `IQ1_M` with DSpark keeps no prompt-cache checkpoints, as with
  vision: the device heap they leave is under the 4 GiB that Gufo's cache keeps
  free. Multi-user DSpark cannot prepare its sessions from a shared prefix, so
  that benchmark does not run on resident `IQ1_M`; stream it instead
  ([benchmarks](docs/models/deepseek-v4.1-flash/BENCHMARKS.md#multiple-users-dspark)).

### Dependencies

* **Gufo:** pinned to `0e2de43e` (0.33.0), with disk prompt-cache fixes,
  shared CPU sampling improvements and tool-call correctness fixes.
  Draft-tuning CLI flags now warn and are ignored; DSpark uses the provider's
  defaults.

### Features

* **adapters:** `--adapter FILE` loads a direction adapter, which removes or adds
  per-layer directions in what attention and the experts write into the
  residual streams, or a GGUF LoRA on `attn_output_b`, `ffn_down_shexp` and
  `ffn_down_exps`; `--adapter-scale` sets its strength. Without an adapter the
  output is unchanged. `tools/ds41/direction_adapter.py` converts mako's `.abl`
  files and rank-1 abliteration LoRAs
  ([E97](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e97--direction-adapters-retained),
  [E98](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e98--lora-adapters-retained)).

### Bug Fixes

* **streaming:** the expert cache fills to its limit at startup; a server
  started after a run with long prompts could leave 0.4-0.6 GiB of it unused
  for its whole run ([E104](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e104--a-96-row-prefetch-stage-that-keeps-the-most-routed-experts-retained)).

* **streaming:** `--prefill-batch` above 2048 on a streamed model no longer
  leaves the server without a prompt cache
  ([E113](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e113--load-headroom-that-grows-with-the-prefill-batch-retained)).

* **prefill:** the batched expert and dense projections give the same
  result for a token whatever the batch size and wherever the batch starts;
  `--prefill-batch 4096` now gives the same logits as 2048. Replies change
  slightly against earlier builds
  ([E114](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e114--fp16-prefill-kernels-sum-order-by-weight-row-retained)).

### Performance

* **streaming:** the demand stage for missed experts takes 0.6 GiB instead of
  3.7 GiB, which leaves streamed builds ~3.8 GiB more expert cache: decode
  1.1-1.6% and prefill 2.4-2.7% faster on `UD-IQ2_XXS`, outputs unchanged
  ([E119](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e119--a-64-row-demand-stage-misses-past-it-read-into-cache-slots-retained)).
* **streaming:** the prompt cache keeps 16 transfer streams plus one per
  session instead of Gufo's 64, which leaves streamed builds ~0.3 GiB more
  expert cache. The extension now needs Gufo's extension API 0.2.17
  ([E104](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e104--a-96-row-prefetch-stage-that-keeps-the-most-routed-experts-retained)).

* **streaming:** the prefetch stage holds 96 instead of 192 experts and keeps
  the experts most tokens route to when a prompt predicts more, leaving
  ~0.9 GiB more to the expert cache, with identical outputs
  ([E104](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e104--a-96-row-prefetch-stage-that-keeps-the-most-routed-experts-retained)).

* **adapters:** LoRA decode adds each matrix's low-rank term inside the
  residual update instead of a launch of its own; with the 120-matrix L22
  abliteration LoRA, decode is 0.4% faster with identical outputs
  ([E106](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e106--lora-up-terms-in-the-hc-expand-retained)).

* **prefill:** `attn_q_b` and the Engram projection run in chunks that stay
  in the GPU's 32 MiB cache, and the BF16 router, compressor and indexer
  kernels build their matrix-core operands with fewer LDS reads; fresh
  20-24K-token prompts are 1.6-2.1% faster with identical outputs
  ([E108](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e108--dense-fp16-prefill-workgroups-in-cache-sized-chunks-retained),
  [E109](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e109--bf16-matmul-operands-without-rocwmma-weights-not-staged-in-lds-retained)).

* **engram:** MXFP8 Engram rows decode about 9× faster on the host, with
  identical values; MXFP8 reads cost the same as IQ4_NL
  ([E107](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e107--mxfp8-engram-cost-and-host-dequantization-retained)).

* **streaming:** the expert cache evicts by frequency class: an expert used
  twice as often may stay unused twice as long. A replayed agent session
  reads 6% less and runs ~1% faster, and a code prompt after a chat answer
  is processed 10% faster, at 1-2% on a cold first prompt; outputs are
  identical
  ([E105](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e105--eviction-by-frequency-class-retained)).

* **dspark:** greedy decoding resumes drafting as soon as a probe shows that it
  pays, and every request starts with a fresh draft policy. Greedy chat decodes
  4-7% faster and code up to 7.5% faster on streamed builds, with identical
  output; sampled decoding is unchanged
  ([E99](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e99--greedy-dspark-reset-the-policy-per-request-probes-set-the-saving-retained)).
* **dspark:** the drafter's attention reads its keys once per head instead of
  once per head and row, 4× faster; drafting cycles take ~1% less time
  ([E102](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e102--dspark-attention-one-block-per-head-retained)).
* **dspark:** the drafter's Q4_K projections over several rows use the split
  GEMV, with identical results; drafting cycles take a further 0.2-0.5% less
  time
  ([E103](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e103--drafter-q4_k-projections-through-the-split-gemv-retained)).

## 0.1.0 (2026-11-08)

First release as a Gufo source extension, built on upstream Gufo `main`
(extension API 0.2.14).

### Features

* DeepSeek V4.1 Flash on one Strix Halo: `IQ1_M` and the expert-pruned Coder
  resident, larger builds with `--stream-experts`, optionally reading expert
  tails from a second drive.
* DSpark speculative decoding and prompt lookup, greedy and sampled.
* Gufo's OpenAI-compatible text API with reasoning, native tool calls, image
  input, concurrent sessions, and RAM and disk prompt caching.
* `IQ4_NL`, `Q5_K`, `Q8_0` and `MXFP8` Engram tables.
* `tools/ds41/engram_split.py` sets the shard position of an Engram table, on a
  copy or in place, so one table can serve builds with different shard counts.

### Bug Fixes

* Resident `IQ1_M` with DSpark and vision loads with a RAM prompt cache inside
  the host memory reserve.
