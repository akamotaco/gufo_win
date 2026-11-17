# Changelog

Notable user-facing changes are recorded here. Versions follow `version` in
[extension.json](extension.json).

## Unreleased

### Bug Fixes

* **prompt cache:** a prompt-cache hit into a session that held another prompt
  keeps its restored state. Before, up to 6 in 10 such hits gave a greedy reply
  that parted from the uncached reply at its second token
  ([E121](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e121--wait-for-the-decoder-reset-before-a-prompt-cache-restore-retained)).

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
