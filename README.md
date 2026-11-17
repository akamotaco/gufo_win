# DeepSeek V4.1 Flash for Gufo

Run **DeepSeek-V4.1-Flash-1017**, a 763B-parameter mixture-of-experts model with a
native 1M-token context, on a single AMD Strix Halo with 128 GB of unified memory.

This is a [Gufo](https://github.com/gufo-org/gufo) source extension: it builds into
Gufo's OpenAI-compatible server and CLI, with its own HIP kernels for `gfx1151`.

- **Fits one machine.** `IQ1_M` and the expert-pruned Coder run fully resident.
  The larger builds keep their dense weights resident and stream routed experts
  from the SSD through a cache, optionally split across two drives.
- **Fast decoding.** DSpark speculative decoding and prompt lookup, with greedy
  output identical to plain decoding.
- **A complete server.** Gufo's OpenAI-compatible text API with reasoning, native
  tool calls (buffered and streamed), image input, concurrent sessions, and RAM and
  disk prompt caching.

| Greedy, one session, DSpark, vision | Prefill, 24K tokens | Decode, chat | Decode, long document | Decode, repetitive |
| --- | ---: | ---: | ---: | ---: |
| `IQ1_M`, resident | 977 tok/s | 28.2 tok/s | 37.2 tok/s | 84.7 tok/s |
| `IQ1_M`, streamed | 1,282 tok/s | 26.0 tok/s | 35.8 tok/s | 83.4 tok/s |
| `UD-IQ1_M`, streamed | 1,195 tok/s | 20.4 tok/s | 30.7 tok/s | 75.0 tok/s |
| `UD-IQ2_XXS`, streamed | 1,050 tok/s | 17.8 tok/s | 25.2 tok/s | 67.2 tok/s |

`MXFP8` Engram, November 2026. The repetitive answer copies a 1.8K-token passage,
the case prompt lookup speeds up, as when a pasted file comes back edited. How these
were measured, and the comparison with the engine this extension replaces, are in
[docs/VALIDATION.md](docs/VALIDATION.md).

> [!WARNING]
> **By default the model takes the whole machine.** Resident builds and streamed
> builds with the default expert cache leave about 2 GiB of host memory free, so a
> desktop, browser or other GPU workload can push the server or the system out of
> memory. To share the machine, stream and cap the expert cache; see
> [Leaving memory for other work](#leaving-memory-for-other-work).

Maintainer: Francesco Moca ([@franzmoca](https://github.com/franzmoca)).
[Report bugs here](https://github.com/franzmoca/ds41-gufo-extension/issues), with the
GGUF file names when a model does not load.

## Models

| Build | Size | bpw, all | bpw, experts | Avg of 6 | Aider | Runs | Tested |
| --- | ---: | ---: | ---: | ---: | ---: | --- | :---: |
| [ISTA GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) `IQ2_XXS`, 192 of 384 experts | 67.2 GiB | 2.06 | 1.94 | – | – | Resident, about 44 GiB free | Yes |
| [ISTA GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) `IQ1_M` | 112.4 GiB | 1.75 | 1.69 | 81.78 | 29.78 | Resident; streamed with vision | Yes |
| [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) `UD-IQ1_S` | 114.0 GiB | 1.77 | 1.70 | 83.53 | 36.52 | `--stream-experts` | No |
| [ISTA GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) `IQ2_XXS` | 128.7 GiB | 2.00 | 1.94 | 83.68 | 36.52 | `--stream-experts` | Yes |
| [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) `UD-IQ1_M` | 130.2 GiB | 2.03 | 1.95 | 85.55 | 42.13 | `--stream-experts` | Yes |
| [ISTA GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) `IQ2_XS` | 144.6 GiB | 2.25 | 2.19 | 85.64 | 41.57 | `--stream-experts` | No |
| [Unsloth](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) `UD-IQ2_XXS` | 154.0 GiB | 2.40 | 2.31 | 86.63 | 44.38 | `--stream-experts` | Yes |

Size and bits per weight exclude the Engram table; routed experts hold 97-99% of
the weights. The Coder keeps `IQ2_XXS`'s expert quantization for half the experts,
so its dense weights weigh more in its average. The scores are
[Unsloth's benchmarks](https://unsloth.ai/docs/models/deepseek-v4.1-flash#full-benchmarking-methodologies): "Avg of 6" averages AIME25, HMMT Feb26, GPQA-D, MMLU-Pro,
IFEval and LiveCodeBench v6 (temperature 1.0, 32K-token limit), and Aider is the
Aider polyglot score. The 8-bit `UD-Q8_K_XL` reference averages 87.68. The Coder is
not in them.

Besides the backbone, a server uses:

- **An Engram table**, read from the SSD and never resident; see [Engram tables](#engram-tables).
- **A DSpark drafter** for speculative decoding: Lucebox's
  [`DSpark-draft-Q2K-Q4K`](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF)
  (4.6 GiB).
- **The vision projector**, optional, for image input: ISTA's
  [`mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf`](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
  (0.9 GiB).

With vision on, stream `IQ1_M` even though it fits resident: resident, it leaves
too little GPU memory for prompt-cache checkpoints and prefills in smaller batches.
Streaming it costs 2-8% of decode speed and gains 15-31% in prefill.

## Quick start

Build Gufo with the extension (it needs Gufo's extension API 0.3.3 or later, upstream `main`):

```sh
git clone https://github.com/franzmoca/ds41-gufo-extension
cd ds41-gufo-extension
nix build    # ./result/bin/gufo
```

Without Nix, with [Gufo's prerequisites](https://github.com/gufo-org/gufo#build-from-source):

```sh
git clone https://github.com/gufo-org/gufo
cd gufo
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/ds41-gufo-extension
cmake --build --preset release --parallel 4
```

Download streamed `UD-IQ1_M` and the drafter, then serve it. Unsloth's builds include
their Engram table as the last shard, so the download is complete as it is:

```sh
mkdir ds41 && cd ds41
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "UD-IQ1_M/*" --local-dir .
hf download Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF \
  DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf --local-dir .

gufo serve llm \
  -m UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf \
  --stream-experts \
  --speculative dspark \
  --dspark-model DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf \
  --cache-disk ./cache --host 0.0.0.0 --port 8080
```

ISTA publishes one Engram file for all its builds; download it with the shards and
link it as the last one, for example for resident `IQ1_M`:

```sh
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ1_M-0000[1-5]-of-00006.gguf" DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf --local-dir .
ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00006-of-00006.gguf
```

Add `--mmproj mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` for image input.

| Option | Effect |
| --- | --- |
| `--stream-experts` | Keep dense weights resident and stream routed experts from the SSD |
| `--expert-cache-bytes N` | Cap the expert cache; see [Leaving memory for other work](#leaving-memory-for-other-work) |
| `--dspark-model PATH` | DSpark drafter for `--speculative dspark` |
| `--prefill-batch N` | Most prompt tokens per prefill batch, 16 to 4096 (default: 2048) |
| `--exact-prefill` | Run the decoder layers over every prompt token, at about half the prefill speed |

Useful Gufo options: `--context N` per session (default: the native 1M tokens),
`--sessions N` (about 0.9 GiB each at 1M), `--think off|on|auto`,
`--cache-ram-bytes N`, and `--cache-disk DIR`, which keeps prompt-cache checkpoints
across restarts.

**Bounded replay.** DeepSeek V4.1's layers 0-19 compute the shared KV that its
decoder layers 20-39 read. By default a prompt runs the decoder layers only over its
last 140 tokens: their 128-token attention window plus 12 tokens of backtracking,
the same default as DeepSeek's own inference engine. The full KV is still computed
for every token. `--exact-prefill` runs the decoder layers over the whole prompt;
the next tokens' predictions barely change (254-255 of 256 top-1 matches, see
[QUALITY.md](docs/models/deepseek-v4.1-flash/QUALITY.md#decoder-swa-bounded-replay-m5)).

## Engram tables

DeepSeek V4.1 adds Engram tables, an n-gram memory of 100-190 GiB that the model
looks up for every token. Each build's last shard is its table, and Gufo reads the
rows it needs from the SSD, so it takes no model memory.

| Table | Size | Comes with | Decode |
| --- | ---: | --- | --- |
| `IQ4_NL` | 103 GiB | ISTA's builds; Unsloth's `UD-IQ1_M` and `UD-IQ1_S` | Baseline |
| `Q5_K` | 126 GiB | Unsloth's `UD-IQ2_XXS` | Not measured |
| `MXFP8` | 189 GiB | Unsloth's `engram/` folder, the checkpoint's own format | Same per step; 4-12% slower with DSpark |

Both tables cost about 1.2 ms of a ~60 ms decode step, and reading either takes
the same time. With DSpark, `MXFP8` decoded 4-12% slower in our runs because its
replies differ, and on them the drafter was accepted less often or ran shorter
drafts ([E107](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e107--mxfp8-engram-cost-and-host-dequantization-retained)).

`MXFP8` is lossless. In the measurements it moves no task score and lowers
perplexity slightly only on text the model has likely memorised
([EVAL.md](docs/models/deepseek-v4.1-flash/EVAL.md)), but in daily use the model has
felt better with it. If you have the disk space, try it and keep the one you prefer:
download it from Unsloth's `engram/` folder and link it in place of the last shard. It
is labelled shard 6 of 6, which fits the six-shard builds as it is; five-shard builds
need the step below.

### One table for several builds

Builds can share one table file through links, which saves 100-190 GiB per build. A
table also records its shard position, and a build whose shard count differs refuses
it. Linking a 6-of-6 table into a five-shard build fails with:

```text
Failed to open GGUF: Inconsistent split GGUF shard metadata: .../DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00005-of-00005.gguf
```

Set the numbers the build expects with `tools/ds41/engram_split.py`, on a copy (free
on Btrfs and XFS, a full copy elsewhere) or in place:

```sh
python3 tools/ds41/engram_split.py DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file \
  DeepSeek-V4.1-Flash-1017-engram-MXFP8-5of5.gguf --shard 5 --of 5
python3 tools/ds41/engram_split.py DeepSeek-V4.1-Flash-1017-engram-MXFP8.gguf_file \
  --in-place --shard 5 --of 5
```

Only those two fields change. After an in-place change, every build that links the
file must have that shard count.

## Leaving memory for other work

A streamed build's expert cache fills the free memory, minus 2 GiB, which is the
fastest setting but leaves nothing for anything else. `--expert-cache-bytes N` caps
it, and the rest stays free for a desktop, other models or other jobs. A smaller
cache means more SSD reads: decode slows, and recovers as the cache learns which
experts the current work routes to.

Any build can stream, including the ones that fit resident. The Coder, streamed with
the drafter and one session at a 32K context (`gufo bench`, tg128):

| Expert cache | Decode |
| --- | ---: |
| All experts (61.5 GiB, the default) | 24.7 tok/s |
| 30 GiB | 14.5 tok/s from a cold cache, 21.8 tok/s once warm |
| 16 GiB | 8.1 tok/s cold, 12.8 tok/s warm |

The dense weights, the drafter and the session stay resident in every case, so
memory use is about 13 GiB plus the cap.

## Streaming and expert tails

With `--stream-experts`, routed experts are read from the SSD into a cache that
fills the free memory, so more free memory means fewer disk reads. Streaming is
never chosen automatically: a build that does not fit fails to load and says what
it needs. The automatic RAM prompt cache is 1 GiB with `--cache-disk` and 3 GiB
without, and the expert cache gets the rest.

**A second NVMe drive helps.** Pack each expert's tail into a file on the second
drive and link it beside the first shard as `MODEL.gguf.expert-tails`. Every expert
read then becomes a head read from the GGUF and a tail read from that file, issued
together, so both drives serve each miss. Output does not change.

```sh
python3 tools/ds41/pack_expert_tails.py \
  --model UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf \
  --out /second-drive/UD-IQ1_M.expert-tails
ln -s /second-drive/UD-IQ1_M.expert-tails \
  UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf.expert-tails
```

The tails file is 61.6 GiB for `IQ2_XXS`, 61.5 GiB for `UD-IQ1_M` and 73.0 GiB for
`UD-IQ2_XXS`. With two PCIe 4.0 x4 drives a cold miss takes 0.87 ms instead of
1.32-1.47 ms. The smaller the expert cache is against the model, the more it helps:

| Expert cache | Prefill | Decode |
| --- | ---: | ---: |
| About 100 GiB | +5-17% | +1-3% |
| 88 GiB | +18-27% | +2-5% |
| 64 GiB | +47-65% | +12-23% |

At load the engine checks the file against the model; on any mismatch it logs why
and reads only the GGUF. The
[streaming design](docs/models/deepseek-v4.1-flash/STREAMING.md) has the details.

## Adapters

`--adapter FILE` applies an adapter at run time without touching the model files, and
`--adapter-scale X` sets its strength (default 1). It loads two kinds of GGUF: direction
adapters (per-layer vectors removed from the layer outputs) and LoRAs on
`attn_output_b`, `ffn_down_shexp` and `ffn_down_exps`.

Adapters tested so far:

| Adapter | Kind | KL vs. no adapter | Setup |
| --- | --- | ---: | --- |
| [skafte/mako-ablations](https://huggingface.co/skafte/mako-ablations) `deepseek-v4.1-flash-1017/refusal.abl` | directions, 2 per layer | 0.015 | convert (below) |
| [BoldingBuilds/DeepSeek-V4.1-Flash-1017-Abliteration-LoRA-GGUF](https://huggingface.co/BoldingBuilds/DeepSeek-V4.1-Flash-1017-Abliteration-LoRA-GGUF) `...-rank1-L22.gguf` | rank-1 LoRA | 0.012 (0.011 converted) | use as is, or convert (below) |
| [SaturnHeaven/DeepSeek-V4.1-Flash-1017-heretic-lora-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-1017-heretic-lora-GGUF) | rank-2 LoRA | 0.018 | use as is |

KL is the mean per-token KL divergence from the model without an adapter, on resident
`IQ1_M` over a 3K-token text
([E97](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e97--direction-adapters-retained),
[E98](docs/models/deepseek-v4.1-flash/EXPERIMENTS.md#e98--lora-adapters-retained)).

```sh
# mako file -> direction adapter
python3 tools/ds41/direction_adapter.py from-abl refusal.abl mako.gguf --arch deepseek41
# rank-1 LoRA -> 1.6 MB direction adapter (optional; the LoRA also loads as is)
python3 tools/ds41/direction_adapter.py from-lora \
  DeepSeek-V4.1-Flash-1017-Abliteration-rank1-L22.gguf l22.gguf --model MODEL.gguf

gufo serve llm --model MODEL.gguf --adapter mako.gguf
```

Without `--adapter` the output is unchanged. With one, decode is 0.5-3% slower.

## Memory and host setup

Resident `IQ1_M` at the native 1M context, one session:

| Component | Memory |
| --- | ---: |
| Backbone | 112.3 GiB |
| Session | 2.3 GiB |
| DSpark drafter | 4.6 GiB |
| Vision projector | 0.9 GiB |
| **Host memory left**, server ready | **1.7 GiB**, 1.6 GiB with vision |

Streamed `UD-IQ2_XXS` with the drafter and vision gets a 95-96 GiB expert cache and
leaves about 2 GiB free.

- Set the UMA carve-out to its minimum in the firmware; the model lives in GTT.
- Let the GPU map nearly all memory with `amdgpu.gttsize=126976 ttm.pages_limit=32505856`.
  `/sys/class/drm/card*/device/mem_info_gtt_total` should then report about 133 GB.
- `CmaFree` in `/proc/meminfo` should be near 0. On Ubuntu 26.04, boot with `kho=off`.
- `ulimit -l unlimited` lets the prompt cache pin its memory.

## Known limitations

- Resident `IQ1_M` with DSpark or vision keeps no prompt-cache checkpoints: the
  device heap they leave is under the 4 GiB that Gufo's cache keeps free. A
  conversation that continues still reuses its session, but repeats, edits and
  branches of earlier prompts prefill again, and concurrent sessions cannot be
  prepared from a shared prefix (the multi-user DSpark benchmark does not run).
  Stream it instead.
- `UD-IQ1_S` and `IQ2_XS` have not been run.

## Documentation

- [Model docs](docs/models/deepseek-v4.1-flash/): benchmarks, quality, decode and
  prefill analysis, the streaming design, the backlog and every experiment. This
  extension continues the DeepSeek V4.1 engine of the
  [franzmoca/gufo](https://github.com/franzmoca/gufo) fork, where most of those
  numbers were measured.
- [Validation](docs/VALIDATION.md): this extension's test results.
- [Tools](tools/README.md): model-file tools, streaming simulators, references and
  kernel benchmarks.
- [CHANGELOG.md](CHANGELOG.md): changes by version.
- [UPSTREAM.md](UPSTREAM.md): source layout and origin.
- [AGENTS.md](AGENTS.md): build, test and measurement rules for contributors and
  coding agents.

## License

MIT ([LICENSE](LICENSE)). Code derived from DS4 keeps its license
([LICENSE.ds4](LICENSE.ds4)), code copied from Gufo keeps Gufo's
([LICENSES/gufo.txt](LICENSES/gufo.txt)), and the vendored llama.cpp kernels are
described in [VENDOR.md](kernels/rocm/mmq/VENDOR.md). Model weights are downloaded
separately under their own licenses.
