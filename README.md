# Gufo: The ***Open Source*** Strix Halo AI engine 

<p align="center">
  <img src="assets/gufo-logo.jpg" alt="Gufo logo" width="180">
</p>

## gufo_win: native Windows build with Gemma 4

This repository runs Gufo **natively on Windows 11** (no WSL) on AMD Strix Halo
(`gfx1151`), and includes the **Gemma 4 engine** from the `feat/gemma` fork.
It combines three sources:

| Layer | Source | What it brings |
| --- | --- | --- |
| Engine | [gufo-org/gufo](https://github.com/gufo-org/gufo) `main` | Gufo itself: Qwen3.8-Flash-Next, Qwen3.8-27B, DeepSeek V4 Flash and more |
| Gemma 4 | [franzmoca/gufo](https://github.com/franzmoca/gufo) `feat/gemma` | Native Gemma 4 26B-A4B / 31B engine with MTP, vision and tool calls (described in the next section) |
| Windows port | [pixmaate/gufo](https://github.com/pixmaate/gufo) `main` | POSIX compatibility layer (`compat/win32`), Winsock and WDDM memory handling, build and launch scripts ([docs/WINDOWS.md](docs/WINDOWS.md)) |

On top of those, this branch ports the upstream code that arrived after the
Windows port was written (the new disk cache, socket and diagnostics code) and
adapts the build to TheRock 10.2. Every change, conflict decision and open item
is recorded in [docs/WINDOWS-PORTING-NOTES.md](docs/WINDOWS-PORTING-NOTES.md)
(Korean).

### Measured on Windows

Gemma 4 26B-A4B Q8_0 on one Strix Halo machine (Windows 11, 96 GB GPU carve-out),
same GGUF and prompts against a ROCm llama.cpp `llama-server` (b10967, flash
attention, no speculative decoding). Server-reported rates, temperature 0, prompt
cache defeated, one server at a time, October 18, 2026:

| Prompt | llama.cpp prefill | Gufo prefill | llama.cpp decode | Gufo decode | Gufo decode, MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2.1K tokens | 1,624–1,735 | 3,028 | 31.1 | 40.4 | 57.3 |
| 8.5K tokens | 602–608 | 3,062 | 29.4–29.8 | 39.9 | 54.0 |
| 31.8K tokens | 959–965 | 2,584 | 29.6 | 37.6 | 51.9 |

Model load took 8 s natively against 42 s for the same build reading the GGUF
from `/mnt/c` under WSL2.

The other models on the same machine and build, same prompts and method (Gufo only;
MTP uses each model's drafter):

| Model | Context | GPU in use | Prompt | Prefill | Decode | Decode, MTP |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next UD-Q4_K_XL, 1 session | 262,144 | 90.9 GB dedicated (91.4 peak) with MTP + vision | 2.1K | 1,491–1,496 | 28.1* | 41.5–44.0 |
| | | | 8.5K | 1,466–1,478 | 28.1* | 48.4–51.2 |
| | | | 31.8K | 1,497–1,515 | 28.0* | 62.1–65.4 |
| | | | 250.8K | 1,332–1,339 | | 54.5–66.5 |
| Qwen3.8-Flash-Next AtomicChat `AD-4.85bpw-Q5_K_M`, 1 session | 262,144 | 86.3 GiB with MTP + vision | 2.1K | 1,381–1,425 | 29.1 | 36.6–37.9 |
| | | | 8.5K | 1,396–1,422 | 29.3 | 47.4 |
| | | | 31.8K | 1,441–1,465 | 29.0 | 38.5–38.9 |
| | | | 159.9K | 1,338 | 28.1 | 42.1 |
| Gemma 4 31B Q4_K_M (MeroMero) | 200,000 | 32.5 GiB | 2.1K | 469–474 | 11.2 | 22.4 |
| | | | 8.5K | 449–462 | 10.9 | 21.7 |
| | | | 31.8K | 407–436 | 10.2 | 18.3 |
| Gemma 4 31B Q8_0 (MeroMero) | 262,144 | 49.7 GiB (51.2 peak) | 244.1K | 188 | | 6.0 |

- AtomicChat on the same file under a ROCm llama.cpp build (its MTP drafter, same fixed
  prompts): prefill 1,041 / 1,163 / 1,218 / 1,155 tok/s, decode with MTP 40.7 / 43.1 /
  50.4 / 45.5 / 34.2 (short / 2.1K / 8.5K / 31.8K / 159.9K) and 27.6–28.4 without it.
  Gufo's kernels decode faster; the gap at short contexts is the speculative cycle
  ([docs/WINDOWS-DECODE-OVERHEADS.md](docs/WINDOWS-DECODE-OVERHEADS.md)).
- Both Gemmas at 262K at once (26B Q8 on one port, 31B Q8 on another) peaked at
  85.2 GiB dedicated while the 26B took a 244K-token prompt (570 tok/s prefill) and the
  31B a 62K one, with no page-file writes.

- Flash-Next MTP uses Unsloth's `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`; Gufo
  rejects the `shared-Q4_K_M` drafter (Q4_K `hc_*` tensors) and the F16 vision
  projector (use `mmproj-BF16.gguf`). A short prompt with a 512-token reply decodes at
  41.7–44.0 tok/s with MTP; the long-prompt MTP rates are high partly because the
  benchmark prompt repeats itself and prompt lookup copies from it. *The no-MTP column
  was measured while another program held 12.6 GB of GPU memory and may be low.
- Stay inside the carve-out. Past it the display driver moves buffers into system
  memory, and Windows wrote to the page file: Flash-Next with **2 sessions x 262K** +
  MTP + vision sat at 95.3 GB dedicated idle and paged out ~20 MB when two long
  requests started. One session x 262K + MTP + vision peaked at 91.4 GB through a
  250K-token prompt with no page-file growth over 10 minutes idle and the benchmark.
  Gufo's Windows free-memory figure counts shared memory too, so it does not stop
  this by itself; check what other programs (e.g. ComfyUI) hold before loading.
- An `mmproj-BF16.gguf` beside the model (or in its parent folder) is loaded
  automatically. Keep it in a subfolder and pass `--mmproj` to choose vision.
- Windows charges GPU allocations to the commit limit as well, so keep a large page
  file; it is reserved, not written (391 MB in use throughout these runs).
- Gemma 4 12B (QAT Q4_0 and Q4_K_M fine-tunes) loads but fails its first request with
  `split attention head grouping unsupported`: the Gemma 4 engine does not cover the
  12B's head layout yet. This is not specific to Windows.

### Memory setup on a 128 GB machine

- **GPU carve-out (Variable Graphics Memory): 96 GB, no more.** Windows then sees about
  32 GB of RAM, and its shared-GPU-memory limit is half of that, ~16 GB. That limit is
  not a separate setting; a larger carve-out shrinks it.
- **Keep a large page file** (96 GB initial, 160 GB maximum here): Windows charges GPU
  allocations to the commit limit. It is reserved, not written.
- **Pass `--cache-ram-bytes`** (4 GiB for Flash-Next, 2 GiB per Gemma server worked well).
- **Flash-Next with another GPU program:** load Flash-Next (1 session x 262K, MTP,
  vision) first, then ComfyUI. Anima image generation ran beside it with ComfyUI's
  overflow in shared memory (13.2 of 15.8 GB), Windows RAM at ~80% in use, commit
  127.7 of 128.7 GB and no page-file writes; Flash-Next kept its speed (prefill
  1,471-1,506 tok/s, decode 40.6-47 tok/s). There is little headroom left in that state.

### Build on Windows

Prerequisites are in [docs/WINDOWS.md](docs/WINDOWS.md#prerequisites): TheRock ROCm
for gfx1151, Visual Studio Build Tools (C++), CMake, Ninja and vcpkg with
`icu curl openssl libpng libjpeg-turbo libwebp pkgconf` (`x64-windows`). Then:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 `
  -Rocm C:\TheRock\build -Vcpkg C:\vcpkg -Ninja C:\tools\ninja\ninja.exe
build\release\gufo.exe diagnose
```

This branch is built and tested with TheRock **10.2.0** (the port was validated
on 10.0.0; `build.ps1` warns about the difference). Serve a model with a fixed
prompt-cache budget, since the automatic one pins half of the free RAM:

```powershell
build\release\gufo.exe serve --host 127.0.0.1 --port 8093 llm `
  --model G4-26B-A4B-Q8_0.gguf --speculative mtp --mtp-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf `
  --context 65536 --cache-ram-bytes 4294967296
```

### Status

- Working: build, `gufo diagnose`, Gemma 4 26B-A4B and 31B with and without MTP,
  Qwen3.8-Flash-Next UD-Q4_K_XL with and without MTP and vision, and the
  [AtomicChat abliterated](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF)
  `AD-4.85bpw-Q5_K_M` mix (IQ4_XS/IQ4_NL/Q5_0 experts, Q4_1 n-gram table, BF16
  `ple_conv1d`). Its IQ4 experts expand to Q8_0's form while the routed F16 WMMA
  prefill GEMM fetches them ([docs/IQ4-WMMA-PLAN.md](docs/IQ4-WMMA-PLAN.md)), so it
  prefills faster than llama.cpp on the same file (see the table above). Its HC
  inject weights are Q8_0 and are expanded to F32 at load; before that fix the model
  garbled identifiers it copied.
- Not supported: Gemma 4 12B (see above).
- Not yet verified on this build: the disk prompt cache (`--cache-disk`), other
  Flash-Next quants, and the full test suite (`BUILD_TESTING` is off; some
  new upstream tests still use raw POSIX sockets).
- Not included yet from the Windows port: the Flash-Next decode tuning
  (`tuning.hpp` switches), the Qwen3.8-27B Windows kernels and Qwen3.6-35B-A3B.
  They conflict with newer upstream code and are listed in the porting notes.
- Updates: `main` merges `franzmoca/gufo feat/gemma` (which merges upstream
  periodically), and Windows-port commits are picked by hand.

Gufo is MIT licensed. Thanks to the Gufo maintainers, to franzmoca for the
Gemma 4 engine and to pixmaate for the Windows port.

---

## About this fork: Gemma 4 on Gufo

This fork of [gufo-org/gufo](https://github.com/gufo-org/gufo) adds a native
**Gemma 4** engine to Gufo, with its own HIP kernels for prefill, decode and MTP
verification on Strix Halo. Gemma 4 isn't on upstream's roadmap, so it is
maintained here on the default branch, `feat/gemma`. Upstream
[PR #426](https://github.com/gufo-org/gufo/pull/426) stays open for feedback.

> [!NOTE]
> **What's new:** the [Gemma changelog](CHANGELOG_GEMMA.md) lists the fork's
> changes, newest first, with a link to each pull request.

### What it runs

- **Gemma 4 26B-A4B** (mixture of experts): Unsloth UD-Q4_K_XL, UD-Q6_K_XL,
  UD-Q8_K_XL and the QAT checkpoint.
- **Gemma 4 31B**: Unsloth UD-Q4_K_XL and UD-Q8_K_XL, and the 31B QAT (Q4_0).
- **Fine-tunes and uncensored versions** of both models: most quant shapes run,
  including llama.cpp's standard K-quant mixes (Q4_K_M, Q5_K_M, Q6_K), with the
  base model's MTP drafter. A GGUF that ships an older revision of the Gemma 4
  chat template loads with a warning and uses the built-in template. Fine-tunes
  are not qualified to the cards' limits; the 26B card records one fine-tune's
  results.

### Features

- MTP speculative decoding with the `gemma4-assistant` drafters, with a draft
  length learned from how many drafts are accepted.
- Image input through the Gemma 4 vision tower (`--mmproj`), with selectable
  image token budgets (`--image-tokens`).
- Gemma's reasoning channel and native tool-call syntax, with tool-call
  arguments streamed as they are generated.
- Gufo's serving features work with Gemma 4 too: every text endpoint, RAM and
  disk prompt caching, and batched concurrent sessions.

Benchmarks against llama.cpp and quality reports are in each model card:
[26B-A4B](docs/models/gemma-4-26b-a4b/README.md),
[31B](docs/models/gemma-4-31b/README.md) and
[31B QAT](docs/models/gemma-4-31b-qat/README.md).

### Status

The fork merges upstream `main` periodically and is currently synced with
v0.17.0. Like upstream, it is Strix Halo only (`gfx1151`, Linux).
Everything below the line is upstream's README; its build steps apply
unchanged ([Build from source](#build-from-source)). If a Gemma 4 GGUF doesn't
load, open an issue on this fork with the file name. Bug reports, benchmark
numbers from other setups, feature requests and PRs are welcome here.
Thanks to the Gufo maintainers for the engine this builds on.

---

## Upstream Gufo README

Gufo is a vertical local inference engine specifically built and optimized for the AMD Strix Halo hardware:
Ryzen AI MAX+ 395 systems with Radeon 8060S (`gfx1151`), up to 128 GiB of unified memory.

**Contributions are welcome!**

See the [changelog](CHANGELOG.md) and [GitHub Releases](https://github.com/gufo-org/gufo/releases)
for user-facing changes and release history.

> [!TIP]
> There are two Gufo variants that haven't been merged yet: A **[Windows port](https://github.com/pixmaate/gufo)** and **[Gufo RDMA for Dual Strix Halo](https://github.com/neuhaus/gufo)**. Check them out!

## Models and benchmarks

All model documentation lives under [docs/models](docs/models/README.md):

| Model | Inference modes | Hugging Face weights | Benchmarks | Quality |
| --- | --- | --- | --- | --- |
| [Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) | Q4/Q8, images, AR, DFlash2 | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) / [Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf) · [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) | Q4: **656.33 tok/s pp**; up to **70.56 tok/s tg** single user and **123.00 aggregated tok/s** on 8 concurrent requests with DFlash2 · [Benchmarks](docs/models/qwen3.8-27b/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-27b/QUALITY.md) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md) | Q4, images, AR, MTP | Unsloth [Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) | **1,787.54 tok/s pp** (AR); up to **116.94 tok/s tg** single user and **245.11 aggregated tok/s** on 8 concurrent requests with MTP · [Benchmarks](docs/models/qwen3.8-flash-next/BENCHMARKS.md) | [Quality](docs/models/qwen3.8-flash-next/QUALITY.md) |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md) | AR, DSpark | [antirez Flash 0731 IQ2XXS](https://huggingface.co/antirez/deepseek-v4-gguf/blob/1cd7b564460821938add0475a60b942c409295e0/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) · [DSpark](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf) | **484.62 tok/s pp**; up to **26.62 tok/s tg** single user and **54.74 aggregated tok/s** on 8 concurrent requests with DSpark · [Benchmarks](docs/models/deepseek-v4-flash/BENCHMARKS.md) | [Quality](docs/models/deepseek-v4-flash/QUALITY.md) |
| [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md) | Speech recognition | [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | **15.27× realtime** · [Benchmarks](docs/models/qwen3-asr/BENCHMARKS.md) | [Quality](docs/models/qwen3-asr/QUALITY.md) |
| [Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md) | Speech synthesis and voice cloning | BF16 [CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) / [Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Up to **2.54× realtime**; **201 ms** to first audio (CustomVoice) · [Benchmarks](docs/models/qwen3-tts/BENCHMARKS.md) | [Quality](docs/models/qwen3-tts/QUALITY.md) |
| [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) | BF16 image generation and editing | [Complete pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/b3179ad355be050328e483a9dfdd9e60cd62adfa) | In progress · [Benchmarks](docs/models/qwen-image-2.1/BENCHMARKS.md) | [Quality](docs/models/qwen-image-2.1/QUALITY.md) |
| [MiniMax H3](docs/models/minimax-h3/README.md) | BF16 text to video/audio | [FL2VA pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/FL2VA) | In progress · [Benchmarks](docs/models/minimax-h3/BENCHMARKS.md) | [Quality](docs/models/minimax-h3/QUALITY.md) |

Peak measured workloads; text pp is autoregressive (AR), while tg uses the
named speculative mode. Peaks include repetitive output; aggregate tg sums
individual request decode rates. Qwen27B's single-user peak uses the short-prompt
C1 workload. Audio excludes loading.
Each model guide lists the required files and complete benchmark settings.

## Philosophy

- Contributions are welcome! We need the help of Strix Halo community to keep improving gufo!
- We would like this to be the one-stop shop for Strix Halo Local AI enthusiasts: batteries included for text, audio, image, and video models.
- Build and optimize specifically for the Strix Halo 128 GiB hardware. Smaller memory configurations should still work and preserve the speed benefits for models that can fit on memory.
- Support only the best available models for their size that can run on this hardware: less code to maintain, more focused optimization and testing work.
- Preserve quality when optimizing. Each model's quality report records independent numerical checks, execution consistency and unresolved gaps. Don't reuse kernels across different models to limit blast radius of a code change.
- Treat concurrent requests, cancellation and conversation caching as first-class workloads.
- Keep production dependencies small and development tools separate.

## Quickstart

```sh
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-GGUF
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  --repo-type model \
  --local-dir models/Qwen3.8-27B-DFlash2-GGUF
podman pull ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
podman run --rm \
  --userns=keep-id:uid=1000,gid=1000 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add keep-groups \
  --ulimit memlock=-1 \
  -p 8080:8080 \
  -v ./models:/models:ro \
  ghcr.io/gufo-org/toolboxes/gufo-runtime:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --speculative dflash2 \
  --dflash-model /models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Rootless Podman needs `crun` for `--group-add keep-groups`. Your host user must
have read/write access to `/dev/kfd` and `/dev/dri/renderD*`, usually through the
`render` and `video` groups; log out and back in after changing membership.
Container groups named `video`/`render` do not preserve host supplementary
groups. Check `id`, `ls -l /dev/kfd /dev/dri/renderD*`, and
`podman info --format '{{.Host.OCIRuntime.Name}}'` if ROCm reports no device.
See [Podman's rootless group-access guidance](https://github.com/containers/podman/blob/main/troubleshooting.md#20-passed-in-devices-or-files-cant-be-accessed-in-rootless-container).

On Fedora or another SELinux-enforcing host, GPU enumeration can succeed while
SELinux blocks mapping `/dev/kfd`, causing ROCr to report a misleading
“Memory critical” error. Check the **host** audit log:

```sh
sudo ausearch -m avc -ts recent | grep -E '/dev/kfd|hsa_device_t'
```

If it shows a denied `map` for the container, Podman documents this fix:

```sh
sudo setsebool -P container_use_devices true
```

This persistently allows containers to access device labels for devices passed
into them; it affects all containers on that host. Review that policy scope
before enabling it. See [Podman's device documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html#device-host-device-container-device-permissions)
and the [SELinux container policy](https://github.com/containers/container-selinux/blob/main/container.te).

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-UD-Q8_K_XL",
    "messages": [{"role": "user", "content": "Say something"}]
  }'
```

For Open WebUI, VS Code, OpenAI SDK and Responses-API clients (for example
Codex), set the API base URL to `http://localhost:8080/v1`. Chat Completions and
Responses support text, images, function tools, structured output and streaming.
See the [API contract](docs/SERVER.md).

The text server uses the model's native context by default and generates until
EOS or the context is full. `--context N` sets context capacity per session;
`--max-tokens N` sets a default response limit that clients can override.
Reasoning tokens count toward that response limit.

## Harness notes

- **Qwen with Codex:** Codex sends developer messages mid-conversation, after
  compaction or a settings change. Qwen's template accepts only one leading
  system turn, so Gufo moves them there. The request that introduces one is
  prefilled again; later requests reuse the cache. See the
  [API contract](docs/SERVER.md).

## Build from source

Linux x86-64 on AMD Strix Halo (`gfx1151`) is the supported target. CMake owns
one production configuration for both Nix and ordinary Linux builds. Tests,
profilers, tuning executables and Python reference runners are not installed
with the production package. No `build.sh` wrapper is needed.

Windows 11 x64 builds natively against AMD's TheRock ROCm; see
[Windows](docs/WINDOWS.md) for the toolchain, a setup check, the build script
and the launchers (`start.cmd`, `tools\windows\start.ps1`); releases carry a
prebuilt zip.

### With Nix

```sh
nix build
./result/bin/gufo diagnose
./result/bin/gufo serve llm --model /path/to/model.gguf
```

[flake.lock](flake.lock) pins the dependencies. `nix develop` adds profiling,
model-download and independent evaluation tools; these are not runtime
requirements. Optional benchmark baselines are selected separately with
`nix shell .#ds4-reference`, `.#llama-cpp-reference` or
`.#llama-cpp-mtp-reference`; see [benchmarking](docs/BENCHMARKS.md).
See [testing](docs/TESTING.md) for the small hosted CI suite and
explicit local quality checks.

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the
following development libraries. The currently qualified toolchain is GCC
15.3 and ROCm 7.2.3. Attention and audio convolution kernels are compiled
directly from HIP. Python, Triton/AOTriton, Composable Kernel and MIOpen are
not production build or runtime requirements.

| Dependency | Used for |
| --- | --- |
| ROCm HIP compiler/runtime, hipBLAS, hipBLASLt, rocBLAS | GPU execution and matrix multiplication |
| hipCUB, rocPRIM, rocWMMA headers | Compiled GPU kernels |
| ICU, libcurl, OpenSSL, libpng, libjpeg, libwebp | Tokenization, HTTPS, hashing and images |
| FFmpeg and ffprobe | Video/audio output; invoked as separate executables |

Install ROCm using [AMD's Linux instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
Use the development packages for the libraries above. ROCm normally installs
under `/opt/rocm`.

For example, on Debian/Ubuntu the ordinary system libraries are:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev ffmpeg

# ROCm libraries from the table, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/gufo diagnose
./build/release/gufo serve llm --model /path/to/model.gguf
```

Configuring fails at `find_package(hipblas)` when those ROCm packages are
missing. Other distributions name them `-devel` instead of `-dev`.
For nonstandard installations, pass ordinary CMake paths, for example
`cmake --preset release -DCMAKE_PREFIX_PATH="/opt/rocm"`.
If compiler discovery picks a system Clang, also pass
`-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++`.
Use `cmake --install build/release` to install Gufo,
its runtime data and license notices. The GPU driver must allow your user to
access `/dev/kfd` and `/dev/dri`; model weights are acquired separately.

The same source, compiler flags and install rules serve both builds. Nix pins
the complete toolchain for reproducible comparisons; changing the compiler or
math libraries requires the affected model's quality checks.

## License

Gufo's original code is [MIT licensed](LICENSE). Adapted code and dependencies
retain their own notices in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and `licenses/`, installed under `share/licenses/gufo`. Model weights are not
bundled and retain their publishers' terms.

## Reference Projects

The initial design is informed by the following open source projects:

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for compact model serving, GGUF, and CPU/GPU correctness paths.
- [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp), [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp), and [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo) for Strix Halo optimization inspiration.
- [vLLM](https://github.com/vllm-project/vllm) for continuous batching and paged request scheduling.
- [hipEngine](https://github.com/shisa-ai/hipEngine) for torch-free HIP execution, and native speculative-cycle work.
- [ds4](https://github.com/antirez/ds4) for DeepSeek V4 Flash, MoE scheduling, and DSpark.
- [audio.cpp](https://github.com/0xShug0/audio.cpp) for audio models for tts and asr tasks.
