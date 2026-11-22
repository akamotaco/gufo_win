# Gemma 4 for Gufo

> [!NOTE]
> **This is the `gemma4-ext` branch of [akamotaco/gufo_win](https://github.com/akamotaco/gufo_win):**
> [franzmoca/gemma4-gufo-extension](https://github.com/franzmoca/gemma4-gufo-extension) `main`
> plus four changes for native Windows and one for extension API 0.3:
>
> - The runner declares a scheduling profile (512-token passes, one layer group,
>   decode width 1: what the host used before for runners without one) and the
>   manifest asks for `>= 0.3.0`. Gufo 0.3 (#1175) rejects runners without a profile.
>
> - `extension.json` lists `windows` among the platforms.
> - Cache gather copies use a pointer only where it is also the device address
>   (`continuation_adapter.cpp`, `Accessible`). Windows maps registered host memory
>   at a different device address; a gather given the host pointer fails the first
>   cache capture with "unspecified launch failure".
> - A failed cache copy clears the HIP error and logs `[gemma4] event=cache_copy_failed`
>   with the pointers' attributes, so it costs a cache miss instead of failing the
>   next decode with a stale "invalid argument".
> - The vision encoder pads its MLP down weights on the host and uploads them with
>   one 1D copy (`vision/encoder.hip`, `UploadHalfPadded`). A 2D copy from the mapped
>   GGUF fails inside the Windows driver ("Failed PAL memory allocation") while
>   reporting success, so those 27 weights stayed unwritten and image answers had
>   nothing to do with the image.
>
> Build it with gufo_win `main` and `-DGUFO_EXTENSION_DIRS=<this folder>`; see the
> gufo_win README. Everything below is the extension's own README.

A [Gufo](https://github.com/gufo-org/gufo) source extension for Google's
Gemma 4 on Linux x86-64 / Strix Halo `gfx1151`:

- Gemma 4 31B (Unsloth `UD-Q4_K_XL`, `UD-Q8_K_XL`)
- Gemma 4 31B QAT (Q4_0)
- Gemma 4 26B-A4B mixture of experts (`UD-Q4_K_XL`, `UD-Q6_K_XL`, `UD-Q8_K_XL`)

with `gemma4-assistant` MTP drafting, prompt lookup, the BF16 vision sidecar,
batched sessions, native tool calls and the shared serving cache.

Maintainer: Francesco Moca ([@franzmoca](https://github.com/franzmoca)).

Report bugs and regressions in the
[issues of this repository](https://github.com/franzmoca/gemma4-gufo-extension/issues),
with the GGUF file name when a model does not load.

## Features

- `gufo serve llm`: every text endpoint, reasoning (`reasoning_content`,
  `reasoning_tokens` and reasoning budgets), native tool calls (buffered and
  streamed), image input, batched sessions, and RAM and disk prompt caching.
- `gufo prompt` and `gufo bench` through the extension's drivers.
- Provider options: `--image-tokens`, `--draft-calibration`, `--draft-trace` and
  `--prefill-weights`. MTP drafts up to 15 tokens with the calibrated policy;
  Gufo ignores `--draft-tokens`, `--min-draft-tokens` and `--draft-policy`
  since [#1124](https://github.com/gufo-org/gufo/pull/1124).

Benchmarks and quality reports are in the model cards:
[26B-A4B](docs/models/gemma-4-26b-a4b/README.md),
[31B](docs/models/gemma-4-31b/README.md) and
[31B QAT](docs/models/gemma-4-31b-qat/README.md).

## Validation

Compared with the `feat/gemma` engine on November 6, 2026, over HTTP with the
cards' benchmark tool, in balanced order, on Strix Halo at the
accelerator-performance profile. Single-user greedy MTP, UD-Q4_K_XL, pp2048 and
tg128, mixed and repetitive text, depths 0 and 8,192:

| Model | Prefill | Decode |
| --- | ---: | ---: |
| 31B | -0.3% to -0.5% | -0.1% to +0.3% |
| 26B-A4B | -2.1% to +2.1%, noisy | -1.2% to +2.2% |

Sampled MTP at depth 0 against the cards: 26B-A4B prefill 4,010 (card 3,898),
decode 71.8 (card 73.8 ± 2.7); 31B prefill 565 (card 561), decode 25.5 (card
22.3). The target tests pass on 31B and 26B-A4B, the Pi coding-agent replay
passes 36 of 36 on 26B-A4B, and the tool, image, cache and long-context
functional suites pass on both, apart from `image-count` below.

Not repeated on this build: QAT, and the complete functional set.

On Gufo `f01a4570` (extension API 0.3.0, November 11, 2026), with 26B-A4B
Unsloth UD-Q4_K_M, the Q8_0 MTP drafter and the BF16 projector, ROCm 10.1:
the CPU tests, the GPU operator tests and `gemma4.target`, `continuation`,
`continuation_perturbed`, `vision_encoder` and `vision_session` pass, as do
the `sampling-defaults` and `tool-native-schemas` functional suites (MTP).
The 31B models, QAT and the remaining suites were not run on this revision.

On Gufo v0.35.0 (extension API 0.3.20, November 17, 2026), with 26B-A4B
Unsloth UD-Q4_K_XL, the MTP drafter and the BF16 projector, ROCm 10.1: all 19
`gemma4` ctest tests pass, CPU and model-backed. Of the 63 functional suites
in the `pr` tier (two sessions, 32,768 context), 61 pass; `image-detail`
(below) and `reasoning-budget` fail. The full-tier `prefill-scheduling`,
cache and metrics suites pass at a 65,536 context, apart from
`cache-pressure` (below).

## Known limits

- The functional suites `image-detail` (it assumes Qwen's image token counts)
  and `image-count` on 26B-A4B (thinking passes that test's 256-token limit)
  fail for those test reasons.
- `reasoning-budget` assumes thinking is on by default and that the reasoning
  and answer fit in 600 tokens; Gemma defaults to thinking off and, with
  `--think on`, needs more. Budgets themselves are honored: 0, 1, 4, 16 and
  64 give exactly that many reasoning tokens, each a prefix of the uncapped
  reasoning.
- With a reasoning budget, the answer can contain a raw `<channel|>`. When
  the budget cuts the thought mid-sentence, Gemma keeps thinking in the
  answer, then closes the thought itself, and that marker reaches the user
  (6 of 48 budgeted replies on 26B-A4B, with and without MTP). The fix
  belongs in Gufo's sampler: after the forced close, never emit `<channel|>`
  or `<|channel>` again.
- With a reasoning budget, `reasoning_tokens` can report one more than the
  budget, though the model thought for exactly the budget. When Gemma writes
  its `thought` header with a double-newline token, the host counts that
  token as reasoning (3 of 6 prompts on 26B-A4B).
- `cache-pressure` fails on `pressure_ui_title`: the side request embeds a
  chat that ends in "Reply with only ALPHA", and Gemma follows it instead of
  the request's own instruction.
- In-place same-slot restore and batches of 256 cache transfers on one stream
  lease (instead of 16 with a lease each) are not upstream. Gufo's shared
  transfer-stream leases (v0.35.0) share native streams between leases; they
  do not change the batch size.
- The Python tools in `tools/` are untested here. `draft_policy_bench.py` and
  `proposal_replay.py` select policies and draft counts with Gufo's draft
  flags, which Gufo now ignores.

## Build with Gufo

The extension needs extension API 0.3.20 or later (Gufo v0.35.0, which
requires every text runner to declare a scheduling profile,
[#1175](https://github.com/gufo-org/gufo/pull/1175)); `flake.lock` pins the
Gufo revision it was last built and tested with.

```sh
git clone https://github.com/gufo-org/gufo
cd gufo
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/gemma4-gufo-extension
cmake --build --preset release --parallel 4
./build/release/gufo diagnose --section extensions
```

With Nix, `nix build` in this repository builds the Gufo revision pinned in
`flake.nix` with this extension. The manifest maps the `mtp` mode to
`--mtp-model`, so a server generated with Gufo's `mkGufoServe` and
`speculative = "mtp"` starts the drafter.

```sh
./build/release/gufo serve llm \
  -m gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --speculative mtp --mtp-model mtp-gemma-4-31B-it-Q8_0.gguf
```

The vision sidecar `mmproj-BF16.gguf` is found beside the model, or given with
`--mmproj`.

## Releases

Each release is tagged `vX.Y.Z` with its notes from
[CHANGELOG.md](CHANGELOG.md) on the
[releases page](https://github.com/franzmoca/gemma4-gufo-extension/releases).
Its `flake.lock` pins the Gufo revision it was validated with, so a flake can
pin a release, and Renovate can raise the tag:

```nix
inputs.gemma4.url = "github:franzmoca/gemma4-gufo-extension/v0.2.1";
# packages: gemma4.packages.x86_64-linux.default
```

Leave out `inputs.gemma4.inputs.gufo.follows`: overriding Gufo builds a pair
the release was not checked with.

A release is one `chore(gemma4): release X.Y.Z` commit on `main` that sets
`version` in `extension.json` and moves `Unreleased` in the changelog to
`## X.Y.Z (YYYY-MM-DD)`, naming the Gufo pin and the extension API it needs.
After it merges, `git tag -a vX.Y.Z -m X.Y.Z <commit> && git push origin
vX.Y.Z` publishes it: `.github/workflows/release.yml` checks the tag against
`extension.json` and posts the changelog section. Raise the minor version
when the Gufo pin or the extension API requirement changes, and the patch
version for fixes on the same pin.

## Tests

Test and probe executables link Gufo's public implementation targets
(`gufo_ext_core`, `gufo_ext_cache`, `gufo_ext_hip`). Model tests read
`GUFO_GEMMA4_MODEL`, `GUFO_GEMMA4_MTP_MODEL` and `GUFO_GEMMA4_MMPROJ` and are
skipped without them.

## License

The extension source is MIT licensed ([LICENSE](LICENSE)). Code copied from
Gufo (`hip/ggml/`, `hip/half_gemm.*`, `mtp_sampling.hpp`, `prompt_lookup.hpp`,
`vision/resize.*` and `DraftVocab` in `draft_vocab.hpp`; each names its
source) keeps Gufo's MIT license ([LICENSES/gufo.txt](LICENSES/gufo.txt)). Gemma 4 weights
are downloaded separately under their own license.
