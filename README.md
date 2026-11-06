# Gemma 4 for Gufo

> [!NOTE]
> **This is the `gemma4-ext` branch of [akamotaco/gufo_win](https://github.com/akamotaco/gufo_win):**
> [franzmoca/gemma4-gufo-extension](https://github.com/franzmoca/gemma4-gufo-extension) `main`
> plus four changes for native Windows:
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

## Status: work in progress

The engine is moving here from the
[franzmoca/gufo](https://github.com/franzmoca/gufo) `feat/gemma` branch
([gufo-org/gufo#426](https://github.com/gufo-org/gufo/pull/426)). Until the
port is validated, use `feat/gemma` for real workloads.

Working today, on Gemma 4 31B `UD-Q4_K_XL` with the MTP drafter:

- `gufo serve llm`: chat, reasoning (`reasoning_content` and
  `reasoning_tokens`), native tool calls (buffered and streamed), image input,
  and prompt-cache reuse.
- `gufo prompt` and `gufo bench` through the extension's drivers.
- Provider options: `--image-tokens`, `--draft-calibration`, `--draft-trace` and
  `--prefill-weights`. MTP drafts up to 15 tokens with the calibrated policy
  unless `--draft-tokens` or `--draft-policy` say otherwise.
- Unit tests: configuration, tokenizer, chat template, tool syntax and the
  text contracts without models; kernels, target parity, continuation and
  vision with models.

Still missing:

- **Parity validation** against `feat/gemma`: greedy outputs, decode KL,
  functional suites (including cache and long context) and speed for all
  three checkpoints. Only 31B `UD-Q4_K_XL` has run so far, as a smoke test.
- **Cache features not upstream:** in-place same-slot restore and shared
  transfer streams (batches of 256 jobs instead of 16).
- **Known failures:** `reasoning-budget` (budgets are ignored; the runner
  declares no single-token budget tags) and `tool-images` (with thinking off,
  the model's empty `thought\n<channel|>` after a tool response reaches the
  content). Both also fail on `feat/gemma`.
- **Model cards and documentation** (benchmarks, quality, experiments) still
  live in `feat/gemma`; the Python tools in `tools/` are untested here.

## Build with Gufo

The extension needs extension API 0.2.9. Until the two pull requests for
provider options ([#979](https://github.com/gufo-org/gufo/issues/979)) and
speculation defaults merge upstream, build it on the `gemma-main` branch of
[franzmoca/gufo](https://github.com/franzmoca/gufo): upstream `main` plus those
two changes.

```sh
git clone -b gemma-main https://github.com/franzmoca/gufo
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
