# Changelog

Notable user-facing changes are recorded here. Versions follow `version` in
[extension.json](extension.json).

## Unreleased

### Features

* **windows:** the manifest lists Windows among the supported platforms, so a
  Gufo `windows-release` build discovers the extension and serves Gemma 4.
  Tested on Windows 11 with Strix Halo `gfx1151` and TheRock 10.0.0, on a Gufo
  fork; there, Gemma 4 captures longer than about 24K tokens need 1 GiB cache
  segments, a change in Gufo core
  ([#9](https://github.com/franzmoca/gemma4-gufo-extension/pull/9), by
  [@akamotaco](https://github.com/akamotaco)).

### Bug Fixes

* **prompt cache:** cache copies use the gather kernel only where the device
  address of a pointer equals the pointer, and copy the other runs with
  `hipMemcpyAsync`. On Windows, HIP maps registered host memory at a different
  device address, and the gather read and wrote the wrong memory. Linux takes
  the same path as before
  ([#9](https://github.com/franzmoca/gemma4-gufo-extension/pull/9), by
  [@akamotaco](https://github.com/akamotaco)).
* **prompt cache:** a failed cache copy skips its checkpoint without failing
  the request. Before, its HIP error stayed behind, and the next kernel launch
  check failed the request with it. The failure is logged as
  `event=cache_copy_failed` with the error, the size and both pointers
  ([#9](https://github.com/franzmoca/gemma4-gufo-extension/pull/9), by
  [@akamotaco](https://github.com/akamotaco)).
* **vision:** the padded vision MLP weights are padded on the host and uploaded
  with one copy. On Windows, the 2D copy from the memory-mapped GGUF failed
  while reporting success, so the weights stayed zero and answers ignored the
  image ([#9](https://github.com/franzmoca/gemma4-gufo-extension/pull/9), by
  [@akamotaco](https://github.com/akamotaco)).

## 0.2.1 (2026-11-20)

Built on Gufo v0.35.0 (`0260e116`); needs extension API 0.3.20 or later.

### Bug Fixes

* **moe routing:** a token whose router logits are NaN routes to experts in
  range. Before, it selected an expert id past the expert table (128 at 128
  experts, `0x7FFFFFFF` at 256), and the routed products, the expert counts
  and the expert scales were indexed out of bounds
  ([#4](https://github.com/franzmoca/gemma4-gufo-extension/pull/4), by
  [@mduka](https://github.com/mduka)).
* **moe routing:** a +INF router logit gives its row finite weights, split
  over its +INF experts. Before, every weight of the row was NaN
  ([#5](https://github.com/franzmoca/gemma4-gufo-extension/pull/5)).

### Build

* **releases:** each release is tagged `vX.Y.Z` and published as a GitHub
  release with its changelog section, so a flake can pin a validated release.

## 0.2.0 (2026-11-17)

Built on Gufo v0.35.0 (`0260e116`); needs extension API 0.3.20 or later.

### Dependencies

* **Gufo:** pinned to v0.35.0. The extension now needs extension API 0.3.20 or
  later, which requires every text runner to declare a scheduling profile
  ([gufo-org/gufo#1175](https://github.com/gufo-org/gufo/pull/1175)). The
  runner declares 512 prompt tokens per pass beside decoders, and decode steps
  batch up to 8 sessions.

### Bug Fixes

* **prefill:** each prefill step ends at the engine's chunk, so the host
  reports progress during a long prompt and schedules other requests between
  its chunks. Before, a prompt with no peers ran as one step.
* **tool calls:** tool calls and their history keep the model's number
  spelling ([@mduka](https://github.com/mduka)).

### Performance

* **prompt cache:** cache transfer leases are sized from the serving slots:
  two sessions share 3 native streams instead of owning 64.

## 0.1.0 (2026-11-17)

Built on Gufo `49380168`; needs extension API 0.2.14 or later.

First release as a Gufo source extension. Gemma 4 moves out of the
`feat/gemma` Gufo fork into this repository, which builds against upstream
Gufo instead of a patched copy. Over HTTP on Strix Halo, single-user greedy
MTP stays within 0.5% of `feat/gemma` on 31B and within about 2% (noisy) on
26B-A4B.

### Features

* **models:** Gemma 4 31B (Unsloth `UD-Q4_K_XL`, `UD-Q8_K_XL`), 31B QAT
  (Q4_0) and the 26B-A4B mixture of experts (`UD-Q4_K_XL`, `UD-Q6_K_XL`,
  `UD-Q8_K_XL`) on Strix Halo `gfx1151`.
* **serving:** `gufo serve llm` with every text endpoint, reasoning
  (`reasoning_content`, `reasoning_tokens` and reasoning budgets), native tool
  calls (buffered and streamed), image input through the BF16 vision sidecar,
  batched sessions, and RAM and disk prompt caching; `gufo prompt` and
  `gufo bench` through the extension's drivers.
* **speculation:** `gemma4-assistant` MTP drafting of up to 15 tokens with the
  calibrated policy, and prompt lookup.
* **provider options:** `--image-tokens`, `--draft-calibration`,
  `--draft-trace` and `--prefill-weights`.
* **docs:** the 26B-A4B, 31B and 31B QAT model cards, with their benchmarks
  and quality reports, now live here.

### Bug Fixes

* **reasoning:** reasoning budgets are honored past the thought header.
* **reasoning:** the thought block is closed after tool responses when
  thinking is off.
* **context:** `--context 0` loads the model's native context.
* **build:** the chat template links ICU, which it includes
  ([#1](https://github.com/franzmoca/gemma4-gufo-extension/pull/1), by
  [@fedeizzo](https://github.com/fedeizzo)), and
  prefill attention builds with strict floating point under Clang 24.
