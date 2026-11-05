# Functional suites

Gemma 4 runs Gufo's own HTTP suites (`tests/functional/` of a Gufo
checkout). The host harness has no `gemma4` sampling preset and no Gemma
expectations, so [run.py](run.py) copies it beside the report, applies
[host-harness.patch](host-harness.patch) and runs the copy. The host checkout
stays read only. The patch is the `feat/gemma` harness (`83b764b4`) rebased
onto gemma-main and must apply without fuzz.

They need a Gufo built with this extension, real weights and a GPU. Run them
with the Gufo development shell's Python. Each run creates a new report
directory; the patched harness goes to `<output>.harness/`, with its
provenance (host revision, patch hash) in `provenance.json`.

```sh
G=~/projects/gufo-ext M=~/models/gemma-4
nix develop $G -c python3 tests/functional/run.py --host $G -- \
  --output out/31b-mtp --record-baseline --sampling-preset gemma4 \
  --expected-input-modalities text,image --suite tools --suite cache -- \
  $G/build/gemma-ext-release/gufo serve llm \
  --model $M/gemma-31b-it/gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --mmproj $M/gemma-31b-it/mmproj-BF16.gguf --sessions 2 \
  --speculative mtp --draft-tokens 15 \
  --mtp-model $M/drafts/31b-unsloth-mtp/MTP/mtp-gemma-4-31B-it-Q8_0.gguf
```

Pass `--draft-tokens 15` explicitly: the host cannot tell Gemma whether it was
given, and its own default is 7.

## What the patch adds

Gemma-specific, keyed by `--sampling-preset gemma4`:

- `run.py`, `openai_sdk.py`: the `gemma4` preset (temperature 1, top-k 64,
  top-p 0.95; thinking off by default and requested with
  `chat_template_kwargs.enable_thinking`, since Gemma has no effort levels).
- `templates.py`: facts of Gemma's chat template that bound reuse and
  arguments. History drops the thinking-off opening (7 tokens) and earlier
  reasoning. A string `const` shows as a one-value `enum`; a non-string
  `const` is hidden. Calls carry typed values. Gemma may skip thinking when
  it is on.
- The suites that read them: `cache_concurrency`, `cache_growth`,
  `cache_shared_prefix`, `image_inputs`, `system_injection`, `tool_agent`,
  `tool_choice`, `tool_native`, `tool_reasoning`, `tool_streaming`.

Model-neutral harness fixes that belong upstream:

- `run.py`: restart the server before `sampling-defaults`, `progress`,
  `cache-edits` and `cache-growth`, and after the RAM cache refuses a capture
  at its record limit. Each suite's server log is recorded in
  `report.json` `server_logs` (read by `metrics.py` and `server_metrics.py`).
  `--max-pending-per-client` defaults to `sessions + 1` for the metrics suite.
- `openai_sdk.py`: thread-safe check recording, interrupted Responses calls
  accepted when not executable, the auto-tools stop limit cut one token
  before a complete call, and both color spellings for the image tool call.
