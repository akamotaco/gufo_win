# AGENTS.md

This repository is a Gufo source extension: the DeepSeek V4.1 Flash engine
(provider `deepseek41`), its kernels, tools and documentation.
[UPSTREAM.md](UPSTREAM.md) describes the source layout. The production target is
Linux x86-64 AMD Strix Halo (`gfx1151`).

## Build

```sh
nix build                    # Gufo with this extension: ./result/bin/gufo
nix flake update gufo        # move the pinned upstream Gufo (flake.lock)
```

The flake builds tracked files only; stage new files before `nix build`. Measure
performance with `result/bin/gufo` from `nix build`, and keep `flake.lock` the
same in both arms of a comparison.

Tests and development builds run from a Gufo checkout at the pinned revision,
with this repository as an extension directory:

```sh
cd /path/to/gufo
nix develop -c cmake --preset gpu-test -B build/ds41 \
  -DGUFO_EXTENSIONS='*' -DGUFO_EXTENSION_DIRS=/path/to/ds41-gufo-extension
nix develop -c cmake --build build/ds41 --target deepseek41flash_tests
cd build/ds41 && nix develop /path/to/gufo -c ctest -R '^deepseek41flash\.' --output-on-failure
```

`deepseek41flash_tests` builds every test binary, so `ctest` never runs a stale
one. Run the test covering the change (`-R '^deepseek41flash\.expert_store$'`).
The `cpu` tests need no GPU; the `hip` tests need `gfx1151` but no model.

## Extension API

Code includes Gufo only through `<gufo/ext/...>`. `extension.json` declares the
minimum API version. When the extension needs something the API lacks, change
upstream Gufo first: one small PR with its `ext-api` change record under
`include/gufo/ext/changes/`, described on Gufo's own terms (no extension
results). Raise `api_version` once it merges.

## Formatting

`.clang-format` applies to new and changed C++ lines. Most of `runtime/` and
`kernels/rocm/` predates it: never run `clang-format -i` or a format fix over
those directories, since it rewrites unrelated code. Check the files you touched
with `clang-format --dry-run` and compare the warning count against the base
revision; wrap long lines by hand.

## Functional checks

Gufo's text API suites (`tests/functional/` in a Gufo checkout) run against
`gufo serve` of this build. Select suites explicitly: `conversation`, `tools`,
`long-context`, `cache`, `image-inputs`, `tool-images` and `reasoning-budget`
cover this provider ([docs/VALIDATION.md](docs/VALIDATION.md)). Include
`long-context` and `cache` for continuation or prompt-cache changes, and
`tests/functional/pi_agent.py` for coding-agent loops. A missing-model skip is
not a pass.

## Measuring

- Measure what users run: streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
  DSpark and vision, unless the change targets another build.
- Check the host first: `tuned-adm active` is `accelerator-performance` and the
  fans are under PWM control. Never pin GPU clocks.
- Start a fresh server per arm: the expert cache, prompt cache and DSpark policy
  depend on request history, and the expert cache size on the memory free at
  startup (log it).
- Alternate the order of the arms (ABBA). A second run is ~3% slower and the
  first run after idle up to 25% faster. Build nothing during timing runs.
- An output-preserving change must give identical replies and draft counts in
  every arm; check that before reading timings.
- For agent workloads, record one session with `tools/ds41/agent_replay.py`
  and replay it against each arm, so every arm sees the same requests.
- Expert-streaming rules can be compared offline first: `DS41_EXPERT_TRACE=PATH`
  writes a routing trace for `tools/ds41/{prefetch,evict,replay}_sim.py`
  ([tools/README.md](tools/README.md)).
- Keep small measured gains. Report noisy results as inconclusive rather than
  repeating until a run agrees.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md),
and [.agents/skills/benchmark-model/SKILL.md](.agents/skills/benchmark-model/SKILL.md)
to refresh the benchmark card. `tools/bench/` holds standalone benchmarks of this
extension's kernels (`nix develop github:gufo-org/gufo -c tools/bench/build.sh <name>`).
Gufo's `tools/prof/prof.py`, `tools/prof/isa_mix.py` and
`tools/bench/gfx1151_peak.hip` run from a Gufo checkout. Successful optimizations
become the default, with no switch to restore the old path.

## Documentation

Model documentation lives in [docs/models/deepseek-v4.1-flash/](docs/models/deepseek-v4.1-flash/):

- `EXPERIMENTS.md`: one `E<n>` section per experiment, retained or rejected,
  plus its index row. Retained JSON, patches and recordings go under
  `artifacts/<date>-<topic>/`.
- `BACKLOG.md`: open work, ranked; move finished items to its done table.
- `BENCHMARKS.md` and `QUALITY.md`: the current card and its evidence.
- `DECODE.md`, `PREFILL.md`, `STREAMING.md`: design and analysis; update them
  when the behavior they describe changes.

Committed artifacts contain no usernames, host names or private paths; scrub
recordings before adding them.

## Development

- Follow Conventional Commits with a single-line message, scope `deepseek41`.
- Add each user-facing change to the `Unreleased` section of
  [CHANGELOG.md](CHANGELOG.md), linking its experiment.
- PR descriptions, the changelog and the README state changes and results, not
  how the work was done.
- Use `gh` for GitHub operations after checking `gh auth status`.

## Releasing

A release is one `chore(deepseek41): release X.Y.Z` commit on `main` and its tag:

1. Set `version` in `extension.json`, and move `Unreleased` in `CHANGELOG.md` to
   `## X.Y.Z (YYYY-MM-DD)`, naming the pinned Gufo revision and the extension API
   it needs.
2. Run the functional checks on the build at that pin and record them under
   `Functional checks` in [docs/VALIDATION.md](docs/VALIDATION.md).
3. After the commit merges, tag it and push the tag:
   `git tag -a vX.Y.Z -m X.Y.Z <commit> && git push origin vX.Y.Z`.
   `.github/workflows/release.yml` checks the tag against `extension.json` and
   publishes the changelog section as the GitHub release.

Raise the minor version when the Gufo pin or the extension API requirement
changes, and the patch version for fixes on the same pin.
