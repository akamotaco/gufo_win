# Validation

How this extension was checked against the engine it replaced, the DeepSeek V4.1
build of the [franzmoca/gufo](https://github.com/franzmoca/gufo) fork, on one Strix
Halo (`gfx1151`) at the accelerator-performance profile.

## Speed against the fork

Streamed `UD-IQ2_XXS`, `IQ4_NL` Engram, DSpark, vision, one session, disk cache, cold
expert cache, balanced order, average of two runs, November 7, 2026:

| Greedy | Extension | Fork |
| --- | ---: | ---: |
| Prefill, 3.5K tokens (tok/s) | 595 | 586 |
| Prefill, 24K tokens (tok/s) | 1,045 | 1,041 |
| Decode, chat (tok/s) | 19.4 | 19.4 |
| Decode, code (tok/s) | 28.5 | 28.7 |
| Decode, long document (tok/s) | 27.3 | 26.9 |

The same with the `MXFP8` Engram, on upstream Gufo `main` (decode averages four runs
per build, November 7-8, 2026; prefill is one run):

| Greedy | Extension | Fork |
| --- | ---: | ---: |
| Prefill, 3.5K tokens (tok/s) | 589 | 586 |
| Prefill, 24K tokens (tok/s) | 1,050 | 1,022 |
| Decode, chat (tok/s) | 17.8 | 17.9 |
| Decode, code (tok/s) | 26.1 | 26.1 |
| Decode, long document (tok/s) | 25.2 | 25.3 |

Decode varies more between runs than between the two builds: DSpark learns its draft
lengths from measured step times, and the expert cache size depends on the memory free
at startup. Generated text is identical on these requests.

## README figures

Greedy, `MXFP8` Engram, DSpark, vision, one session, disk cache, November 8, 2026.
Prefill is a 24K-token document summarized. The decode requests generate 512 tokens:

- **Chat:** a short story about a lighthouse keeper, written from a one-line prompt.
- **Long document:** a section-by-section explanation of a 24K-token document.
- **Repetitive:** a 1.8K-token passage of random filler text, introduced as filler
  for a transcription test and copied verbatim, as on the model cards. All four
  builds return the same text. Prompt lookup verifies up to 15 tokens at a time.

The repetitive figure averages two requests (different passages) in each of two runs
per build, in balanced order:

| Build | Run 1 | Run 2 | Average |
| --- | ---: | ---: | ---: |
| `IQ1_M`, resident | 84.2, 84.6 | 85.1, 84.9 | 84.7 |
| `IQ1_M`, streamed | 83.6, 83.5 | 83.1, 83.3 | 83.4 |
| `UD-IQ1_M`, streamed | 72.1, 77.9 | 70.9, 79.1 | 75.0 |
| `UD-IQ2_XXS`, streamed | 61.7, 72.8 | 61.3, 72.9 | 67.2 |

The expert cache served 98% (`IQ1_M`), 90% (`UD-IQ1_M`) and 83% (`UD-IQ2_XXS`) of
routed experts on these requests.

## Resident against streamed `IQ1_M`

`IQ1_M`, `MXFP8` Engram, DSpark, vision, one session, average of two runs, balanced
order, November 8, 2026:

| Greedy | Resident | `--stream-experts` |
| --- | ---: | ---: |
| Prefill, 2.3K tokens (tok/s) | 1,032 | 1,183 (+15%) |
| Prefill, 24K tokens (tok/s) | 977 | 1,282 (+31%) |
| Repeat of the 24K-token prompt (s) | 24.6 | 0 (checkpoint) |
| Decode, chat (tok/s) | 28.2 | 26.0 (-8%) |
| Decode, code (tok/s) | 80.3 | 76.5 (-5%) |
| Decode, long document (tok/s) | 37.2 | 35.8 (-4%) |
| Decode, repetitive (tok/s) | 84.7 | 83.4 (-2%) |

Resident `IQ1_M` with the drafter and the vision projector leaves under 1 GiB of GPU
heap. The prefill buffer is sized from what is left (624 tokens per batch instead of
2,048), and Gufo's 4 GiB heap reserve leaves no room for prompt-cache checkpoints. A
conversation that continues on its session still reuses it (24K-token history: the
next turn starts in 0.2 s), but repeats, edits and branches of earlier prompts prefill
again.

## Functional checks

Release 0.5.1, November 24, 2026, on streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
DSpark and the vision projector, built at Gufo `60c48924` and run with Gufo's runner at
the same revision:

- With two sessions, the `conversation`, `tools`, `long-context`, `cache` (disk restart
  included) and `image-inputs` suites passed. After them, `tool-images` fails its
  text-array replay as in 0.5.0. `reasoning-budget` fails its control case: the
  uncapped reply reasons for 62 tokens, under the suite's 64-token floor, with the same
  answer. E138's ten-way attention split changes the rounding (0.5.0 reasoned for 68
  tokens; the build before E138 passes).
- The Pi coding-agent check passed its 8 tasks and `image-read`.
- With two sessions, two fresh 29K-token prompts sent at once both answered correctly.
  One took the wide prefill rows, and they were returned when it ended.
- The 53-request Pi coding-agent replay is self-consistent in both 0.5.0 and 0.5.1
  (two runs each, identical replies and drafts). E138 changes 33 of the 53 replies
  against 0.5.0. Over the whole session decode takes 139.3 s instead of 146.9 s (5.1%
  faster); in the 20 replies that match, it is 9.5% faster (19 of 20 faster).

Release 0.5.0, November 24, 2026, on streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
DSpark and the vision projector, built at Gufo `60c48924` and run with Gufo's runner at
the same revision:

- With two sessions, the `conversation`, `tools`, `long-context`, `cache` (disk restart
  included), `image-inputs` and `reasoning-budget` suites passed. After them,
  `tool-images` fails its text-array replay as in 0.4.0 (83 of 86 tokens restored);
  alone it passes.
- The 53-request Pi coding-agent replay gave E130's replies and drafts (3,787 drafted,
  3,110 accepted); against 0.4.0, replies 44, 45 and 49 differ, as E130 records for
  Gufo `60c48924`.

Release 0.4.0, November 23, 2026, on streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
DSpark and the vision projector, built at Gufo `e7e38411` and run with Gufo's runner at
the same revision:

- With two sessions, the `conversation`, `tools`, `long-context`, `cache` (disk restart
  included), `image-inputs` and `reasoning-budget` suites passed. After them,
  `tool-images` fails its text-array replay as in 0.3.0 (83 of 86 tokens restored);
  alone it passes.
- The 53-request Pi coding-agent replay gave 0.3.1's replies except requests 44 and 45,
  where Gufo `e7e38411` resumes from three fewer cached tokens (E129).

Release 0.3.1, November 21, 2026, on streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
DSpark and the vision projector, built at Gufo `a27bbf98` and run with Gufo's runner at
the same revision: with two sessions, the `conversation`, `tools`, `long-context`,
`cache` (disk restart included) and `image-inputs` suites passed. The unsloth Q8_0
DSpark drafter loads with `--cache-disk` and answers a request (#11).

Release 0.3.0, November 17, 2026, on streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
DSpark and the vision projector, built at Gufo `a27bbf98` and run with Gufo's runner at
the same revision:

- With two sessions, the `conversation`, `tools`, `long-context`, `cache` (disk restart
  included), `prefill-scheduling` and `image-inputs` suites passed. After them,
  `tool-images` fails its text-array replay: the replay lands on the other session and
  restores 83 of its 86 tokens instead of all of them. The previous pin, `0e2de43e`,
  fails the same way after the same suites; `tool-images` alone passes with one or two
  sessions on both.
- With one session, `reasoning-budget` with thinking on passed.
- The Pi coding-agent replay passed its 8 tasks and `image-read`.

Release 0.2.0, November 11, 2026, on streamed `UD-IQ2_XXS` with the `MXFP8` Engram and
DSpark, built at Gufo `0e2de43e` and run with Gufo's runner at the same revision:

- The `conversation`, `tools`, `long-context` and `cache` suites (disk restart
  included), and `reasoning-budget` with thinking on, passed.
- With the vision projector loaded, the `image-inputs` and `tool-images` suites passed.
- The Pi coding-agent replay passed 8 of 8 tasks, `image-read` included.

On November 8, 2026, on streamed `UD-IQ2_XXS` and the resident Coder with DSpark:

- Gufo's `conversation`, `tools`, `long-context`, `cache` (disk restart included),
  `image-inputs`, `tool-images` and `reasoning-budget` functional suites. The cache
  suite's `disk-spacing` case needs [gufo-org/gufo#1089](https://github.com/gufo-org/gufo/pull/1089),
  which fixes the request it replays after the restart.
- The Pi coding-agent replay: 7 of 7 tasks.
- `gufo prompt`, and `gufo bench` greedy, sampled and with `--exact-prefill`.
- Every build in the README's model table marked as tested, started with vision, the
  disk cache and DSpark.

## Differences from the fork

- The prompt cache keeps 17 transfer streams with one session (16 plus one per
  session); the fork kept 16. Before Gufo let extensions set the count, the extension
  had 64, which held about 0.4 GiB more GPU memory: streamed builds got that much less
  expert cache, and resident `IQ1_M` with vision got the smaller prefill batch above
  (the fork gets 1,296 tokens, so a 24K-token prompt prefills in 20.4 s instead of
  24.6 s).
- Gufo's 4 GiB heap reserve can trim the prompt-cache backing at startup (seen once in
  four runs: 829 MiB instead of 1 GiB, with no speed change).
- The engine-against-oracle checks in
  [QUALITY.md](models/deepseek-v4.1-flash/QUALITY.md) use probes built in the fork's
  tree; this extension does not build them. Greedy text matches the fork, so they
  were not repeated.
