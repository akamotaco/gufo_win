# DeepSeek V4.1 Flash benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. ISTA-DASLab GSQ-RCO
backbones: `IQ1_M` resident (first table of each section) and `IQ2_XXS` with
streamed routed experts (second table, after the rule); Unsloth `UD-IQ1_M`
(third table) and `UD-IQ2_XXS` (fourth table) with streamed routed experts,
sharing the ISTA Engram file;
`IQ4_NL` Engram tables read from disk, Lucebox Q2_K/Q4_K DSpark drafter. Gufo only: no other
engine runs DeepSeek V4.1 Flash on this machine, so the tables carry no
reference columns. HTTP through `gufo serve`, greedy, thinking off. The
`IQ1_M` and `UD-IQ2_XXS` single-user tables, their multi-user AR tables and
the `UD-IQ2_XXS` multi-user DSpark table were measured November 10, 2026 on
the production `nix build` of the extension at `255e1739` with Gufo
`0e2de43e`. The other tables come from the DeepSeek fork: October 31 on
`0eb9e89c`, and the loading and `UD-IQ2_XXS` memory tables November 2-3 on
`26c7598f`. Tuned profile
`accelerator-performance`, GPU DPM level `auto`, Tctl 37–58 °C at the start of
each run. Prefill rates
are the server's own prompt timings for real chat text; see
[native prefill](#native-prefill) for `gufo bench`.
[Method and quality](QUALITY.md) · [Experiments](EXPERIMENTS.md) ·
[Streaming design](STREAMING.md)

**Streamed `IQ2_XXS`, `UD-IQ1_M` and `UD-IQ2_XXS`.** The 128.75 GiB, 130 GiB and 154 GiB models keep their dense weights
resident and reads routed experts from a Samsung 990 PRO (PCIe 4.0 NVMe)
into an expert cache that fills the free memory (`--stream-experts`,
defaults). Its speed depends on the disk: a slower SSD costs time on every
expert-cache miss, and misses are most frequent in a new server's first
requests. Each depth sweep and concurrency level starts a
new server, so these rows include that warm-up; a server that has served
similar text decodes faster.

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| DS4.1 Flash IQ1_M AR<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1250.35 | 24.88 |
| 4,096 | 1166.25 | 24.76 |
| 8,192 | 1204.50 | 24.66 |
| 12,288 | 1209.08 | 24.62 |
| 16,384 | 1170.65 | 24.53 |
| 32,768 | 1195.28 | 24.40 |
| 65,536 | 1101.88 | 24.10 |
| 131,072 | 1050.85 | 23.50 |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

---

<!-- bench:single-ar-iq2xxs -->
| DS4.1 Flash IQ2_XXS streamed AR<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1055.67 | 20.96 |
| 4,096 | 1057.61 | 22.24 |
| 8,192 | 1077.23 | 21.77 |
| 12,288 | 1096.16 | 22.60 |
| 16,384 | 1034.58 | 22.74 |
| 32,768 | 960.09 | 22.66 |
| 65,536 | 958.27 | 21.22 |
| 131,072 | 894.97 | 21.80 |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-iq2xxs.svg)

---

<!-- bench:single-ar-udiq1m -->
| DS4.1 Flash UD-IQ1_M streamed AR<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 999.66 | 19.43 |
| 4,096 | 1062.53 | 20.24 |
| 8,192 | 1034.47 | 20.55 |
| 12,288 | 924.84 | 20.66 |
| 16,384 | 984.08 | 20.32 |
| 32,768 | 964.94 | 20.39 |
| 65,536 | 854.50 | 19.28 |
| 131,072 | 844.68 | 19.69 |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-udiq1m.svg)

---

<!-- bench:single-ar-udiq2xxs -->
| DS4.1 Flash UD-IQ2_XXS streamed AR<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1019.95 | 17.29 |
| 4,096 | 1053.71 | 18.35 |
| 8,192 | 1067.24 | 18.52 |
| 12,288 | 1082.83 | 18.56 |
| 16,384 | 1035.23 | 19.07 |
| 32,768 | 1046.99 | 18.92 |
| 65,536 | 943.26 | 17.36 |
| 131,072 | 970.20 | 18.49 |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-udiq2xxs.svg)

## Single user, DSpark

pp is the highest measured rate per depth across mixed/repetitive text. The
mixed workload asks for a summary and a story (a story at 64K); the
repetitive one copies a passage, introduced as random filler for a
transcription test (otherwise `UD-IQ1_M` comments on the passage and stops),
which prompt lookup verifies up to 15 tokens at a time. Greedy output is identical to plain decoding.

<!-- bench:single-dspark -->
| DS4.1 Flash IQ1_M DSpark<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | Gufo tg repetitive (tok/s) |
| ---: | ---: | ---: | ---: |
| 0 | 1240.34 | 30.99 | 75.36 |
| 4,096 | 1212.48 | 31.19 | 74.17 |
| 8,192 | 1198.81 | 32.37 | 75.16 |
| 12,288 | 1201.82 | 34.74 | 75.21 |
| 16,384 | 1135.02 | 30.75 | 73.26 |
| 32,768 | 1140.12 | 32.13 | 71.36 |
| 65,536 | 1059.29 | 23.86 | 67.47 |
| 131,072 | 1051.10 | 32.91 | 64.29 |
<!-- /bench -->

![Single user, DSpark](artifacts/charts/single-dspark.svg)

---

<!-- bench:single-dspark-iq2xxs -->
| DS4.1 Flash IQ2_XXS streamed DSpark<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | Gufo tg repetitive (tok/s) |
| ---: | ---: | ---: | ---: |
| 0 | 1063.61 | 23.91 | 57.38 |
| 4,096 | 1066.77 | 28.86 | 62.57 |
| 8,192 | 1026.58 | 26.29 | 66.78 |
| 12,288 | 890.68 | 23.75 | 58.40 |
| 16,384 | 806.24 | 26.87 | 55.44 |
| 32,768 | 921.94 | 29.32 | 64.48 |
| 65,536 | 874.47 | 21.20 | 62.24 |
| 131,072 | 862.76 | 22.75 | 59.74 |
<!-- /bench -->

![Single user, DSpark](artifacts/charts/single-dspark-iq2xxs.svg)

---

<!-- bench:single-dspark-udiq1m -->
| DS4.1 Flash UD-IQ1_M streamed DSpark<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | Gufo tg repetitive (tok/s) |
| ---: | ---: | ---: | ---: |
| 0 | 1018.34 | 22.29 | 52.42 |
| 4,096 | 948.66 | 23.42 | 60.37 |
| 8,192 | 965.43 | 26.34 | 63.56 |
| 12,288 | 846.81 | 22.79 | 55.87 |
| 16,384 | 779.05 | 24.80 | 54.17 |
| 32,768 | 872.23 | 21.50 | 61.74 |
| 65,536 | 816.24 | 18.79 | 59.06 |
| 131,072 | 826.08 | 22.79 | 57.27 |
<!-- /bench -->

![Single user, DSpark](artifacts/charts/single-dspark-udiq1m.svg)

---

<!-- bench:single-dspark-udiq2xxs -->
| DS4.1 Flash UD-IQ2_XXS streamed DSpark<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | Gufo tg repetitive (tok/s) |
| ---: | ---: | ---: | ---: |
| 0 | 1042.97 | 24.53 | 49.46 |
| 4,096 | 1004.79 | 25.48 | 57.72 |
| 8,192 | 1050.50 | 25.15 | 64.10 |
| 12,288 | 1070.83 | 25.06 | 61.60 |
| 16,384 | 1023.93 | 30.38 | 60.55 |
| 32,768 | 1037.42 | 25.39 | 61.92 |
| 65,536 | 949.52 | 19.23 | 58.61 |
| 131,072 | 960.85 | 23.61 | 57.32 |
<!-- /bench -->

![Single user, DSpark](artifacts/charts/single-dspark-udiq2xxs.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Every session is prepared from a C1 prompt checkpoint before timed decoding;
throughput sums individual request decode rates. Concurrent sessions decode
in one forward; every output matches its isolated C1 reference.

<!-- bench:multi-ar -->
| DS4.1 Flash IQ1_M AR<br>Users | Gufo AR (tok/s) |
| ---: | ---: |
| 1 | 24.21 |
| 2 | 41.83 |
| 4 | 65.37 |
| 6 | 78.27 |
| 8 | 87.91 |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

---

<!-- bench:multi-ar-iq2xxs -->
| DS4.1 Flash IQ2_XXS streamed AR<br>Users | Gufo AR (tok/s) |
| ---: | ---: |
| 1 | 20.53 |
| 2 | 36.54 |
| 4 | 59.33 |
| 6 | 71.20 |
| 8 | 81.90 |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-iq2xxs.svg)

---

<!-- bench:multi-ar-udiq1m -->
| DS4.1 Flash UD-IQ1_M streamed AR<br>Users | Gufo AR (tok/s) |
| ---: | ---: |
| 1 | 19.05 |
| 2 | 33.89 |
| 4 | 56.85 |
| 6 | 69.20 |
| 8 | 79.73 |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-udiq1m.svg)

---

<!-- bench:multi-ar-udiq2xxs -->
| DS4.1 Flash UD-IQ2_XXS streamed AR<br>Users | Gufo AR (tok/s) |
| ---: | ---: |
| 1 | 16.72 |
| 2 | 30.95 |
| 4 | 53.62 |
| 6 | 66.34 |
| 8 | 77.87 |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-udiq2xxs.svg)

## Multiple users, DSpark

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. Concurrent sessions verify their blocks jointly in batches of up to
16 rows; every output matches its isolated AR reference.

**Resident `IQ1_M` with DSpark is not supported in this table on the current
build.** Its weights and the 4.59 GiB drafter leave about 2.2–2.6 GB of device
heap, below the 4 GiB that Gufo's prompt cache keeps free, so the cache holds no
checkpoints. Every session would have to be prepared from a C1 checkpoint, so
the table cannot run. The `IQ1_M` rows below were measured on the fork, which
kept a 1 GiB reserve. Streamed builds still prepare their sessions, though the
same gate holds their cache at the 2.2 GB it commits at startup.

<!-- bench:multi-dspark -->
| DS4.1 Flash IQ1_M DSpark<br>Users | Gufo mixed (tok/s) | Gufo repetitive (tok/s) |
| ---: | ---: | ---: |
| 1 | 31.41 | 75.43 |
| 2 | 45.87 | 88.65 |
| 4 | 60.44 | 94.07 |
| 6 | 66.50 | 107.06 |
| 8 | 74.88 | 99.25 |
<!-- /bench -->

![Multiple users, DSpark](artifacts/charts/multi-dspark.svg)

---

<!-- bench:multi-dspark-iq2xxs -->
| DS4.1 Flash IQ2_XXS streamed DSpark<br>Users | Gufo mixed (tok/s) | Gufo repetitive (tok/s) |
| ---: | ---: | ---: |
| 1 | 23.92 | 53.42 |
| 2 | 37.98 | 76.89 |
| 4 | 52.58 | 88.89 |
| 6 | 61.17 | 99.46 |
| 8 | 71.11 | 95.71 |
<!-- /bench -->

![Multiple users, DSpark](artifacts/charts/multi-dspark-iq2xxs.svg)

---

<!-- bench:multi-dspark-udiq1m -->
| DS4.1 Flash UD-IQ1_M streamed DSpark<br>Users | Gufo mixed (tok/s) | Gufo repetitive (tok/s) |
| ---: | ---: | ---: |
| 1 | 21.57 | 50.91 |
| 2 | 37.20 | 74.56 |
| 4 | 52.40 | 87.62 |
| 6 | 60.22 | 97.60 |
| 8 | 72.26 | 95.46 |
<!-- /bench -->

![Multiple users, DSpark](artifacts/charts/multi-dspark-udiq1m.svg)

---

<!-- bench:multi-dspark-udiq2xxs -->
| DS4.1 Flash UD-IQ2_XXS streamed DSpark<br>Users | Gufo mixed (tok/s) | Gufo repetitive (tok/s) |
| ---: | ---: | ---: |
| 1 | 21.75 | 48.30 |
| 2 | 39.19 | 70.37 |
| 4 | 58.57 | 85.18 |
| 6 | 63.60 | 95.75 |
| 8 | 72.34 | 96.26 |
<!-- /bench -->

![Multiple users, DSpark](artifacts/charts/multi-dspark-udiq2xxs.svg)

## Prompt lookup

Repeating a 3,000-character C++ file with one function renamed (greedy, 900
tokens, think off): 67.6 tok/s with 16-row verification against 58.5 with the
previous 7-row limit; 62 verifications instead of 131, every copy accepted,
identical output ([E15](EXPERIMENTS.md)). Without a drafter, lookup alone
verifies copies the same way.

## Native prefill

`gufo bench` (synthetic token IDs, one 2048-row pass per pp2048) on the
October 29 `cd8b92c1` build: pp2048 1,394 ± 37 tok/s at depth 0 (three runs; 1,205 on the
October 26 build, [E16 record](artifacts/2026-10-26-prefill-batch/summary.txt)). Served chat prefill is lower for two measured
reasons: real text costs 5–7% more than the synthetic IDs in matched
batches, and the prompt cache's stable-boundary checkpoint adds a 3-token
pass (about 65 ms) to every chat request ([backlog](BACKLOG.md)). A profile
of served requests shows no other host idle time.

Prefill runs the decoder layers (20–39) over each prompt's last 140 tokens
only (decoder SWA bounded replay, [QUALITY.md](QUALITY.md));
`--exact-prefill` runs every decoder row at about half the prefill speed.
4,096-token batches gain 2–3% natively but leave the prompt cache no
memory in the default resident setup (E16), so 2048 stays the batch; a
streamed build runs long prompts in 4,096-token passes on demand (E134).

## Vision encoder

`ds41_vision_probe` (gpu-test build, third of three encodes), ViT and aligner
of one image; the span then prefills like text.

| DS4.1 Flash vision<br>Image | Patches | Tokens | Encode (ms) |
|---|---:|---:|---:|
| corn.jpeg | 1,551 | 189 | 83 |
| carrots.jpeg | 3,774 | 444 | 236 |
| 4000 × 3000 (token cap) | 8,748 | 1,001 | 746 |

`gufo serve`: carrots (456 prompt tokens) prefilled in 1.6 s including the
encode, then decoded at 23.6 tok/s; the capped image (1,011 prompt tokens)
answered in 4.1 s. The sidecar adds 0.90 GiB of device memory.

## Loading time

C1, capacity 262144, DSpark. Page cache dropped before the start, to HTTP
readiness; the weights upload from the files directly. Streamed builds
upload their dense weights and preload the built-in ranking of frequently
routed experts before they are ready.

<!-- bench:loading -->
| DS4.1 Flash<br>Target | Gufo ready (s) |
| --- | ---: |
| IQ1_M | 20.04 |
| IQ2_XXS streamed | 17.33 |
| UD-IQ1_M streamed | 17.33 |
| UD-IQ2_XXS streamed | 17.07 |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

Capacity 262144 tokens, AR. Device total minus free, which on this unified
memory APU tracks system-wide use (3.01 GiB idle before the server starts);
it includes the prompt cache's retained checkpoints. Streamed `IQ2_XXS`,
`UD-IQ1_M` and `UD-IQ2_XXS` fill the memory their sessions leave, minus
2 GiB, with the expert cache, so their rows show the cache's size, not a
requirement (the `IQ2_XXS` and `UD-IQ1_M` rows predate this sizing and left
4 GiB).

<!-- bench:memory -->
| DS4.1 Flash IQ1_M AR<br>Workload | Gufo GiB |
| --- | ---: |
| pp2048 + tg128 | 119.50 |
| 16K prefix, pp4096 + tg128 | 119.63 |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

---

<!-- bench:memory-iq2xxs -->
| DS4.1 Flash IQ2_XXS streamed AR<br>Workload | Gufo GiB |
| --- | ---: |
| pp2048 + tg128 | 120.86 |
| 16K prefix, pp4096 + tg128 | 120.98 |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-iq2xxs.svg)

---

<!-- bench:memory-udiq1m -->
| DS4.1 Flash UD-IQ1_M streamed AR<br>Workload | Gufo GiB |
| --- | ---: |
| pp2048 + tg128 | 120.91 |
| 16K prefix, pp4096 + tg128 | 121.03 |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-udiq1m.svg)

---

<!-- bench:memory-udiq2xxs -->
| DS4.1 Flash UD-IQ2_XXS streamed AR<br>Workload | Gufo GiB |
| --- | ---: |
| pp2048 + tg128 | 122.00 |
| 16K prefix, pp4096 + tg128 | 122.12 |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-udiq2xxs.svg)

Resident components (`ds41_load_probe --upload` and `gufo serve`):

| DS4.1 Flash<br>Component | GiB |
| --- | ---: |
| Weights (token embedding stays in the mapped file) | 112.27 |
| One session at the native 1M context | +2.3 (0.9 state, 1.4 batch scratch shared by all sessions) |
| Each further session at 1M | +0.9 |
| DSpark drafter, Q2_K/Q4_K | +4.59 |
| Vision sidecar | +0.90 |

With the drafter, one session at the native context and the server ready,
1.75 GiB of host memory remains available, and the prompt cache holds about
1.25 GB of checkpoints. A 27,559-token prompt captured in 9.5 ms (101 MB)
and restored in 2.9 ms with the same reply; short turns capture in
0.9–4.2 ms and restore in 0.8 ms.
