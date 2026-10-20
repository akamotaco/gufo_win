# Gemma fork changelog

Changes made in this fork's Gemma 4 engine, newest first. Each entry is a brief
summary; its pull request has the details, measurements and validation.
Upstream Gufo's changes are in [CHANGELOG.md](CHANGELOG.md).

## 2026-10-20: MTP on depth-upscaled 31B targets ([#4](https://github.com/franzmoca/gufo/pull/4))

- MTP decoding accepts targets of up to 96 layers instead of 64, so
  depth-upscaled 31B models such as BeaverAI/Spectre-37B-v1a (72 layers) run
  with the 31B drafter instead of failing at the first verification cycle:
  greedy decode 9.8 to 18.0 tok/s on Spectre-37B-v1a Q4_K_M. 26B-A4B and 31B
  are unchanged.

## 2026-10-19: Continuations from the component cache ([#3](https://github.com/franzmoca/gufo/pull/3))

- Gemma 4 serves continuations through upstream's component cache instead of
  whole-session snapshots. Checkpoints keep global rows, sliding-window rows
  and a small private state apart, so a conversation's checkpoints share rows
  and a restore loads only what its slot lacks: a 26B swipe of a 16K-token
  prompt restores 1 MB in under 1 ms.
- The disk cache (`--cache-disk`) uses the component disk tier: after a
  restart, a 5.5K-token 31B conversation reaches its next first token in
  0.43 s instead of 9.9 s.
- Prefill no longer splits at checkpoints, captures no longer wait for
  another session's prefill (26B: 8–9 ms against legacy's 204–325 ms), and
  two requests finishing prefill together decode as one batch.
- Fixed 31B restores stalling 4–17 ms after the cache grew its host memory,
  and captures stalling while it grew.
- Synced with upstream main through gufo-org/gufo#477 (0.17.1).
- `cache_snapshot_ms` reports a prompt capture's own time; it included the
  wait until the scheduler collected the finished capture.
- On 31B MTP, 1,105 of 1,164 per-request timing measures match the legacy
  cache, and prompt captures take a median 0.58 of legacy's time. Restores
  into a free slot, where legacy reused the other slot's session, remain
  slower.

## 2026-10-18: Prompt lookup from upstream and fewer request stalls ([#2](https://github.com/franzmoca/gufo/pull/2))

- Synced with upstream main through gufo-org/gufo#613. Gemma 4 takes its
  prompt-lookup rules where they measured faster, and copies only what every
  comparable earlier passage agrees on: log extraction +26–28% on 26B-A4B and
  +15% on 31B, edits inside long contexts +2–3%.
- A request waiting for its prompt checkpoint no longer waits for another
  request's whole generation (a 1.2 s request took up to 20 s; now 1.7 s).
- Prompt-lookup copies are reported in `lookup_n` / `lookup_n_accepted` and
  the `/metrics` counters.

## 2026-10-17: Faster prefill at depth and MTP verification ([#1](https://github.com/franzmoca/gufo/pull/1))

- Prefill attention and projections: 31B prefill +3.5% with no context, +9%
  at 32K and +22% at 128K; 26B-A4B +12–17% from 16K.
- MTP verification of two to four rows on one wave per head, re-priced draft
  costs and fewer decode round-trips: 31B MTP decode +3.5–9%, 26B up to +13%
  at depth.
- Synced with upstream v0.17.0. A vision sidecar beside the model is now
  prepared at load: the first image request is faster, and text-only serving
  loads about 0.7 s slower and holds about 2.4 GiB more.
- Tool-call fixes: argument schemas beside arrays without `items`, and
  bounded whitespace before a required tool call.
- Benchmark cards refreshed on October 17; the 31B QAT card uses a reduced
  grid.
