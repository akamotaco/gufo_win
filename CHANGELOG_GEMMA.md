# Gemma fork changelog

Changes made in this fork's Gemma 4 engine, newest first. Each entry is a brief
summary; its pull request has the details, measurements and validation.
Upstream Gufo's changes are in [CHANGELOG.md](CHANGELOG.md).

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
