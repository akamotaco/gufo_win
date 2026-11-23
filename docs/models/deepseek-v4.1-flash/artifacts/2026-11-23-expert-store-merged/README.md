# The expert store as merged into Gufo (E130)

`arms.json`: streamed `UD-IQ2_XXS` with the `MXFP8` Engram, DSpark and vision,
one fresh server per run, all built with `nix build`: the branch on Gufo
`9e4e9129` (P, before review), on `4c8fa6a2` (R, the reviewed head) and on
`60c48924` (the merge, `flake.lock`).

- `replay`: E127's 53-request Pi coding-agent replay at one session, traced
  at an 80 GiB cache (P, R), untraced in the order P R R P, and once at the
  pin. Replies are SHA-256 prefixes with the per-run tool-call ids removed.
  Each run lists the server's `expert_cache_grant` fields, its seed lines and
  any `expert_cache_release` or `host_pressure_release` events.
- `trace_comparison`: the traced runs compared prepare by prepare, up to
  request 50, where their routes part.
- `pressure`: R at one session after one short request, with a 4 GiB memory
  hog started while idle, untraced and traced: available memory at each step,
  the release and restore events, and whether the repeated request gave the
  same reply.
- `agent_bench`: the 17-task coding-agent benchmark on R and on P's code
  (Gufo `fd84f34`): per task pass, turns, tool calls, tokens and wall time,
  the server's release and refusal counts, and where each pair of transcripts
  first differs.
- `functional`: Gufo's `conversation`, `long-context` and `cache` suites on R
  and at the pin, case by case.

Prompts, replies and transcripts are left out.
