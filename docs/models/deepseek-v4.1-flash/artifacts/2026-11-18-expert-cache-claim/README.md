# The expert cache sized by serving (E126)

`arms.json`: fresh servers on streamed `UD-IQ2_XXS` with the `MXFP8` Engram,
DSpark and vision, all built against Gufo `eb9dea37` (the E125 store arm on
the branch that adds the store, the others on the branch that adds the
claim). Each arm lists the server's `expert_cache_grant`, `backing_prepared`
and `ram_admission` log fields and the reasons it refused to grow the
prompt cache's backing.

- `replay`: the 53-request Pi coding-agent replay at one session, traced at
  an 80 GiB cache and in the order A B B A; replies are SHA-256 prefixes with
  the per-run tool-call ids removed.
- `two_sessions_size`: one short request at `--sessions 2`.
- `two_sessions_fill`: 16 distinct 6,000-token prompts, then the first four
  again, at `--sessions 2`, with the claim and with the rejected variant that
  reserved the uncommitted backing.

Prompts and replies are left out.
