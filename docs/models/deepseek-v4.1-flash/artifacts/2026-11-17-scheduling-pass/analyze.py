#!/usr/bin/env python3
"""Summarize sweep arms: exactness across arms, then per-value timings (rounds 1-3)."""
import glob, json, statistics as st, sys, collections

d = sys.argv[1]
arms = []
for p in sorted(glob.glob(f"{d}/*.json")):
    j = json.load(open(p))
    arms.append((p.rsplit("/", 1)[1][:-5], j))

# Exactness: every arm must give the same A and B hashes per round.
ref = {}
ok = True
for tag, j in arms:
    for r in j["rounds"]:
        key = (r["round"],)
        h = (r["a_sha"], r["b_sha"], r["b_prompt_n"])
        if key in ref and ref[key][0] != h:
            ok = False
            print(f"MISMATCH round {r['round']}: {tag} {h} vs {ref[key][1]} {ref[key][0]}")
        ref.setdefault(key, (h, tag))
print("outputs identical across arms" if ok else "OUTPUTS DIFFER")

by = collections.defaultdict(list)
for tag, j in arms:
    for r in j["rounds"]:
        if r["round"] == 0:
            continue
        by[int(j["label"])].append((tag, r))

print(f"{'pass':>5} {'B prefill tok/s':>16} {'B TTFT s':>9} {'A stall max s':>14} "
      f"{'A stall sum s':>14} {'A decode tok/s':>15}  per-arm B tok/s")
for v in sorted(by):
    rs = [r for _, r in by[v]]
    pf = [r["b_prompt_n"] / r["b_prompt_ms"] * 1000 for r in rs]
    ttft = [r["b_prompt_ms"] / 1000 for r in rs]
    mx = [r["a_during_max_s"] for r in rs]
    sm = [r["a_during_sum_s"] for r in rs]
    dec = [r["a_predicted_n"] / r["a_predicted_ms"] * 1000 for r in rs]
    arm = collections.defaultdict(list)
    for tag, r in by[v]:
        arm[tag].append(r["b_prompt_n"] / r["b_prompt_ms"] * 1000)
    per = " ".join(f"{t[:2]}:{st.mean(x):.0f}" for t, x in sorted(arm.items()))
    print(f"{v:>5} {st.mean(pf):>16.0f} {st.mean(ttft):>9.2f} {st.mean(mx):>8.2f} (≤{max(mx):.2f})"
          f" {st.mean(sm):>14.1f} {st.mean(dec):>15.1f}  {per}")
