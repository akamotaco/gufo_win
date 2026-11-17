#!/usr/bin/env python3
"""Arrival-during-decode probe for the scheduling profile's pass_tokens.

User A streams a greedy reply; once it has produced --lead chunks, user B sends
an uncached long prompt. Records B's server prefill timing and A's gaps between
streamed chunks while B prefills, plus output hashes for exactness checks.
"""
import argparse, hashlib, http.client, json, sys, threading, time

def post(port, body, stream):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(body),
              {"Content-Type": "application/json"})
    return c, c.getresponse()

def run_a(port, prompt, max_tokens, out, lead_evt, lead):
    body = {"model": "bench", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": 1, "stream": True,
            "stream_options": {"include_usage": True}}
    c, r = post(port, body, True)
    times, text, timings = [], [], None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices", []):
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                times.append(time.monotonic())
                text.append(piece)
                if len(times) == lead:
                    lead_evt.set()
    lead_evt.set()
    out.update(times=times, text="".join(text), timings=timings)

def run_b(port, prompt, out):
    body = {"model": "bench", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32, "temperature": 0, "seed": 1, "stream": False}
    t0 = time.monotonic()
    c, r = post(port, body, False)
    resp = json.loads(r.read())
    out.update(t0=t0, t1=time.monotonic(), resp=resp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--b-chars", type=int, default=30000)
    ap.add_argument("--a-tokens", type=int, default=768)
    ap.add_argument("--lead", type=int, default=16)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    corpus = open(args.corpus).read()
    results = []
    # Round 0 warms the expert cache and is reported but excluded downstream.
    for rnd in range(args.rounds + 1):
        a_prompt = ("Write a long, detailed technical essay about the history of "
                    f"lighthouses and their keepers, part {rnd}.")
        off = rnd * args.b_chars
        b_prompt = ("Summarize the following engineering notes in three sentences.\n\n"
                    + corpus[off:off + args.b_chars])
        a, b, evt = {}, {}, threading.Event()
        ta = threading.Thread(target=run_a, args=(args.port, a_prompt, args.a_tokens, a, evt, args.lead))
        ta.start()
        evt.wait()
        tb = threading.Thread(target=run_b, args=(args.port, b_prompt, b))
        tb.start(); tb.join(); ta.join()
        bt = b["resp"].get("timings", {})
        gaps = [t1 - t0 for t0, t1 in zip(a["times"], a["times"][1:])]
        # A's gaps that overlap B's request span (arrival to reply).
        during = [t1 - t0 for t0, t1 in zip(a["times"], a["times"][1:])
                  if t1 > b["t0"] and t0 < b["t1"]]
        b_text = b["resp"]["choices"][0]["message"].get("content") or ""
        rec = {
            "round": rnd,
            "b_prompt_n": bt.get("prompt_n"), "b_cache_n": bt.get("cache_n"),
            "b_prompt_ms": bt.get("prompt_ms"),
            "b_wall_s": b["t1"] - b["t0"],
            "a_chunks": len(a["times"]),
            "a_predicted_n": (a["timings"] or {}).get("predicted_n"),
            "a_predicted_ms": (a["timings"] or {}).get("predicted_ms"),
            "a_gap_max_s": max(gaps) if gaps else None,
            "a_during_n": len(during),
            "a_during_max_s": max(during) if during else None,
            "a_during_sum_s": sum(during),
            "a_sha": hashlib.sha256(a["text"].encode()).hexdigest()[:16],
            "b_sha": hashlib.sha256(b_text.encode()).hexdigest()[:16],
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump({"label": args.label, "rounds": results}, open(args.out, "w"), indent=1)

if __name__ == "__main__":
    main()
