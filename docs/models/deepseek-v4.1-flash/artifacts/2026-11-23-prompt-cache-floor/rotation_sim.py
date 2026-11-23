#!/usr/bin/env python3
"""C conversations, each growing by ~CHUNK tokens of distinct source text per
turn, sent round by round with up to CLIENTS requests in flight, so sessions
take turns holding conversations and resume them from the prompt cache.

Usage: multi.py PORT SRC_DIR CONVERSATIONS TURNS CLIENTS MAX_TOKENS LOG.jsonl
"""
import concurrent.futures as cf
import http.client
import json
import pathlib
import sys
import time

port, src = int(sys.argv[1]), pathlib.Path(sys.argv[2])
conversations, turns, clients, max_tokens = map(int, sys.argv[3:7])
log = sys.argv[7]
CHUNK_CHARS = 21000  # ~6k tokens of C++

files = sorted(p for p in src.rglob("*.cpp") if p.stat().st_size > 2000)
text = "".join(f"\n// file: {p.relative_to(src)}\n" + p.read_text(errors="replace") for p in files)
need = conversations * turns * CHUNK_CHARS
if len(text) < need:
    sys.exit(f"source text too short: {len(text)} < {need} chars")

conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
conn.request("GET", "/v1/models")
model = json.loads(conn.getresponse().read())["data"][0]["id"]
conn.close()

# Each conversation reads its own stretch of the text, so no two share a
# prefix beyond the system prompt.
span = turns * CHUNK_CHARS
histories = [[{"role": "system", "content": f"You are code reviewer #{c}. Answer briefly."}]
             for c in range(conversations)]


def send(c, turn):
    start = c * span + turn * CHUNK_CHARS
    messages = histories[c] + [{"role": "user", "content":
        "Here is more code:\n```cpp\n" + text[start:start + CHUNK_CHARS] +
        "\n```\nIn two sentences, what does this part do?"}]
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                       "max_tokens": max_tokens, "seed": 1234,
                       "stream_options": {"include_usage": True}})
    t = time.time()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    reply, error, usage = [], None, None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if "error" in ev:
            error = ev["error"]
        for ch in ev.get("choices") or []:
            reply.append((ch.get("delta") or {}).get("content") or "")
        usage = ev.get("usage") or usage
    conn.close()
    return c, turn, messages, "".join(reply), {
        "conversation": c, "turn": turn, "status": resp.status, "error": error,
        "usage": usage, "seconds": round(time.time() - t, 1)}


started = time.time()
failed = 0
with cf.ThreadPoolExecutor(clients) as pool:
    for turn in range(turns):
        for c, t, messages, reply, rec in pool.map(lambda c: send(c, turn), range(conversations)):
            histories[c] = messages + [{"role": "assistant", "content": reply or "(no reply)"}]
            failed += rec["status"] != 200 or rec["error"] is not None
            u = rec["usage"] or {}
            print(f"turn {t} conv {c} {rec['status']} prompt {u.get('prompt_tokens')} "
                  f"cached {u.get('cached_tokens')} {rec['seconds']}s"
                  + (f" ERROR {json.dumps(rec['error'])[:200]}" if rec["error"] else ""), flush=True)
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
print(f"DONE {time.time() - started:.0f}s failed={failed}", flush=True)
