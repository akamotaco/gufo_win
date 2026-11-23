#!/usr/bin/env python3
"""A scripted coding-agent session, sent one request at a time:

- every conversation starts with the same ~6k-token system prompt and tools;
- the main agent grows by tool results of 1-3k tokens with short replies;
- every 8 main turns a subagent runs 4 turns of its own, then the main agent
  resumes;
- at main turns 12 and 20 the last request is regenerated, then the last user
  message is edited (a branch off the shared prefix);
- after main turn 30 a second main conversation runs 10 turns, then the first
  resumes for 5 more.

Usage: agent_sim.py PORT SRC_DIR MAX_TOKENS LOG.jsonl
"""
import http.client
import json
import pathlib
import sys
import time

port, src, max_tokens, log = int(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
files = sorted(p for p in src.rglob("*.cpp") if p.stat().st_size > 2000)
text = "".join(f"\n// file: {p.relative_to(src)}\n" + p.read_text(errors="replace") for p in files)
hdrs = sorted(p for p in src.rglob("*.hpp") if p.stat().st_size > 2000)
header_text = "".join(p.read_text(errors="replace") for p in hdrs)

conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
conn.request("GET", "/v1/models")
model = json.loads(conn.getresponse().read())["data"][0]["id"]
conn.close()

# ~6k tokens shared by every conversation: instructions plus "tool" docs.
SYSTEM = ("You are a coding agent working in a C++ repository. Use the tools "
          "below; answer each tool result in at most two sentences, then name "
          "the next file to read.\n\n## Tools\n" + header_text[:21000])


class Text:
    """Hands out consecutive slices of the corpus, from a starting offset."""
    def __init__(self, start):
        self.pos = start

    def take(self, chars):
        out = text[self.pos:self.pos + chars]
        self.pos += chars
        return out


# Tool results vary in size: 1-3k tokens (3.5 chars per token).
SIZES = [3500, 10500, 7000, 5250, 8750, 3500, 7000, 10500]


def tool_result(src, i):
    return ("Tool result (read_file):\n```cpp\n" + src.take(SIZES[i % len(SIZES)]) + "\n```")


def send(label, messages):
    body = json.dumps({"model": model, "messages": messages, "stream": True,
                       "max_tokens": max_tokens, "seed": 1234,
                       "stream_options": {"include_usage": True}})
    t = time.time()
    first = None
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
            piece = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content") or ""
            if piece and first is None:
                first = time.time() - t
            reply.append((ch.get("delta") or {}).get("content") or "")
        usage = ev.get("usage") or usage
    conn.close()
    u = usage or {}
    rec = {"label": label, "status": resp.status, "error": error, "usage": usage,
           "ttft": round(first, 2) if first is not None else None,
           "seconds": round(time.time() - t, 1)}
    print(f"{label:16s} {resp.status} prompt {u.get('prompt_tokens')} cached {u.get('cached_tokens')} "
          f"ttft {rec['ttft']}s {rec['seconds']}s" + (f" ERROR {json.dumps(error)[:200]}" if error else ""),
          flush=True)
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return "".join(reply) or "(no reply)"


class Conversation:
    def __init__(self, name, task, offset):
        self.name, self.src, self.turn = name, Text(offset), 0
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": task}]
        self.pending = None

    def step(self, label=None):
        if self.turn:
            self.messages.append({"role": "user", "content": tool_result(self.src, self.turn)})
        reply = send(label or f"{self.name}.{self.turn}", self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        self.turn += 1

    def regenerate(self):
        # The client drops the last reply and sends the same request again.
        self.messages.pop()
        reply = send(f"{self.name}.regen", self.messages)
        self.messages.append({"role": "assistant", "content": reply})

    def edit(self):
        # The user rewrites the last message: a branch before it.
        self.messages.pop()
        last = self.messages.pop()
        self.messages.append({"role": "user", "content": last["content"][:len(last["content"]) // 2] +
                              "\n(Edited: focus only on error handling.)"})
        reply = send(f"{self.name}.edit", self.messages)
        self.messages.append({"role": "assistant", "content": reply})


started = time.time()
main = Conversation("main", "Task: find where the server handles cache restores and explain the flow.", 0)
subagents = 0
for _ in range(31):
    main.step()
    if main.turn in (12, 20):
        main.regenerate()
        main.edit()
    if main.turn % 8 == 0:
        subagents += 1
        sub = Conversation(f"sub{subagents}", f"Subtask {subagents}: summarize the error paths in these files.",
                           2_000_000 + subagents * 200_000)
        for _ in range(4):
            sub.step()
other = Conversation("main2", "Task: review the build scripts for portability problems.", 1_200_000)
for _ in range(10):
    other.step()
for _ in range(5):
    main.step()
print(f"DONE {time.time() - started:.0f}s", flush=True)
