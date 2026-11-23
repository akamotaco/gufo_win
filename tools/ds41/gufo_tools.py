"""Puts Gufo's expert-stream trace tools (tools/expert_stream/) on sys.path.

GUFO_SRC names a Gufo checkout at the pinned revision or later; the trace
reader and the cache model live there since the expert store moved into Gufo.
"""
import os
from pathlib import Path
import sys


def use_expert_stream_tools():
    root = os.environ.get("GUFO_SRC")
    if not root:
        raise SystemExit("set GUFO_SRC to a Gufo checkout: its tools/expert_stream/ reads the trace")
    path = Path(root).expanduser().resolve() / "tools" / "expert_stream"
    if not (path / "expert_trace.py").is_file():
        raise SystemExit(f"{path}/expert_trace.py not found: GUFO_SRC must be a Gufo checkout with the expert store")
    sys.path.insert(0, str(path))
