#!/usr/bin/env python3
"""Copy the host implementations behind the public headers into tests/host/.

The test executables link only gufo_ext_runtime, whose public headers declare
GGUF reading, sampling, JSON constraints, dequantization, hashing, image
decoding, weight upload and the cache types without defining them; the host
links those definitions into `gufo` only. This copies the defining host
sources verbatim, each with a provenance line, and rewrites their
repository-relative includes to the public forwarders (or to a copied private
header when the host publishes none). Only the test targets compile them.

  tools/sync_test_host.py --host /path/to/gufo

Rerun after moving to a new host revision; `--check` reports a stale copy.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "host"

# Host sources the tests need, by the public declarations they define.
CPU_SOURCES = [
    "src/cache/checkpoint.cpp",
    "src/cache/completion.cpp",
    "src/cache/identity.cpp",
    "src/cache/ledger.cpp",
    "src/cache/slot.cpp",
    "src/core/crypto/sha256.cpp",
    "src/core/gguf_reader.cpp",
    "src/core/image.cpp",
    "src/core/json_constraint.cpp",
    "src/core/json_schema_lexeme.cpp",
    "src/core/json_schema_regex.cpp",
    "src/core/quant/ggml_dequant.cpp",
    "src/core/sampling.cpp",
    "src/core/sampling_logit_row.cpp",
    "src/core/hip/hipblaslt_plan_database.cpp",
]
HIP_SOURCES = [
    "src/core/hip/host_allocations.cpp",
    "src/core/hip/transfer_pool.cpp",
    "src/core/hip/weight_upload.cpp",
]
# Files the sources include that the host does not publish. Their
# definitions, where they have any, are in gufo_platform, which
# gufo_ext_runtime links.
PRIVATE = [
    "src/core/json_schema_regex.hpp",
    "src/core/platform/page_pin.hpp",
    "src/core/platform/socket.hpp",
    "src/core/quant/ggml_iq_grids.inc",
    "src/core/sampling_logit_row.hpp",
]

INCLUDE = re.compile(r'^(\s*#\s*include\s+)"(src/[^"]+)"(.*)$')


def git(host: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(host), *args],
                                   text=True).strip()


def closure(host: Path, header: str) -> set[str]:
    seen = {header}
    frontier = [header]
    while frontier:
        source = host / frontier.pop()
        if not source.is_file():
            continue
        for line in source.read_text().splitlines():
            if (m := INCLUDE.match(line)) and m.group(2) not in seen:
                seen.add(m.group(2))
                frontier.append(m.group(2))
    return seen


def forwarders(host: Path) -> dict[str, str]:
    """Maps each host header a public forwarder reaches to the forwarder
    with the smallest include closure that reaches it, without HIP headers
    for a host-only one."""
    reach: dict[str, tuple[int, str]] = {}
    for path in sorted((host / "include" / "gufo" / "ext").rglob("*.hpp")):
        public = path.relative_to(host / "include").as_posix()
        targets: set[str] = set()
        for line in path.read_text().splitlines():
            if m := INCLUDE.match(line):
                targets |= closure(host, m.group(2))
        device = any(t.startswith("src/core/hip/") for t in targets)
        for header in targets:
            # A host-only header never takes a forwarder that needs HIP.
            if device and not header.startswith("src/core/hip/"):
                continue
            if header not in reach or len(targets) < reach[header][0]:
                reach[header] = (len(targets), public)
    return {header: public for header, (_, public) in reach.items()}


def local_name(header: str) -> str:
    return Path(header).name


def rewrite(text: str, reach: dict[str, str], origin: str) -> str:
    lines = []
    for line in text.splitlines():
        m = INCLUDE.match(line)
        if m:
            header = m.group(2)
            if header in PRIVATE:
                line = f'{m.group(1)}"tests/host/{local_name(header)}"{m.group(3)}'
            elif header in reach:
                line = f"{m.group(1)}<{reach[header]}>{m.group(3)}"
            else:
                sys.exit(f"{origin}: no public path to {header}")
        lines.append(line)
    return "\n".join(lines) + "\n"


def render(host: Path, revision: str, reach: dict[str, str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for source in CPU_SOURCES + HIP_SOURCES + PRIVATE:
        blob = git(host, "rev-parse", f"{revision}:{source}")
        text = git(host, "show", f"{revision}:{source}") + "\n"
        banner = (f"// Copied verbatim from gufo {source} at {revision} "
                  f"(blob {blob[:12]}); only the includes are rewritten.\n"
                  "// Regenerate with tools/sync_test_host.py; do not edit.\n")
        files[local_name(source)] = banner + rewrite(text, reach, source)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    revision = git(args.host, "rev-parse", args.revision)
    files = render(args.host, revision, forwarders(args.host))
    names = {local_name(s) for s in CPU_SOURCES + HIP_SOURCES + PRIVATE}
    if len(names) != len(CPU_SOURCES + HIP_SOURCES + PRIVATE):
        sys.exit("two host files share a name")
    if args.check:
        stale = [n for n, t in files.items()
                 if not (OUT / n).is_file() or (OUT / n).read_text() != t]
        for name in stale:
            print(f"stale: tests/host/{name}")
        return 1 if stale else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        if old.name not in files and old.name != "README":
            old.unlink()
    for name, text in files.items():
        (OUT / name).write_text(text)
    print(f"{len(files)} files from {revision[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
