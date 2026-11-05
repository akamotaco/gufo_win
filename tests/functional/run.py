#!/usr/bin/env python3
"""Run Gufo's functional suites against a Gemma 4 server.

The host harness (`tests/functional/` of a Gufo checkout) has no `gemma4`
sampling preset and no Gemma expectations. This copies it into a fresh
directory beside the report, applies host-harness.patch (the preset, the
template facts in templates.py and the expectations that read them) and runs
the copy's run.py with the remaining arguments:

  python3 run.py --host ~/projects/gufo-ext -- \\
      --output out/31b-mtp --record-baseline --sampling-preset gemma4 \\
      --suite tools -- \\
      build/release/gufo serve llm --model ... --speculative mtp --mtp-model ...

Everything after the first `--` goes to the host run.py unchanged. The host
checkout is read only; a patch that does not apply cleanly is an error.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PATCH = HERE / "host-harness.patch"


def git(host, *args):
    return subprocess.check_output(["git", "-C", str(host), *args], text=True).strip()


def overlay(host, root):
    """A host checkout's harness, tools and flake.lock with the patch applied."""
    functional = root / "tests" / "functional"
    shutil.copytree(host / "tests" / "functional", functional,
                    ignore=shutil.ignore_patterns("__pycache__"))
    # run.py hashes ../../flake.lock and imports helpers from ../../tools.
    shutil.copy2(host / "flake.lock", root / "flake.lock")
    (root / "tools").symlink_to(host / "tools", target_is_directory=True)
    subprocess.run(["patch", "-p1", "--forward", "--no-backup-if-mismatch",
                    "--fuzz=0", "-s", "-i", str(PATCH)], cwd=functional, check=True)
    return functional


def main():
    if "--" not in sys.argv:
        sys.exit("usage: run.py --host GUFO_CHECKOUT -- HOST_RUN_ARGS...")
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", type=Path, required=True,
                        help="Gufo checkout whose tests/functional is run")
    args = parser.parse_args(sys.argv[1:split])
    forwarded = sys.argv[split + 1:]
    host = args.host.resolve()
    output = None
    for index, arg in enumerate(forwarded):
        if arg == "--output" and index + 1 < len(forwarded):
            output = Path(forwarded[index + 1]).resolve()
        elif arg.startswith("--output="):
            output = Path(arg.split("=", 1)[1]).resolve()
    if output is None:
        sys.exit("pass --output to the host run.py")
    # The host runner refuses an existing report directory, so the harness
    # copy sits beside it.
    root = output.with_name(output.name + ".harness")
    if root.exists():
        sys.exit(f"{root} exists; choose a new --output")
    functional = overlay(host, root)
    status = subprocess.run(["git", "-C", str(host), "status", "--porcelain",
                             "--", "tests/functional"], capture_output=True,
                            text=True, check=True).stdout
    (root / "provenance.json").write_text(json.dumps({
        "host": str(host), "host_revision": git(host, "rev-parse", "HEAD"),
        "host_functional_modified": bool(status.strip()),
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return subprocess.call([sys.executable, str(functional / "run.py"), *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
