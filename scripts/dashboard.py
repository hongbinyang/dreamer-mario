"""Launch TensorBoard for one or more named runs.

    python scripts/dashboard.py --name flag-run
    python scripts/dashboard.py --name flag-run --name sparse-ablation   # compare side by side
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", required=True,
                         help="run name under runs/; repeat to compare multiple runs")
    parser.add_argument("--logdir", default="runs", help="parent directory runs live under")
    args = parser.parse_args()

    root = pathlib.Path(args.logdir)
    missing = [n for n in args.name if not (root / n).exists()]
    if missing:
        raise SystemExit(f"no such run(s) under {root}/: {', '.join(missing)}")

    if len(args.name) == 1:
        subprocess.run(["tensorboard", "--logdir", str(root / args.name[0])])
    else:
        spec = ",".join(f"{n}:{root / n}" for n in args.name)
        subprocess.run(["tensorboard", f"--logdir_spec={spec}"])


if __name__ == "__main__":
    main()
