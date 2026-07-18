"""List or delete named training runs under runs/.

    python scripts/cleanup.py --list
    python scripts/cleanup.py --name old-run --yes
    python scripts/cleanup.py --name old-run-1 --name old-run-2 --yes

Deletion is a dry-run unless --yes is passed: without it, this only prints
what would be deleted (checkpoints run 100-200+ MB each, so it's easy to
nuke real training time by mistake).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import torch


def _dir_size_mb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _step(run_dir: pathlib.Path):
    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        return "-"
    return torch.load(ckpt, map_location="cpu").get("step", "-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list all runs with step and size")
    parser.add_argument("--name", action="append", default=[],
                         help="run name(s) to delete; repeat for multiple")
    parser.add_argument("--yes", action="store_true", help="actually delete (otherwise dry-run)")
    parser.add_argument("--logdir", default="runs", help="parent directory runs live under")
    args = parser.parse_args()

    root = pathlib.Path(args.logdir)
    root.mkdir(exist_ok=True)

    if args.list or not args.name:
        runs = sorted(d for d in root.iterdir() if d.is_dir())
        if not runs:
            print(f"no runs under {root}/")
            return
        for d in runs:
            print(f"{d.name:30s} step {str(_step(d)):>10s}  {_dir_size_mb(d):8.1f} MB")
        return

    for name in args.name:
        target = root / name
        if not target.exists():
            print(f"skip: no such run '{name}'")
            continue
        size_mb = _dir_size_mb(target)
        if args.yes:
            shutil.rmtree(target)
            print(f"deleted {target} ({size_mb:.1f} MB)")
        else:
            print(f"would delete {target} ({size_mb:.1f} MB) — pass --yes to actually delete")


if __name__ == "__main__":
    main()
