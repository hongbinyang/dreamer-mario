"""Scan runs/ and runs_ppo/ for status info to display in the web GUI.

Read-only: never trains, evaluates, or deletes anything. Reimplements the
same small approach scripts/cleanup.py already uses (_dir_size_mb / step
lookup) rather than importing from it -- scripts/ are thin CLI entry
points, not a library other packages should import from.
"""
from __future__ import annotations

import pathlib

from . import jobs


def _dir_size_mb(path: pathlib.Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _dreamer_step(run_dir: pathlib.Path):
    ckpt = run_dir / "ckpt.pt"
    if not ckpt.exists():
        return None
    import torch
    try:
        return torch.load(ckpt, map_location="cpu").get("step")
    except Exception:
        # A single unreadable/corrupt checkpoint shouldn't take down the
        # whole runs list -- just show this one run's step as unknown.
        return None


def _ppo_event_dir(run_dir: pathlib.Path) -> pathlib.Path | None:
    """SB3 writes tfevents into a PPO_<n> subfolder, incrementing n on every
    .learn() call against the same tensorboard_log path -- pick the newest."""
    candidates = sorted(run_dir.glob("PPO_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _latest_scalar(event_dir: pathlib.Path, tag: str):
    """Latest value of a TensorBoard scalar tag under event_dir, or None if
    missing (e.g. no video/log step has happened yet) or unreadable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    try:
        ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 1})
        ea.Reload()
        events = ea.Scalars(tag)
        return events[-1].value if events else None
    except Exception:
        return None


def _run_info(d: pathlib.Path, kind: str) -> dict:
    event_dir = d if kind == "dreamer" else _ppo_event_dir(d)
    best_x = _latest_scalar(event_dir, "episode/best_x") if event_dir else None
    flags = _latest_scalar(event_dir, "episode/flags") if event_dir else None
    return {
        "name": d.name,
        "kind": kind,
        "step": _dreamer_step(d) if kind == "dreamer" else None,
        "best_x": int(best_x) if best_x is not None else None,
        "flags": int(flags) if flags is not None else None,
        "size_mb": round(_dir_size_mb(d), 1),
        "mtime": d.stat().st_mtime,
        "running": jobs.is_run_active(d.name),
    }


def scan(logdir: str = "runs", logdir_ppo: str = "runs_ppo") -> list[dict]:
    """All runs under both directories, newest first."""
    out = []
    root = pathlib.Path(logdir)
    if root.exists():
        out += [_run_info(d, "dreamer") for d in root.iterdir() if d.is_dir()]
    root_ppo = pathlib.Path(logdir_ppo)
    if root_ppo.exists():
        out += [_run_info(d, "ppo") for d in root_ppo.iterdir() if d.is_dir()]
    return sorted(out, key=lambda r: r["mtime"], reverse=True)
