"""Launch TensorBoard for one or more named runs.

    python scripts/dashboard.py --name trial
    python scripts/dashboard.py --name trial --name trial-sparse   # compare side by side
    python scripts/dashboard.py --name trial --ppo-name trial      # Dreamer vs its PPO baseline
    python scripts/dashboard.py --name trial --port 6100           # pick a specific port
"""
from __future__ import annotations

import argparse
import pathlib
import signal
import subprocess


def main():
    # A backgrounded launch (as the web GUI always uses) has SIGINT set to
    # SIG_IGN before exec by the shell; Python respects an inherited SIG_IGN
    # and won't otherwise install its own handler. Reset explicitly so Stop
    # actually works -- see scripts/train.py for the full explanation. This
    # script additionally relies on subprocess.run()'s own behavior of
    # kill()-ing its child on any exception (including the KeyboardInterrupt
    # this reset enables) to make sure the tensorboard child dies too, not
    # just this wrapper process -- verified directly, see docs/webui.md.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", default=[],
                         help="Dreamer run name under --logdir; repeat to compare multiple")
    parser.add_argument("--ppo-name", action="append", default=[],
                         help="PPO baseline run name under --ppo-logdir; repeat to compare "
                              "multiple, mixes freely with --name")
    parser.add_argument("--logdir", default="runs", help="parent directory Dreamer runs live under")
    parser.add_argument("--ppo-logdir", default="runs_ppo",
                         help="parent directory PPO baseline runs live under")
    parser.add_argument("--port", type=int, default=None,
                         help="TensorBoard port; omit for TensorBoard's own default (6006)")
    args = parser.parse_args()

    if not args.name and not args.ppo_name:
        raise SystemExit("pass at least one --name or --ppo-name")

    root = pathlib.Path(args.logdir)
    ppo_root = pathlib.Path(args.ppo_logdir)
    # PPO entries get a "ppo-" label prefix so a same-named Dreamer/PPO pair
    # (the common case, e.g. --name trial --ppo-name trial) don't collide
    # under one ambiguous "trial" label in TensorBoard's run list.
    entries = [(n, root / n) for n in args.name] + \
              [(f"ppo-{n}", ppo_root / n) for n in args.ppo_name]
    missing = [str(p) for _, p in entries if not p.exists()]
    if missing:
        raise SystemExit(f"no such run(s): {', '.join(missing)}")

    cmd = ["tensorboard"]
    if args.port is not None:
        cmd += ["--port", str(args.port)]
    if len(entries) == 1:
        cmd += ["--logdir", str(entries[0][1])]
    else:
        spec = ",".join(f"{label}:{path}" for label, path in entries)
        cmd += [f"--logdir_spec={spec}"]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
