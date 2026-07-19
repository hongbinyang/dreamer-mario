"""Evaluate a trained agent.

    python scripts/evaluate.py --name trial --episodes 5 --video eval.mp4
    python scripts/evaluate.py --ckpt runs/trial/ckpt.pt --episodes 5   # equivalent
    python scripts/evaluate.py --name trial --set run.seed=1             # override on top of the checkpoint's config
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.agent import DreamerAgent  # noqa: E402
from dreamer.config import apply_overrides, dict_to_ns, pick_device  # noqa: E402
from dreamer.envs.mario import make_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--name", help="run name under --logdir; resolves to <logdir>/<name>/ckpt.pt")
    which.add_argument("--ckpt", help="explicit checkpoint path")
    parser.add_argument("--logdir", default="runs", help="parent directory runs live under (with --name)")
    parser.add_argument("--set", action="append", default=[],
                         help="dotted.key=value override applied on top of the checkpoint's config")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video", default=None, help="save first episode to mp4")
    parser.add_argument("--fps", type=float, default=None,
                         help="video playback fps; defaults to real-time (60 / env.frame_skip)")
    parser.add_argument("--device", default=None,
                         help="auto (default) | cpu | cuda[:N] | mps | tpu; overrides run.device")
    args = parser.parse_args()
    ckpt_path = args.ckpt or str(pathlib.Path(args.logdir) / args.name / "ckpt.pt")

    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict_to_ns(apply_overrides(ckpt["cfg"], args.set))
    if args.device:
        cfg.run.device = args.device
    device = pick_device(cfg.run.device)
    env = make_env(cfg)
    agent = DreamerAgent(env.num_actions, cfg, device)
    agent.load(ckpt_path)

    returns, xs, flags = [], [], 0
    frames = []
    for ep in range(args.episodes):
        obs = env.reset()
        state = agent.init_policy_state()
        is_first, done = True, False
        info = {}
        while not done:
            action, state = agent.act(obs, state, is_first, greedy=True)
            is_first = False
            obs, _, done, info = env.step(action)
            if ep == 0 and args.video:
                frames.append(env.render_frame())
        returns.append(info.get("episode_return", 0.0))
        xs.append(int(info.get("x_pos", 0)))
        flags += int(bool(info.get("flag_get", False)))
        print(f"episode {ep}: return {returns[-1]:.1f}, x_pos {xs[-1]}, "
              f"flag {bool(info.get('flag_get', False))}")

    print(f"\nmean return {np.mean(returns):.1f} | mean x {np.mean(xs):.0f} "
          f"| flag rate {flags}/{args.episodes}")
    if args.video and frames:
        # Each frame is one env.step(), which already advances env.frame_skip
        # real NES frames -- so real-time playback is 60 / frame_skip fps,
        # not an arbitrary constant.
        fps = args.fps if args.fps is not None else 60.0 / cfg.env.frame_skip
        imageio.mimsave(args.video, frames, fps=fps)
        print(f"wrote {args.video} at {fps:.1f} fps")
    env.close()


if __name__ == "__main__":
    main()
