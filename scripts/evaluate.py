"""Evaluate a trained agent.

    python scripts/evaluate.py --ckpt runs/<name>/ckpt.pt --episodes 5 --video eval.mp4
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.agent import DreamerAgent  # noqa: E402
from dreamer.config import load_config, pick_device  # noqa: E402
from dreamer.envs.mario import make_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video", default=None, help="save first episode to mp4")
    parser.add_argument("--device", default=None,
                         help="auto (default) | cpu | cuda[:N] | mps | tpu; overrides run.device")
    args = parser.parse_args()

    import torch
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in ckpt:  # exact config the model was trained with
        from dreamer.config import dict_to_ns
        cfg = dict_to_ns(ckpt["cfg"])
    else:
        cfg = load_config(["--config", args.config])
    if args.device:
        cfg.run.device = args.device
    device = pick_device(cfg.run.device)
    env = make_env(cfg)
    agent = DreamerAgent(env.num_actions, cfg, device)
    agent.load(args.ckpt)

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
        imageio.mimsave(args.video, frames, fps=30)
        print(f"wrote {args.video}")
    env.close()


if __name__ == "__main__":
    main()
