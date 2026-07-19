"""Generate the showcase video: Dreamer dreaming Mario.

Collects a real episode with the trained policy, then produces a
side-by-side video: [ real | imagined | error ]. The imagined half is an
open-loop rollout: the model sees `--context` real frames, then predicts
the next `--horizon` frames purely in latent space (replaying the same
actions), decoded back to pixels.

    python scripts/dream.py --name trial --out dream.mp4
    python scripts/dream.py --ckpt runs/trial/ckpt.pt --out dream.mp4  # equivalent
    python scripts/dream.py --name trial --set run.seed=1               # override on top of the checkpoint's config
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import imageio
import numpy as np
import torch

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
    parser.add_argument("--out", default="dream.mp4")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--fps", type=float, default=None,
                         help="video playback fps; defaults to real-time (60 / env.frame_skip)")
    parser.add_argument("--device", default=None,
                         help="auto (default) | cpu | cuda[:N] | mps | tpu; overrides run.device")
    args = parser.parse_args()
    ckpt_path = args.ckpt or str(pathlib.Path(args.logdir) / args.name / "ckpt.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict_to_ns(apply_overrides(ckpt["cfg"], args.set))
    if args.device:
        cfg.run.device = args.device
    device = pick_device(cfg.run.device)
    env = make_env(cfg)
    agent = DreamerAgent(env.num_actions, cfg, device)
    agent.load(ckpt_path)

    # Collect one episode with the trained policy.
    obs_list, act_list = [], []
    obs = env.reset()
    state = agent.init_policy_state()
    is_first, done = True, False
    obs_list.append(obs)
    act_list.append(0)
    while not done and len(obs_list) < args.context + args.horizon + 1:
        action, state = agent.act(obs, state, is_first, greedy=True)
        is_first = False
        obs, _, done, _ = env.step(action)
        obs_list.append(obs)
        act_list.append(action)
    env.close()

    T = len(obs_list)
    horizon = min(args.horizon, T - args.context)
    if horizon <= 0:
        raise SystemExit("episode too short; lower --context")

    batch = {
        "obs": torch.as_tensor(np.stack(obs_list)[None], device=device),
        "action": torch.nn.functional.one_hot(
            torch.as_tensor(act_list, device=device)[None],
            env.num_actions).float(),
        "is_first": torch.zeros(1, T, device=device),
    }
    batch["is_first"][0, 0] = 1.0

    video = agent.wm.video_pred(batch, context=args.context, horizon=horizon)
    video = video.numpy()  # (T, H, 3W, C)
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    if args.upscale > 1:
        video = np.repeat(np.repeat(video, args.upscale, axis=1), args.upscale, axis=2)

    fps = args.fps if args.fps is not None else 60.0 / cfg.env.frame_skip
    imageio.mimsave(args.out, list(video), fps=fps)
    print(f"wrote {args.out} at {fps:.1f} fps: {args.context} context frames, "
          f"then {horizon} imagined frames. Layout: [truth | model | error].")


if __name__ == "__main__":
    main()
