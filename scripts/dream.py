"""Generate the showcase video: Dreamer dreaming Mario.

Collects a real episode with the trained policy, then produces a
side-by-side video: [ real | imagined | error ]. The imagined half is an
open-loop rollout: the model sees `--context` real frames, then predicts
the next `--horizon` frames purely in latent space (replaying the same
actions), decoded back to pixels.

    python scripts/dream.py --ckpt runs/<name>/ckpt.pt --out dream.mp4
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
from dreamer.config import load_config, pick_device  # noqa: E402
from dreamer.envs.mario import make_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="dream.mp4")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=56)
    parser.add_argument("--upscale", type=int, default=4)
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

    imageio.mimsave(args.out, list(video), fps=10)
    print(f"wrote {args.out}: {args.context} context frames, "
          f"then {horizon} imagined frames. Layout: [truth | model | error].")


if __name__ == "__main__":
    main()
