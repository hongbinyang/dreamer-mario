"""Fast end-to-end smoke test (~1-2 min on CPU/MPS).

    python scripts/smoke_test.py            # tiny model, random env data, 3 train steps
    python scripts/smoke_test.py --no-env   # skip the emulator, use synthetic frames

Checks: env wrapper, replay buffer, world-model loss, imagination,
actor-critic loss, one full agent.train_step, and video_pred shapes.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.agent import DreamerAgent  # noqa: E402
from dreamer.config import load_config, pick_device  # noqa: E402
from dreamer.replay import ReplayBuffer  # noqa: E402


def tiny_overrides():
    return [
        "--set", "model.cnn_depth=8",
        "--set", "model.deter=64",
        "--set", "model.stoch=8",
        "--set", "model.classes=8",
        "--set", "model.hidden=64",
        "--set", "replay.batch_size=4",
        "--set", "replay.seq_len=16",
        "--set", "train.imag_horizon=5",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None,
                         help="auto (default) | cpu | cuda[:N] | mps | tpu; overrides run.device")
    args = parser.parse_args()

    cfg = load_config(["--config", args.config] + tiny_overrides())
    if args.device:
        cfg.run.device = args.device
    device = pick_device(cfg.run.device)
    print(f"device: {device}")

    num_actions = 5
    obs_shape = (cfg.env.size, cfg.env.size, 1 if cfg.env.grayscale else 3)

    replay = ReplayBuffer(2000, obs_shape, num_actions,
                          cfg.replay.seq_len, cfg.replay.batch_size)

    if args.no_env:
        print("filling replay with synthetic frames...")
        rng = np.random.default_rng(0)
        for i in range(300):
            obs = rng.integers(0, 255, obs_shape, dtype=np.uint8)
            replay.add(obs, int(rng.integers(num_actions)), float(rng.normal()),
                       is_first=(i % 100 == 0), is_terminal=(i % 100 == 99))
    else:
        print("running random agent in the real env...")
        from dreamer.envs.mario import make_env
        env = make_env(cfg)
        num_actions = env.num_actions
        obs = env.reset()
        replay.add(obs, 0, 0.0, True, False)
        for _ in range(300):
            a = int(np.random.randint(num_actions))
            obs, r, done, info = env.step(a)
            replay.add(obs, a, r, False, done)
            if done:
                obs = env.reset()
                replay.add(obs, 0, 0.0, True, False)
        env.close()
        print(f"env ok: obs {obs.shape} {obs.dtype}, x_pos {info.get('x_pos')}")

    agent = DreamerAgent(num_actions, cfg, device)
    n_params = sum(p.numel() for p in agent.wm.parameters())
    print(f"world model params: {n_params/1e6:.2f}M (tiny test config)")

    for i in range(3):
        batch = replay.sample()
        metrics = agent.train_step(batch)
        print(f"train step {i}: wm/loss {metrics['wm/loss']:.2f}, "
              f"ac/actor_loss {metrics['ac/actor_loss']:.3f}, "
              f"ac/critic_loss {metrics['ac/critic_loss']:.3f}")

    batch = {k: torch.as_tensor(v, device=device) for k, v in replay.sample().items()}
    video = agent.wm.video_pred(batch, context=3, horizon=8)
    print(f"video_pred ok: {tuple(video.shape)} (T, H, 3W, C)")

    # Acting path.
    obs = np.zeros(obs_shape, dtype=np.uint8)
    state = agent.init_policy_state()
    action, state = agent.act(obs, state, is_first=True)
    action, state = agent.act(obs, state, is_first=False)
    print(f"act ok: sampled action {action}")
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
