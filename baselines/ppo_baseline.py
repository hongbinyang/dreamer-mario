"""PPO baseline (stable-baselines3) for the sample-efficiency comparison.

Uses the SAME MarioEnv wrapper (same frame skip, resize, reward), so the
env-frame counts on the x-axis are directly comparable with Dreamer.

    pip install stable-baselines3   # optional dependency, not in requirements
    python baselines/ppo_baseline.py --steps 1000000
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.config import load_config  # noqa: E402
from dreamer.envs.mario import make_env  # noqa: E402


class GymnasiumMario(gym.Env):
    """Adapts MarioEnv to the Gymnasium API that SB3 expects."""

    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.env = make_env(cfg)
        h, w, c = self.env.obs_shape
        self.observation_space = gym.spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(self.env.num_actions)

    def reset(self, seed=None, options=None):
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(int(action))
        return obs, reward, done, False, info

    def close(self):
        self.env.close()


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=1_000_000)
    args = parser.parse_args()

    cfg = load_config(["--config", args.config])
    env = Monitor(GymnasiumMario(cfg))
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="runs_ppo",
                n_steps=512, batch_size=256, learning_rate=2.5e-4, ent_coef=0.01)
    model.learn(total_timesteps=args.steps)
    model.save("runs_ppo/ppo_mario")
    env.close()


if __name__ == "__main__":
    main()
