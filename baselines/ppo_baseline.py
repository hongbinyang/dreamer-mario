"""PPO baseline (stable-baselines3) for the sample-efficiency comparison.

Uses the SAME MarioEnv wrapper (same frame skip, resize, reward), so the
env-frame counts on the x-axis are directly comparable with Dreamer.

    pip install stable-baselines3 gymnasium   # optional, not in requirements
    python baselines/ppo_baseline.py --name trial

stable-baselines3 only supports cpu/cuda (see
stable_baselines3.common.utils.get_device, whose docstring says so
explicitly) -- there is no MPS or TPU path here, unlike the Dreamer scripts.
On an Apple Silicon Mac this always runs on CPU.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.config import dict_to_ns, load_yaml  # noqa: E402
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


def make_metrics_callback():
    """Logs episode/best_x and episode/flags under the exact tag names
    Dreamer's train.py uses (same cumulative-max / cumulative-count
    semantics too), so the two overlay directly in one TensorBoard chart
    instead of eyeballing differently-named curves side by side. SB3 has no
    built-in equivalent -- Monitor only tracks reward/length by default."""
    from stable_baselines3.common.callbacks import BaseCallback

    class MarioMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_x = 0
            self.flags = 0

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    self.best_x = max(self.best_x, int(info.get("x_pos", 0)))
                    self.flags += int(bool(info.get("flag_get", False)))
            self.logger.record("episode/best_x", self.best_x)
            self.logger.record("episode/flags", self.flags)
            return True

    return MarioMetricsCallback()


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                         help="run identifier; output under runs_ppo/<name>/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[],
                         help="dotted.key=value overrides, e.g. --set ppo.ent_coef=0.05")
    parser.add_argument("--device", default="auto",
                         help="auto (default: cuda if available, else cpu) | cpu | cuda[:N] -- "
                              "SB3 has no MPS/TPU support, unlike the Dreamer scripts")
    args = parser.parse_args()

    cfg = dict_to_ns(load_yaml(args.config, args.set))

    logdir = pathlib.Path("runs_ppo") / args.name
    logdir.mkdir(parents=True, exist_ok=True)

    env = Monitor(GymnasiumMario(cfg))
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=str(logdir), device=args.device,
                n_steps=cfg.ppo.n_steps, batch_size=cfg.ppo.batch_size,
                learning_rate=cfg.ppo.learning_rate, ent_coef=cfg.ppo.ent_coef)
    model.learn(total_timesteps=cfg.ppo.total_steps, callback=make_metrics_callback())
    model.save(str(logdir / "model"))
    env.close()


if __name__ == "__main__":
    main()
