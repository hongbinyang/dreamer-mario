"""Super Mario Bros environment for Dreamer.

Wraps gym-super-mario-bros (old Gym API) into a minimal, torch-free
interface:

    obs = env.reset()                          # uint8 (size, size, C)
    obs, reward, done, info = env.step(a)      # a: int action index

Handles: old/new gym API differences, action sets, frame skip with
max-pooling, resizing to 64x64, optional grayscale, optional sparse reward
(flag-only), and per-episode stats in `info`.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python is required: pip install opencv-python") from e


ACTION_SETS = {
    # (name) -> list of NES button combos, passed to JoypadSpace.
    "right_only": [["NOOP"], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"]],
    "simple": [["NOOP"], ["right"], ["right", "A"], ["right", "B"],
               ["right", "A", "B"], ["A"], ["left"]],
}


class MarioEnv:
    def __init__(self, env_id: str = "SuperMarioBros-1-1-v0",
                 action_set: str = "right_only", frame_skip: int = 4,
                 size: int = 64, grayscale: bool = False,
                 sparse_reward: bool = False, seed: int = 0):
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace

        self._raw = gym_super_mario_bros.make(env_id)
        self._env = JoypadSpace(self._raw, ACTION_SETS[action_set])
        self.num_actions = len(ACTION_SETS[action_set])
        self.frame_skip = frame_skip
        self.size = size
        self.grayscale = grayscale
        self.sparse_reward = sparse_reward
        self._seed = seed
        self._episode_return = 0.0
        self._episode_len = 0

    # ------------------------------------------------------------------ api
    @property
    def obs_shape(self) -> tuple:
        return (self.size, self.size, 1 if self.grayscale else 3)

    def reset(self) -> np.ndarray:
        out = self._env.reset()
        # Old gym returns obs; gym>=0.26 returns (obs, info).
        obs = out[0] if isinstance(out, tuple) else out
        self._episode_return = 0.0
        self._episode_len = 0
        return self._process(obs)

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}
        frames = []
        for _ in range(self.frame_skip):
            out = self._env.step(action)
            if len(out) == 5:  # gym>=0.26: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:              # old gym: obs, reward, done, info
                obs, reward, done, info = out
                done = bool(done)
            total_reward += float(reward)
            frames.append(obs)
            if done:
                break
        # Max over the last two frames (standard anti-flicker for NES/Atari).
        frame = np.maximum(frames[-1], frames[-2]) if len(frames) >= 2 else frames[-1]

        if self.sparse_reward:
            total_reward = 100.0 if info.get("flag_get", False) else 0.0

        self._episode_return += total_reward
        self._episode_len += 1
        info = dict(info)
        info["episode_return"] = self._episode_return
        info["episode_len"] = self._episode_len
        return self._process(frame), total_reward, done, info

    def render_frame(self) -> np.ndarray:
        """Full-resolution RGB frame for videos."""
        return np.asarray(self._raw.screen, dtype=np.uint8).copy()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------- internal
    def _process(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.uint8)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = obs[..., None]
        return obs


def make_env(cfg, seed: int = 0) -> MarioEnv:
    e = cfg.env
    return MarioEnv(
        env_id=e.id,
        action_set=e.action_set,
        frame_skip=e.frame_skip,
        size=e.size,
        grayscale=e.grayscale,
        sparse_reward=e.sparse_reward,
        seed=seed,
    )
