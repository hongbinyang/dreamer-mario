"""Uniform sequence replay buffer.

Stores a flat stream of steps in preallocated numpy arrays (obs as uint8 to
keep memory low: 200k steps of 64x64 RGB is ~2.3 GB). Sequences may cross
episode boundaries; the RSSM resets its state wherever `is_first` is set,
which is how the official DreamerV3 replay works too.
"""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, num_actions: int,
                 seq_len: int, batch_size: int, seed: int = 0):
        self.capacity = capacity
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.is_first = np.zeros((capacity,), dtype=bool)
        self.is_terminal = np.zeros((capacity,), dtype=bool)
        self.pos = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: np.ndarray, action: int, reward: float,
            is_first: bool, is_terminal: bool):
        """action = the action that led INTO this obs (0 on is_first)."""
        p = self.pos
        self.obs[p] = obs
        self.action[p] = action
        self.reward[p] = reward
        self.is_first[p] = is_first
        self.is_terminal[p] = is_terminal
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def _valid_start(self, idx: int) -> bool:
        """Reject sequences that cross the write pointer (the 'seam' where the
        newest data overwrote the oldest)."""
        if not self.full:
            return idx + self.seq_len <= self.pos
        # Ring buffer: the window [idx, idx+L) contains the seam at self.pos
        # iff the ring-distance from idx to pos is less than L.
        return (self.pos - idx) % self.capacity >= self.seq_len

    def sample(self) -> dict:
        assert len(self) > self.seq_len + 1, "not enough data in replay buffer"
        idxs = []
        while len(idxs) < self.batch_size:
            idx = int(self.rng.integers(0, len(self)))
            if self._valid_start(idx):
                idxs.append(idx)
        seqs = {k: [] for k in ("obs", "action", "reward", "is_first", "is_terminal")}
        for idx in idxs:
            sl = (np.arange(idx, idx + self.seq_len) % self.capacity)
            seqs["obs"].append(self.obs[sl])
            seqs["action"].append(self.action[sl])
            seqs["reward"].append(self.reward[sl])
            seqs["is_first"].append(self.is_first[sl])
            seqs["is_terminal"].append(self.is_terminal[sl])
        batch = {k: np.stack(v) for k, v in seqs.items()}
        # One-hot actions for the RSSM.
        onehot = np.zeros((*batch["action"].shape, self.num_actions), dtype=np.float32)
        np.put_along_axis(onehot, batch["action"][..., None], 1.0, axis=-1)
        batch["action"] = onehot
        batch["reward"] = batch["reward"].astype(np.float32)
        batch["is_first"] = batch["is_first"].astype(np.float32)
        batch["is_terminal"] = batch["is_terminal"].astype(np.float32)
        # Mark the first step of every sampled sequence as is_first so the
        # RSSM starts from a fresh state (we have no context before it).
        batch["is_first"][:, 0] = 1.0
        return batch
