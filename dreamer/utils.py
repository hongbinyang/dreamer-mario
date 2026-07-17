"""Core math utilities from the Dreamer papers.

- symlog / symexp:        DreamerV3 (robust scale-invariant regression targets)
- TwoHotDistSymlog:       DreamerV3 (categorical regression for reward/value)
- lambda_return:          Dreamer v1 (TD-lambda targets computed in imagination)
- Moments:                DreamerV3 (percentile-based return normalization)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHotDistSymlog:
    """Categorical distribution over exponentially-spaced bins (DreamerV3).

    The network outputs `num_bins` logits over bins spaced linearly in
    symlog-space between low and high. Scalar targets are encoded as a
    "two-hot" vector (weight split between the two nearest bins), and the
    loss is cross-entropy. The mean decodes back through symexp.
    This lets one fixed hyperparameter set handle rewards of any scale.
    """

    def __init__(self, logits: torch.Tensor, low: float = -20.0, high: float = 20.0):
        self.logits = logits  # (..., num_bins)
        self.num_bins = logits.shape[-1]
        self.bins = torch.linspace(low, high, self.num_bins, device=logits.device)

    @property
    def mean(self) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """x: raw-scale scalar tensor broadcastable to logits[..., 0]."""
        x = symlog(x)
        x = torch.clamp(x, self.bins[0], self.bins[-1])
        # Find surrounding bins.
        above = torch.searchsorted(self.bins, x, right=True)
        above = torch.clamp(above, 1, self.num_bins - 1)
        below = above - 1
        lo, hi = self.bins[below], self.bins[above]
        weight_hi = (x - lo) / torch.clamp(hi - lo, min=1e-8)
        weight_lo = 1.0 - weight_hi
        target = torch.zeros_like(self.logits)
        target.scatter_(-1, below.unsqueeze(-1), weight_lo.unsqueeze(-1))
        target.scatter_add_(-1, above.unsqueeze(-1), weight_hi.unsqueeze(-1))
        log_pred = F.log_softmax(self.logits, dim=-1)
        return (target * log_pred).sum(-1)


def lambda_return(
    reward: torch.Tensor,      # (H, B)   r_1 .. r_H
    value: torch.Tensor,       # (H, B)   v_1 .. v_H  (value at the *next* state)
    discount: torch.Tensor,    # (H, B)   gamma * cont_1 .. gamma * cont_H
    lam: float,
) -> torch.Tensor:
    """TD(lambda) returns, computed backwards over an imagined trajectory.

    Returns (H, B): R_t for t = 0..H-1 where
      R_t = r_{t+1} + d_{t+1} * ((1 - lam) * v_{t+1} + lam * R_{t+1})
    with R_{H-1} bootstrapped by v_H.
    """
    returns = torch.zeros_like(reward)
    last = value[-1]
    for t in reversed(range(reward.shape[0])):
        last = reward[t] + discount[t] * ((1 - lam) * value[t] + lam * last)
        returns[t] = last
    return returns


class Moments:
    """EMA of the 5th-95th return percentile range (DreamerV3).

    Used to normalize advantages so the entropy coefficient works across
    reward scales: scale = max(1, EMA(P95 - P5)).
    """

    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95):
        self.decay = decay
        self.low_q = low
        self.high_q = high
        self.low = None
        self.high = None

    def __call__(self, returns: torch.Tensor) -> float:
        # Quantiles on CPU: torch.quantile has spotty MPS support.
        x = returns.detach().flatten().float().cpu()
        low = torch.quantile(x, self.low_q).item()
        high = torch.quantile(x, self.high_q).item()
        if self.low is None:
            self.low, self.high = low, high
        else:
            self.low = self.decay * self.low + (1 - self.decay) * low
            self.high = self.decay * self.high + (1 - self.decay) * high
        return max(1.0, self.high - self.low)


class RunningMean:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float):
        self.total += float(value)
        self.count += 1

    def result(self) -> float:
        if self.count == 0:
            return 0.0
        out = self.total / self.count
        self.total, self.count = 0.0, 0
        return out


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
