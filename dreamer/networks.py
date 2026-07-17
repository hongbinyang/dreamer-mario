"""Network building blocks (DreamerV3-style: LayerNorm + SiLU everywhere)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from .utils import TwoHotDistSymlog


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of NCHW tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvEncoder(nn.Module):
    """64x64 image -> flat embedding. 4 stride-2 convs: 64->32->16->8->4."""

    def __init__(self, in_channels: int = 3, depth: int = 32):
        super().__init__()
        d = depth
        layers = []
        chans = [in_channels, d, 2 * d, 4 * d, 8 * d]
        for i in range(4):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1),
                ChannelLayerNorm(chans[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)
        self.out_dim = 8 * d * 4 * 4

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W) in [-0.5, 0.5]
        x = self.net(obs)
        return x.flatten(1)


class ConvDecoder(nn.Module):
    """Latent features -> 64x64 image mean (Normal with unit variance)."""

    def __init__(self, feat_dim: int, out_channels: int = 3, depth: int = 32):
        super().__init__()
        d = depth
        self.linear = nn.Linear(feat_dim, 8 * d * 4 * 4)
        self.d = d
        chans = [8 * d, 4 * d, 2 * d, d]
        layers = []
        for i in range(3):
            layers += [
                nn.ConvTranspose2d(chans[i], chans[i + 1], kernel_size=4, stride=2, padding=1),
                ChannelLayerNorm(chans[i + 1]),
                nn.SiLU(),
            ]
        layers += [nn.ConvTranspose2d(d, out_channels, kernel_size=4, stride=2, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.linear(feat)
        x = x.view(-1, 8 * self.d, 4, 4)
        return self.net(x)  # (B, C, 64, 64), mean in image space


def mlp(in_dim: int, hidden: int, layers: int, out_dim: int | None = None) -> nn.Sequential:
    seq: list[nn.Module] = []
    dim = in_dim
    for _ in range(layers):
        seq += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        dim = hidden
    if out_dim is not None:
        seq += [nn.Linear(dim, out_dim)]
    return nn.Sequential(*seq)


class TwoHotHead(nn.Module):
    """MLP -> TwoHotDistSymlog. Used for reward and value (DreamerV3)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, num_bins: int = 255,
                 zero_init: bool = True):
        super().__init__()
        self.net = mlp(in_dim, hidden, layers, num_bins)
        if zero_init:
            # Zero-init output layer: predicts 0 initially (stabilizes early training).
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, feat: torch.Tensor) -> TwoHotDistSymlog:
        return TwoHotDistSymlog(self.net(feat))


class BernoulliHead(nn.Module):
    """MLP -> Bernoulli. Used for the continuation predictor."""

    def __init__(self, in_dim: int, hidden: int, layers: int):
        super().__init__()
        self.net = mlp(in_dim, hidden, layers, 1)

    def forward(self, feat: torch.Tensor) -> td.Independent:
        logits = self.net(feat).squeeze(-1)
        return td.Independent(td.Bernoulli(logits=logits), 0)


class ActorHead(nn.Module):
    """MLP -> categorical policy over discrete actions, with unimix."""

    def __init__(self, in_dim: int, hidden: int, layers: int, num_actions: int,
                 unimix: float = 0.01):
        super().__init__()
        self.net = mlp(in_dim, hidden, layers, num_actions)
        self.unimix = unimix
        self.num_actions = num_actions

    def forward(self, feat: torch.Tensor) -> td.OneHotCategoricalStraightThrough:
        logits = self.net(feat)
        probs = torch.softmax(logits, dim=-1)
        probs = (1 - self.unimix) * probs + self.unimix / self.num_actions
        return td.OneHotCategoricalStraightThrough(probs=probs)
