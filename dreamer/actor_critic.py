"""Actor-critic learned purely inside the world model's imagination.

Dreamer v1 contribution: backprop value estimates through latent rollouts.
DreamerV3 details used here: twohot symlog critic, percentile return
normalization, REINFORCE gradients for discrete actions, EMA critic
regularizer, and fixed entropy scale.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .networks import ActorHead, TwoHotHead
from .utils import Moments, lambda_return


class ActorCritic(nn.Module):
    def __init__(self, feat_dim: int, num_actions: int, cfg):
        super().__init__()
        c = cfg.model
        t = cfg.train
        self.actor = ActorHead(feat_dim, c.hidden, c.head_layers, num_actions, c.unimix)
        self.critic = TwoHotHead(feat_dim, c.hidden, c.head_layers, c.num_bins)
        self.slow_critic = copy.deepcopy(self.critic)
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)
        self.moments = Moments(decay=t.retnorm_decay)
        self.horizon = t.imag_horizon
        self.gamma = t.gamma
        self.lam = t.lam
        self.entropy_coef = t.entropy_coef
        self.slow_decay = t.slow_critic_decay
        self.slow_reg = t.slow_critic_reg

    def update_slow_critic(self):
        with torch.no_grad():
            for s, f in zip(self.slow_critic.parameters(), self.critic.parameters()):
                s.data.lerp_(f.data, 1.0 - self.slow_decay)

    def loss(self, world_model, start: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """start: posterior states from the world-model batch, flattened to
        (N, ...) and detached. Returns (actor_loss, critic_loss, metrics)."""
        rssm = world_model.rssm
        seq, actions = rssm.imagine(start, self.actor, self.horizon)
        feats = rssm.get_feat(seq)  # (H+1, N, F)

        reward = world_model.reward_head(feats[1:]).mean          # (H, N)
        cont = world_model.cont_head(feats[1:]).mean               # (H, N)
        value = self.critic(feats).mean                            # (H+1, N)

        discount = self.gamma * cont
        returns = lambda_return(reward, value[1:], discount, self.lam)  # (H, N)

        # Trajectory weights: down-weight steps after predicted episode end.
        with torch.no_grad():
            ones = torch.ones_like(discount[:1])
            weights = torch.cumprod(torch.cat([ones, discount[:-1]], 0), 0)

        # ------------------------------------------------------------ actor
        scale = self.moments(returns)
        adv = (returns - value[:-1]).detach() / scale
        policy = self.actor(feats[:-1].detach())
        logp = policy.log_prob(actions.detach())
        entropy = policy.entropy()
        actor_loss = -(weights * (logp * adv + self.entropy_coef * entropy)).mean()

        # ----------------------------------------------------------- critic
        # Train on imagined states (detached features).
        value_dist = self.critic(feats[:-1].detach())
        target = returns.detach()
        critic_loss = -(weights * value_dist.log_prob(target)).mean()
        # Regularize toward the slow critic (stabilizes bootstrapping).
        with torch.no_grad():
            slow_target = self.slow_critic(feats[:-1].detach()).mean
        critic_loss = critic_loss - self.slow_reg * (
            weights * value_dist.log_prob(slow_target)).mean()

        metrics = {
            "ac/actor_loss": actor_loss.item(),
            "ac/critic_loss": critic_loss.item(),
            "ac/entropy": entropy.mean().item(),
            "ac/imag_reward": reward.mean().item(),
            "ac/imag_value": value.mean().item(),
            "ac/return_scale": scale,
            "ac/adv": adv.mean().item(),
        }
        return actor_loss, critic_loss, metrics
