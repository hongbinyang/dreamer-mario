"""Recurrent State-Space Model (RSSM).

State = (h, z):
  h : deterministic recurrent state (GRU)          -- Dreamer v1
  z : stochastic state, 32 categorical groups of   -- discrete latents from
      32 classes with straight-through gradients      DreamerV2
Prior  p(z_t | h_t)            : "imagination" -- predict without seeing obs
Posterior q(z_t | h_t, x_t)    : "perception"  -- correct with the embedding

DreamerV3 details implemented here: unimix (1% uniform mixture) on the
categorical, free bits (clip KL below 1 nat), and KL balancing (separate
dynamics and representation losses).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as td

from .networks import mlp


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, deter: int = 512,
                 stoch: int = 32, classes: int = 32, hidden: int = 512,
                 unimix: float = 0.01):
        super().__init__()
        self.action_dim = action_dim
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        self.stoch_dim = stoch * classes

        # Input to the GRU: previous stochastic state + previous action.
        self.img_in = mlp(self.stoch_dim + action_dim, hidden, 1)
        self.gru = nn.GRUCell(hidden, deter)
        # Prior: predict z logits from h alone.
        self.prior_net = mlp(deter, hidden, 1, self.stoch_dim)
        # Posterior: infer z logits from h and the observation embedding.
        self.post_net = mlp(deter + embed_dim, hidden, 1, self.stoch_dim)

    # ---------------------------------------------------------------- state
    def initial(self, batch: int, device: torch.device) -> dict:
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "stoch": torch.zeros(batch, self.stoch_dim, device=device),
            "logits": torch.zeros(batch, self.stoch, self.classes, device=device),
        }

    def get_feat(self, state: dict) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch_dim

    # ------------------------------------------------------------ internals
    def _dist(self, logits: torch.Tensor) -> td.Independent:
        # logits: (B, stoch, classes) with unimix applied.
        probs = torch.softmax(logits, dim=-1)
        probs = (1 - self.unimix) * probs + self.unimix / self.classes
        return td.Independent(td.OneHotCategoricalStraightThrough(probs=probs), 1)

    def _logits(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.view(-1, self.stoch, self.classes)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        dist = self._dist(logits)
        sample = dist.rsample()  # straight-through one-hot
        return sample.flatten(1)  # (B, stoch*classes)

    # ------------------------------------------------------------- stepping
    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """One step of pure imagination (prior only)."""
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.img_in(x)
        deter = self.gru(x, prev_state["deter"])
        logits = self._logits(self.prior_net(deter))
        stoch = self._sample(logits)
        return {"deter": deter, "stoch": stoch, "logits": logits}

    def obs_step(self, prev_state: dict, prev_action: torch.Tensor,
                 embed: torch.Tensor, is_first: torch.Tensor) -> tuple[dict, dict]:
        """One step of perception. Returns (posterior_state, prior_state)."""
        # Reset state and action where a new episode begins.
        mask = (1.0 - is_first.float()).unsqueeze(-1)
        prev_action = prev_action * mask
        prev_state = {
            "deter": prev_state["deter"] * mask,
            "stoch": prev_state["stoch"] * mask,
            "logits": prev_state["logits"] * mask.unsqueeze(-1),
        }
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], dim=-1)
        logits = self._logits(self.post_net(x))
        stoch = self._sample(logits)
        post = {"deter": prior["deter"], "stoch": stoch, "logits": logits}
        return post, prior

    # ------------------------------------------------------------ sequences
    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor, state: dict | None = None
                ) -> tuple[dict, dict]:
        """Run perception over a (B, T, ...) batch. Returns stacked post/prior.

        action[t] is the action that led INTO obs[t] (zero on is_first).
        """
        B, T = embed.shape[:2]
        if state is None:
            state = self.initial(B, embed.device)
        posts, priors = [], []
        for t in range(T):
            state, prior = self.obs_step(state, action[:, t], embed[:, t], is_first[:, t])
            posts.append(state)
            priors.append(prior)
        stack = lambda seq, key: torch.stack([s[key] for s in seq], dim=1)
        post = {k: stack(posts, k) for k in posts[0]}
        prior = {k: stack(priors, k) for k in priors[0]}
        return post, prior

    def imagine(self, start: dict, policy, horizon: int) -> tuple[dict, torch.Tensor]:
        """Roll out `horizon` steps in latent space using the actor.

        start: dict of flat (N, ...) states. Returns states stacked over time
        (horizon+1, N, ...) including the start state, and the actions taken
        (horizon, N, A).
        """
        state = {k: v.detach() for k, v in start.items()}
        states = [state]
        actions = []
        for _ in range(horizon):
            feat = self.get_feat(state)
            # .sample(), not .rsample(): ActorCritic.loss() always re-derives
            # logp from a fresh, detached forward pass (pure REINFORCE), so no
            # gradient is ever taken through this straight-through path.
            action = policy(feat.detach()).sample()
            state = self.img_step(state, action)
            states.append(state)
            actions.append(action)
        seq = {k: torch.stack([s[k] for s in states], dim=0) for k in state}
        return seq, torch.stack(actions, dim=0)

    # ---------------------------------------------------------------- losses
    def kl_loss(self, post: dict, prior: dict, free_bits: float,
                dyn_scale: float, rep_scale: float) -> tuple[torch.Tensor, dict]:
        """DreamerV3 KL balancing with free bits.

        dyn loss: train the prior to predict the posterior (sg on posterior).
        rep loss: regularize the posterior toward the prior (sg on prior).
        """
        sg = lambda logits: self._dist(logits.detach())
        d = lambda logits: self._dist(logits)
        kl_dyn = td.kl_divergence(sg(post["logits"]), d(prior["logits"]))
        kl_rep = td.kl_divergence(d(post["logits"]), sg(prior["logits"]))
        loss_dyn = torch.clamp(kl_dyn, min=free_bits).mean()
        loss_rep = torch.clamp(kl_rep, min=free_bits).mean()
        loss = dyn_scale * loss_dyn + rep_scale * loss_rep
        metrics = {
            "kl_dyn": kl_dyn.mean().item(),
            "kl_rep": kl_rep.mean().item(),
        }
        return loss, metrics
