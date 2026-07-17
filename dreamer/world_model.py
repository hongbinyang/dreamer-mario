"""World model: learns to simulate Mario inside its latent space."""
from __future__ import annotations

import torch
import torch.nn as nn

from .networks import ConvEncoder, ConvDecoder, TwoHotHead, BernoulliHead
from .rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, action_dim: int, cfg):
        super().__init__()
        c = cfg.model
        in_channels = 1 if cfg.env.grayscale else 3
        self.encoder = ConvEncoder(in_channels, c.cnn_depth)
        self.rssm = RSSM(
            action_dim=action_dim,
            embed_dim=self.encoder.out_dim,
            deter=c.deter,
            stoch=c.stoch,
            classes=c.classes,
            hidden=c.hidden,
            unimix=c.unimix,
        )
        feat = self.rssm.feat_dim
        self.decoder = ConvDecoder(feat, in_channels, c.cnn_depth)
        self.reward_head = TwoHotHead(feat, c.hidden, c.head_layers, c.num_bins)
        self.cont_head = BernoulliHead(feat, c.hidden, c.head_layers)
        self.cfg = cfg

    @staticmethod
    def preprocess(obs: torch.Tensor) -> torch.Tensor:
        """uint8 (B, T, H, W, C) -> float (B, T, C, H, W) in [-0.5, 0.5]."""
        obs = obs.float() / 255.0 - 0.5
        return obs.permute(0, 1, 4, 2, 3)

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict, dict]:
        """batch: obs (B,T,H,W,C) uint8, action (B,T,A), reward (B,T),
        is_first (B,T), is_terminal (B,T)."""
        c = self.cfg
        obs = self.preprocess(batch["obs"])
        B, T = obs.shape[:2]

        embed = self.encoder(obs.flatten(0, 1)).view(B, T, -1)
        post, prior = self.rssm.observe(embed, batch["action"], batch["is_first"])
        feat = self.rssm.get_feat(post)  # (B, T, F)

        # Reconstruction: Normal(mean, 1) NLL == 0.5*MSE summed over pixels.
        recon = self.decoder(feat.flatten(0, 1)).view(B, T, *obs.shape[2:])
        loss_recon = 0.5 * ((recon - obs) ** 2).sum(dim=(2, 3, 4)).mean()

        # Reward and continuation prediction.
        reward_dist = self.reward_head(feat)
        loss_reward = -reward_dist.log_prob(batch["reward"]).mean()
        cont_target = 1.0 - batch["is_terminal"].float()
        cont_dist = self.cont_head(feat)
        loss_cont = -cont_dist.log_prob(cont_target).mean()

        loss_kl, kl_metrics = self.rssm.kl_loss(
            post, prior, c.train.free_bits, c.train.beta_dyn, c.train.beta_rep)

        loss = loss_recon + loss_reward + loss_cont + loss_kl
        metrics = {
            "wm/loss": loss.item(),
            "wm/recon": loss_recon.item(),
            "wm/reward": loss_reward.item(),
            "wm/cont": loss_cont.item(),
            **{f"wm/{k}": v for k, v in kl_metrics.items()},
        }
        return loss, post, metrics

    @torch.no_grad()
    def video_pred(self, batch: dict, context: int = 5, horizon: int = 45) -> torch.Tensor:
        """Open-loop prediction video for logging: `context` frames of
        perception, then pure imagination with the recorded actions.

        Returns (T, H, W*3, C) uint8: [truth | model | error] side by side.
        """
        obs = self.preprocess(batch["obs"][:1])  # single sequence
        B, T = obs.shape[:2]
        horizon = min(horizon, T - context)
        embed = self.encoder(obs.flatten(0, 1)).view(B, T, -1)
        post, _ = self.rssm.observe(
            embed[:, :context], batch["action"][:1, :context],
            batch["is_first"][:1, :context])
        state = {k: v[:, -1] for k, v in post.items()}
        recon_feats = [self.rssm.get_feat({k: v[:, t] for k, v in post.items()})
                       for t in range(context)]
        img_feats = []
        for t in range(context, context + horizon):
            state = self.rssm.img_step(state, batch["action"][:1, t])
            img_feats.append(self.rssm.get_feat(state))
        feats = torch.stack(recon_feats + img_feats, dim=1)  # (1, C+H, F)
        model = self.decoder(feats.flatten(0, 1)).view(1, -1, *obs.shape[2:])
        truth = obs[:, : context + horizon]
        error = (model - truth) / 2.0  # gray = perfect prediction after +0.5 shift
        video = torch.cat([truth, model, error], dim=-1)  # concat on width
        video = ((video + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        return video[0].permute(0, 2, 3, 1).cpu()  # (T, H, 3W, C)
