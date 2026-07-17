"""The full Dreamer agent: world model + actor-critic + optimizers."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .actor_critic import ActorCritic
from .world_model import WorldModel


class DreamerAgent:
    def __init__(self, num_actions: int, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.num_actions = num_actions
        self.wm = WorldModel(num_actions, cfg).to(device)
        self.ac = ActorCritic(self.wm.rssm.feat_dim, num_actions, cfg).to(device)
        t = cfg.train
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=t.model_lr, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=t.ac_lr, eps=1e-8)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), lr=t.ac_lr, eps=1e-8)

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def init_policy_state(self) -> dict:
        state = self.wm.rssm.initial(1, self.device)
        state["action"] = torch.zeros(1, self.num_actions, device=self.device)
        return state

    @torch.no_grad()
    def act(self, obs, state: dict, is_first: bool, greedy: bool = False):
        """obs: uint8 numpy (H, W, C). Returns (action_index, new_state)."""
        self.wm.eval()
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0).unsqueeze(0)
        obs_t = self.wm.preprocess(obs_t)[:, 0]  # (1, C, H, W)
        embed = self.wm.encoder(obs_t)
        is_first_t = torch.tensor([is_first], device=self.device)
        prev_action = state["action"]
        rssm_state = {k: v for k, v in state.items() if k != "action"}
        post, _ = self.wm.rssm.obs_step(rssm_state, prev_action, embed, is_first_t)
        feat = self.wm.rssm.get_feat(post)
        dist = self.ac.actor(feat)
        if greedy:
            action = F.one_hot(dist.probs.argmax(-1), self.num_actions).float()
        else:
            action = dist.sample()
        post["action"] = action
        self.wm.train()
        return int(action.argmax(-1).item()), post

    # ---------------------------------------------------------------- train
    def train_step(self, batch: dict) -> dict:
        t = self.cfg.train
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

        # 1) World model.
        wm_loss, post, metrics = self.wm.loss(batch)
        self.wm_opt.zero_grad(set_to_none=True)
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), t.model_grad_clip)
        self.wm_opt.step()

        # 2) Actor-critic in imagination, starting from every posterior state.
        start = {k: v.detach().flatten(0, 1) for k, v in post.items()}
        actor_loss, critic_loss, ac_metrics = self.ac.loss(self.wm, start)

        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), t.ac_grad_clip)
        torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), t.ac_grad_clip)
        self.actor_opt.step()
        self.critic_opt.step()
        self.ac.update_slow_critic()

        metrics.update(ac_metrics)
        return metrics

    # ------------------------------------------------------------ save/load
    def save(self, path: str, step: int):
        from .config import ns_to_dict
        cfg_dict = ns_to_dict(self.cfg)
        cfg_dict.pop("resume", None)
        torch.save({
            "step": step,
            "cfg": cfg_dict,
            "num_actions": self.num_actions,
            "wm": self.wm.state_dict(),
            "ac": self.ac.state_dict(),
            "wm_opt": self.wm_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.wm.load_state_dict(ckpt["wm"])
        self.ac.load_state_dict(ckpt["ac"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        return ckpt.get("step", 0)
