import torch

from dreamer.actor_critic import ActorCritic
from dreamer.config import dict_to_ns, load_yaml
from dreamer.world_model import WorldModel

TINY_OVERRIDES = [
    "model.cnn_depth=8", "model.deter=16", "model.stoch=4", "model.classes=4",
    "model.hidden=16", "train.imag_horizon=3",
]


def make_agent_parts(num_actions=4):
    raw = load_yaml("configs/default.yaml", TINY_OVERRIDES)
    cfg = dict_to_ns(raw)
    wm = WorldModel(num_actions, cfg)
    ac = ActorCritic(wm.rssm.feat_dim, num_actions, cfg)
    return wm, ac


def test_actor_and_critic_losses_are_isolated_from_each_other_and_world_model():
    """actor_loss should only ever update the actor, critic_loss only the
    critic, and neither should reach the world model -- agent.py trains the
    world model with its own separate optimizer/loss entirely. If the
    advantage or value target weren't properly `.detach()`-ed, this
    wouldn't show up as a crash or an obviously-wrong loss number; it would
    just silently corrupt training."""
    torch.manual_seed(0)
    wm, ac = make_agent_parts()
    start = wm.rssm.initial(4, torch.device("cpu"))
    actor_loss, critic_loss, metrics = ac.loss(wm, start)

    actor_loss.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.actor.parameters())
    assert all(p.grad is None for p in ac.critic.parameters())

    critic_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ac.critic.parameters())
    assert all(p.grad is None for p in wm.parameters())


def test_slow_critic_ema_update():
    wm, ac = make_agent_parts()
    ac.slow_decay = 0.9
    before = [p.clone() for p in ac.slow_critic.parameters()]
    with torch.no_grad():
        for p in ac.critic.parameters():
            p.add_(1.0)  # push critic weights away from the slow critic
    ac.update_slow_critic()
    for b, s, f in zip(before, ac.slow_critic.parameters(), ac.critic.parameters()):
        assert torch.allclose(s, b * 0.9 + f * 0.1, atol=1e-5)
    assert all(not p.requires_grad for p in ac.slow_critic.parameters())
