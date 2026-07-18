import torch

from dreamer.agent import DreamerAgent
from dreamer.config import dict_to_ns, load_yaml

TINY_OVERRIDES = [
    "model.cnn_depth=8", "model.deter=16", "model.stoch=4", "model.classes=4",
    "model.hidden=16", "train.imag_horizon=3",
]


def make_agent(num_actions=4):
    raw = load_yaml("configs/default.yaml", TINY_OVERRIDES)
    cfg = dict_to_ns(raw)
    return DreamerAgent(num_actions, cfg, torch.device("cpu"))


def test_save_writes_atomically_no_leftover_tmp_file(tmp_path):
    agent = make_agent()
    ckpt_path = tmp_path / "ckpt.pt"
    agent.save(ckpt_path, step=42)
    assert ckpt_path.exists()
    assert not (tmp_path / "ckpt.pt.tmp").exists()


def test_save_load_roundtrip_preserves_step_and_weights(tmp_path):
    agent = make_agent()
    ckpt_path = tmp_path / "ckpt.pt"
    with torch.no_grad():
        for p in agent.wm.parameters():
            p.add_(1.0)  # perturb weights so a stale/default load would be detectable
    before = [p.clone() for p in agent.wm.parameters()]

    agent.save(ckpt_path, step=123)

    fresh = make_agent()
    step = fresh.load(str(ckpt_path))
    assert step == 123
    for b, f in zip(before, fresh.wm.parameters()):
        assert torch.allclose(b, f)


def test_save_overwrites_previous_checkpoint_in_place(tmp_path):
    agent = make_agent()
    ckpt_path = tmp_path / "ckpt.pt"
    agent.save(ckpt_path, step=1)
    agent.save(ckpt_path, step=2)
    # Still exactly one file -- checkpoints don't accumulate across saves.
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]
    assert torch.load(ckpt_path, map_location="cpu")["step"] == 2


def test_train_step_reports_grad_norms_and_learning_rates():
    torch.manual_seed(0)
    agent = make_agent(num_actions=4)
    B, T = 2, 8
    batch = {
        "obs": torch.randint(0, 256, (B, T, 64, 64, 3), dtype=torch.uint8),
        "action": torch.nn.functional.one_hot(torch.randint(0, 4, (B, T)), 4).float(),
        "reward": torch.randn(B, T),
        "is_first": torch.zeros(B, T),
        "is_terminal": torch.zeros(B, T),
    }
    batch["is_first"][:, 0] = 1.0

    metrics = agent.train_step(batch)

    for key in ("wm/grad_norm", "ac/actor_grad_norm", "ac/critic_grad_norm"):
        assert key in metrics and metrics[key] >= 0.0
    # Fresh agent, first step -- lr should be exactly the configured value
    # (this is what would silently NOT hold after a resume with an
    # overridden --set train.model_lr=..., which is the whole point of
    # tracking this metric).
    assert metrics["train/model_lr"] == agent.cfg.train.model_lr
    assert metrics["train/ac_lr"] == agent.cfg.train.ac_lr
