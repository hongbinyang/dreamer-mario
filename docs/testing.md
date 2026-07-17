# Testing

```bash
python -m pytest             # ~1-2 sec, pure unit tests, no emulator/GPU involved
python scripts/smoke_test.py # ~1-2 min, exercises the real env + one full train_step end to end
```

`pytest` covers the pure, deterministic, easy-to-get-subtly-wrong pieces: symlog/twohot
round-tripping, `lambda_return` against hand-computed values, the replay buffer's ring-buffer
wraparound (never exercised by the smoke test, since it never fills the buffer past capacity),
KL-balancing stop-gradient placement, and actor/critic/world-model gradient isolation. What it
deliberately does *not* try to test is "does the agent learn to beat Mario" — that's a long-
horizon statistical property of a real training run, not a unit-testable one; `smoke_test.py` and
real runs (watched via [monitoring.md](monitoring.md)) are what validate that.

## `scripts/smoke_test.py` options

| Flag | Default | Meaning |
|---|---|---|
| `--no-env` | `False` | Skip the real NES emulator; use synthetic random frames instead. |
| `--config` | `configs/default.yaml` | Base config — `tiny_overrides()` always shrinks the model/replay dims and `imag_horizon` on top, for speed. |
| `--device` | `None` | `auto` \| `cpu` \| `cuda[:N]` \| `mps` \| `tpu` — see [training.md#choosing-a-device](training.md#choosing-a-device). |
