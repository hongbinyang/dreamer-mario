# Baselines

## PPO (sample-efficiency comparison)

```bash
pip install stable-baselines3 gymnasium
python baselines/ppo_baseline.py --steps 1000000
```

Uses the identical env wrapper, so env-frame counts are directly comparable with Dreamer runs.
Not part of the named-run system used elsewhere ([training.md](training.md),
[monitoring.md](monitoring.md)) — it's a separate script with its own `runs_ppo/` output
directory, for comparison purposes only.
