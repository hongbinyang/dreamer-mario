# Baselines

## PPO (sample-efficiency comparison)

```bash
pip install stable-baselines3 gymnasium
python baselines/ppo_baseline.py --name trial
```

Uses the identical env wrapper, so env-frame counts are directly comparable with a Dreamer run of
the same `--name` (e.g. [docs/training.md](training.md)'s `trial`). Now follows the same
`--name`/`--set` convention as the rest of the project — output goes to `runs_ppo/<name>/`
(`model.zip` plus SB3's own TensorBoard subfolder), separate from Dreamer's `runs/` tree. There's
no resume support here, unlike `train.py` — every invocation starts a fresh model and overwrites
`runs_ppo/<name>/model.zip`.

**Measured throughput: ~89 fps on CPU** (this machine, default hyperparameters) — much faster than
Dreamer's ~11 fps, since PPO has no world-model reconstruction or imagination rollout overhead.
At that rate, `ppo.total_steps=1000000` (the default, matching `train.total_steps`) takes
**≈3 hours**, not the ~25 hours a comparable Dreamer run needs. Don't read too much into that gap
by itself, though — it's expected and part of the point of the comparison: PPO is cheap per step,
but the sample-efficiency question this baseline exists to answer is whether Dreamer needs *far
fewer environment steps* to reach the same performance, not whether it's faster per step.

**stable-baselines3 has no MPS or TPU support** (confirmed directly — `get_device()`'s own
docstring says "for now, it supports only cpu and cuda"), unlike the Dreamer scripts. On an Apple
Silicon Mac, `--device auto` (the default) always resolves to CPU. `--device cuda` still works if
you happen to run this on a CUDA machine instead.

## How to compare after both have run

Dreamer and PPO write TensorBoard logs to separate roots (`runs/<name>/` vs.
`runs_ppo/<name>/PPO_1/`). `scripts/dashboard.py` shows both together via `--ppo-name` (mixes
freely with `--name`, and is exactly what the web GUI's Compare panel builds under the hood — see
[webui.md](webui.md)):

```bash
python scripts/dashboard.py --name trial --ppo-name trial
```

Equivalent to a plain TensorBoard call with `--logdir_spec` (`run_name:path` pairs), if you'd
rather not go through `dashboard.py`:

```bash
tensorboard --logdir_spec=dreamer:runs/trial,ppo:runs_ppo/trial
```

Out of the box, SB3's `Monitor` wrapper only tracks episode reward/length — nothing about
`x_pos` or `flag_get`, so there'd be no PPO-side equivalent of the flag-capture metric the whole
project uses to judge success. `ppo_baseline.py` now installs a small custom callback
(`make_metrics_callback`) that logs `episode/best_x` and `episode/flags` with the exact same tag
names *and* the same cumulative-max / cumulative-count semantics `train.py` uses — verified
directly: running both, then pointing `--logdir_spec` at them, TensorBoard's tag list shows
`episode/best_x`/`episode/flags` under both `dreamer/.` and `ppo/PPO_1`, so they overlay as two
lines on the *same* chart rather than needing to be read as separate, differently-named curves.

`episode/return` doesn't have this treatment — Dreamer's return and SB3's built-in
`rollout/ep_rew_mean` are computed differently enough (different logging cadence, PPO's is a
rolling mean over recent episodes rather than Dreamer's per-log-window mean) that they're better
read as two separate charts, not overlaid.

For a full worked example of reading this comparison — including the failure mode below and what
the actual result looked like — see [dreamer_vs_ppo_trial.md](dreamer_vs_ppo_trial.md).

## A real failure mode: PPO policy collapse, and why `ent_coef` matters

Running `ppo_baseline.py --name trial` with the *previous* default (`ppo.ent_coef=0.02`) collapsed
into a single repeated dead-end trajectory **twice in a row**, in two completely independent runs:
`episode/best_x`, `episode/flags`, `rollout/ep_rew_mean`, and `rollout/ep_len_mean` all went
*perfectly* flat (not just low — bit-for-bit identical across hundreds of thousands of steps),
meaning the policy had converged to a fully deterministic behavior that dies at the same spot every
single episode, with no exploration left to ever escape it. `episode/flags` staying at `0` for a
while early in training is normal (Dreamer's own curve does this too, see
[monitoring.md](monitoring.md)) — a value that's *exactly* constant for an extended stretch,
across multiple logged metrics simultaneously, is the actual signature of collapse, not just slow
progress.

`ppo.ent_coef=0.05` (the current default) was the first value that avoided this on the same setup,
completing the level 8 times over a full `1000000`-step run instead of 0. This isn't a rigorously
swept/validated value — same caveat as before — just confirmed better than `0.02` on this one task.
If you hit the same flatline symptom, try raising `ent_coef` further via
`--set ppo.ent_coef=0.1` (or higher) before assuming something else is broken.

## Options

### `baselines/ppo_baseline.py`

| Flag | Default | Meaning |
|---|---|---|
| `--name` | *(required)* | Run identifier; output under `runs_ppo/<name>/`. |
| `--config` | `configs/default.yaml` | YAML config to load. |
| `--set KEY=VALUE` | *(none; repeatable)* | Dotted-key override — e.g. `--set ppo.ent_coef=0.05`, `--set ppo.total_steps=2000000`, or `--set env.sparse_reward=true` for the A/B comparison. Same mechanism as `train.py`. |
| `--device` | `auto` | `auto` (cuda if available, else cpu) \| `cpu` \| `cuda[:N]` — no `mps`/`tpu`, see above. |

### `ppo` config keys (`configs/default.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `total_steps` | `1000000` | Env steps to train for — matches `train.total_steps` so the two are directly comparable. |
| `n_steps` | `512` | SB3's rollout buffer size (env steps collected per policy update). |
| `batch_size` | `256` | Minibatch size for each PPO update epoch. |
| `learning_rate` | `2.5e-4` | Adam learning rate. |
| `ent_coef` | `0.05` | Entropy bonus coefficient. Bumped from SB3's stock default of `0.01` — `0.02` (this value's own former default) let the policy collapse into a single repeated dead-end trajectory twice in a row on the `trial` comparison; `0.05` was the first value that avoided it, see "A real failure mode" above. Still **not** a rigorously swept/validated value, just confirmed better than the alternatives tried so far — treat it as a starting point, and adjust freely via `--set ppo.ent_coef=...`. |

Not part of the named-run system used for Dreamer runs elsewhere ([training.md](training.md),
[monitoring.md](monitoring.md), [evaluation.md](evaluation.md)) — `scripts/dashboard.py` and
`scripts/cleanup.py` only look under `runs/`, not `runs_ppo/`.
