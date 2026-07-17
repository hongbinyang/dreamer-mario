# Operations: running training, evaluation, and dreaming

Practical commands and what to actually pass on the command line. For what happens *inside* a
training step, see [training_pipeline.md](training_pipeline.md).

## Run the tests

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
real runs (watched via `scripts/dashboard.py`) are what validate that.

## Choosing a device

Every script (`train.py`, `evaluate.py`, `dream.py`, `smoke_test.py`) takes the same `--device`
flag, overriding `run.device` from the config:

```bash
python scripts/train.py --name flag-run                    # auto-detect (default)
python scripts/train.py --name flag-run --device cpu        # force CPU
python scripts/train.py --name flag-run --device cuda:1     # a specific GPU
python scripts/train.py --name flag-run --device mps        # force Apple Silicon GPU
python scripts/train.py --name flag-run --device tpu        # requires torch_xla, see below
```

`auto` (the default) probes in order: **CUDA → MPS → TPU → CPU**, taking the first one available.
Unlike `model.*`/`env.*`, changing `--device` between runs is always safe, including on resume —
checkpoints are just tensors, `agent.load()` moves them onto whatever device you ask for.

**MPS ops fallback is automatic.** Not every PyTorch op is implemented on Apple's MPS backend yet;
`pick_device()` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` itself whenever the resolved device is MPS, so
unsupported ops silently run on CPU instead of crashing. You don't need to set this env var by
hand anymore — it's only worth setting it explicitly to `0` yourself if you want a hard failure
instead, e.g. while tracking down exactly which op isn't supported.

**TPU support (`--device tpu` or auto-detected under `auto`) is untested.** It's wired up via the
optional [`torch_xla`](https://github.com/pytorch/xla) package (never a hard dependency — nothing
else in this project requires it, and it isn't installed by `environment.yml`/`requirements.txt`).
This repo was developed entirely on Apple Silicon/MPS, so if you try it on real TPU hardware and
training doesn't actually speed up, the likely culprit is [dreamer/rssm.py](../dreamer/rssm.py)'s
`imagine()` — a plain Python loop over the imagination horizon. XLA's lazy tracing generally wants
explicit `xm.mark_step()` calls per iteration to compile well; without them it may still run
correctly, just without the performance TPUs are for. Treat this as a starting point, not a
verified fast path.

## Every run has a name — that's the whole model

`--name` is required for training, and it's the only thing that identifies a run. There is no
separate "start" vs. "resume" command: running the same `--name` again just continues it, from
the same directory, with a continuous TensorBoard curve (a second event file is written into the
same run directory — TensorBoard merges event files within a directory automatically).

```bash
python scripts/train.py --name flag-run
```

- First time: starts fresh from `configs/default.yaml` (+ any `--set` overrides) under
  `runs/flag-run/`.
- Every time after: detects `runs/flag-run/ckpt.pt`, loads weights + step counter, and rebuilds
  the model from **the config embedded in that checkpoint** — not from `configs/default.yaml`
  again. `--set` overrides on a resume are applied on top of the checkpoint's config, so loop
  knobs (`train.total_steps`, `train.train_every`, `run.log_every`, ...) are freely adjustable
  between runs, but `model.*` / `env.grayscale` / `env.size` / `env.action_set` must stay whatever
  they were on the first run — changing those makes `agent.load()` fail with a state-dict shape
  mismatch (a safe, loud failure, not silent corruption).

Ctrl-C any time — state is safe up to the last checkpoint (`run.checkpoint_every`, default
25000 steps). Just re-run the same `--name` command to continue.

**One caveat**: the replay buffer itself is not checkpointed, only model/optimizer weights and
the step counter. So every resume goes through a fresh `replay.prefill` (5000 steps by default)
warm-up of random actions before training resumes — same as a brand-new run. Frequent
Ctrl-C/resume cycles mean frequent prefill windows; that's normal, not a bug.

## Understand the two speed regimes before you launch a real run

- **Below `replay.prefill`**: only the environment runs, no gradient steps yet — fast (~100+ env
  fps observed on an M2 Pro).
- **Once prefill is crossed**: `agent.train_step()` fires every `train_every` (16) env steps, and
  throughput drops to a steady state dominated by that gradient-step cost. Measured on an M2 Pro
  with the default config: **~11 env fps steady state**. At 11 fps, the config default
  `train.total_steps=1000000` is ≈25 hours of *cumulative* training time — not necessarily one
  sitting, since you can always stop and resume.

## Recipes

### Start (or continue) a run

```bash
python scripts/train.py --name flag-run
python scripts/train.py --name flag-run --set env.grayscale=true      # only meaningful on first run
python scripts/train.py --name sparse-ablation --set env.sparse_reward=true --set train.total_steps=1000000
```

### Run it detached, so a closed terminal or an accidental Ctrl-C doesn't kill it

```bash
nohup python scripts/train.py --name flag-run --set train.total_steps=1000000 \
  > runs/flag-run/train.log 2>&1 &
disown
tail -f runs/flag-run/train.log
```

(Or use `tmux`/`screen` if you'd rather be able to reattach interactively.)

### Tuning wall-clock speed vs. sample efficiency

`train.train_every` controls how many env steps happen per gradient step (default 16 → the
measured 11 fps above). Raising it, e.g. `--set train.train_every=32`, roughly halves how often
`agent.train_step()` runs — faster wall-clock, fewer gradient updates per env step, so slower
learning per env step. Other speed knobs: `env.grayscale=true` (smaller encoder input),
`model.cnn_depth` / `model.deter` (smaller network). These are architecture/env-shape knobs, so
only set them on a run's *first* invocation — see the resume caveat above.

### Monitor a run

```bash
python scripts/dashboard.py --name flag-run                        # one run
python scripts/dashboard.py --name flag-run --name sparse-ablation  # compare runs side by side
```

Thin wrapper around `tensorboard --logdir runs/<name>` (or `--logdir_spec` for multiple named
runs). Key tags: `episode/best_x` (furthest x-position reached; the flag on 1-1 is ≈3160),
`episode/flags` (level completions), `wm/loss`, `wm/kl_dyn`/`wm/kl_rep` (should hover above the
1.0 free-bits floor, not collapse to it or blow up — see
[design_world_model.md](design_world_model.md)), `ac/return_scale`, `ac/entropy` (see
[design_actor_critic.md](design_actor_critic.md)).

### List and delete runs

```bash
python scripts/cleanup.py --list                       # name, step reached, disk size
python scripts/cleanup.py --name old-experiment          # dry-run: prints what would be deleted
python scripts/cleanup.py --name old-experiment --yes    # actually deletes runs/old-experiment/
python scripts/cleanup.py --name a --name b --yes        # delete several at once
```

Checkpoints are 100–200+ MB each, so this matters — deletion is a dry-run unless `--yes` is
passed.

## Evaluate a checkpoint

```bash
python scripts/evaluate.py --ckpt runs/<name>/ckpt.pt --episodes 5 --video eval.mp4
```

Like resuming, `evaluate.py` rebuilds the exact training config from the checkpoint automatically
— no need to pass matching `model.*`/`env.*` flags here.

## Generate the dream showcase video

```bash
python scripts/dream.py --ckpt runs/<name>/ckpt.pt --out dream.mp4 --context 8 --horizon 56
```

Also auto-rebuilds its config from the checkpoint, same as `evaluate.py`.

## Sparse-reward A/B experiment

```bash
python scripts/train.py --name dense-baseline --set train.total_steps=1000000
python scripts/train.py --name sparse-ablation --set env.sparse_reward=true --set train.total_steps=1000000
python scripts/dashboard.py --name dense-baseline --name sparse-ablation
```

Identical env/model, only the reward signal changes (flag-only vs. dense x-progress) — see the
README's "Suggested experiments".

## PPO baseline

```bash
pip install stable-baselines3 gymnasium
python baselines/ppo_baseline.py --steps 1000000
```

(Not part of the named-run system above — it's a separate script with its own `runs_ppo/`
output directory, for comparison purposes only.)
