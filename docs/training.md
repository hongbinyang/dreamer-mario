# Training

Starting, resuming, and tuning a training run. For what happens *inside* a training step (the
algorithm itself, not the CLI), see [training_pipeline.md](training_pipeline.md).

## Every run has a name — that's the whole model

`--name` is required for training, and it's the only thing that identifies a run. There is no
separate "start" vs. "resume" command: running the same `--name` again just continues it, from
the same directory, with a continuous TensorBoard curve (a second event file is written into the
same run directory — TensorBoard merges event files within a directory automatically).

```bash
python scripts/train.py --name trial
```

- First time: starts fresh from `configs/default.yaml` (+ any `--set` overrides) under
  `runs/trial/`.
- Every time after: detects `runs/trial/ckpt.pt`, loads weights + step counter, and rebuilds
  the model from **the config embedded in that checkpoint** — not from `configs/default.yaml`
  again. `--set` overrides on a resume are applied on top of the checkpoint's config, so loop
  knobs (`train.total_steps`, `train.train_every`, `run.log_every`, ...) are freely adjustable
  between runs, and so are `train.entropy_coef` and `model.unimix` — most of `model.*` and
  `env.grayscale` / `env.size` / `env.action_set` must stay whatever they were on the first run,
  though — changing those makes `agent.load()` fail with a state-dict shape mismatch (a safe,
  loud failure, not silent corruption). Full breakdown of exactly which keys fall in which
  category: [configuration.md](configuration.md#is-it-safe-to-change-this-on-resume).

Ctrl-C any time — state is safe up to the last checkpoint (`run.checkpoint_every`, default
25000 steps). Just re-run the same `--name` command to continue. `agent.save()` writes to a temp
file and atomically renames it into place, so even an interrupt landing mid-save can't leave a
corrupt `ckpt.pt` — the file on disk is always either the previous complete checkpoint or the new
one, never a partial write.

This is true whether the run is in the foreground or detached (the `nohup ... &` recipe below,
or a run started from the [web GUI](webui.md)) — verified directly against a real regression: a
shell backgrounding a process sets `SIGINT` to ignored in the child *before* it execs Python,
and that disposition survives exec, so without an explicit reset a backgrounded `train.py` would
be permanently immune to `SIGINT`/Ctrl-C-from-another-terminal, silently. `scripts/train.py`
resets it back to normal as one of its first actions specifically so this guarantee holds
everywhere, not just when it's the shell's foreground job.

**One caveat**: the replay buffer itself is not checkpointed, only model/optimizer weights and
the step counter. So every resume goes through a fresh `replay.prefill` (5000 steps by default)
warm-up of random actions before training resumes — same as a brand-new run. Frequent
Ctrl-C/resume cycles mean frequent prefill windows; that's normal, not a bug.

## Choosing a device

Every script (`train.py`, `evaluate.py`, `dream.py`, `smoke_test.py`) takes the same `--device`
flag, overriding `run.device` from the config:

```bash
python scripts/train.py --name trial                    # auto-detect (default)
python scripts/train.py --name trial --device cpu        # force CPU
python scripts/train.py --name trial --device cuda:1     # a specific GPU
python scripts/train.py --name trial --device mps        # force Apple Silicon GPU
python scripts/train.py --name trial --device tpu        # requires torch_xla, see below
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
python scripts/train.py --name trial
python scripts/train.py --name trial --set env.grayscale=true      # only meaningful on first run
python scripts/train.py --name trial-sparse --set env.sparse_reward=true --set train.total_steps=1000000
```

### Run a serious long training

Putting the pieces above together for an actual attempt at reliable flag captures, not a quick
check:

1. **Pick a target.** The project's original sizing expectation is ~0.5–1M env frames for strong
   1-1 performance; `configs/default.yaml`'s `train.total_steps=1000000` default matches that. At
   the measured ~11 fps steady state on an M2 Pro, that's **≈25 hours of cumulative training
   time**. You don't have to commit to that number upfront, either — since every run is resumable
   by name, it's fine to start with a smaller `train.total_steps` and just re-run the same `--name`
   with a bigger one later to keep going.

2. **Launch it detached**, so it survives a closed terminal or an accidental Ctrl-C:

   ```bash
   nohup python scripts/train.py --name trial --set train.total_steps=1000000 \
     > runs/trial/train.log 2>&1 &
   disown
   ```

   (Or use `tmux`/`screen` instead, if you'd rather be able to reattach interactively.)

3. **Check in periodically** rather than watching continuously:

   ```bash
   tail -20 runs/trial/train.log            # latest step / best_x / flags line
   python scripts/dashboard.py --name trial  # loss curves, KL, entropy over time
   ```

   See [monitoring.md](monitoring.md) for what each TensorBoard tag means.

4. **Know what disk usage to expect.** `ckpt.pt` is always the same filename — `agent.save()`
   overwrites it every `run.checkpoint_every` steps — so it never grows past its single-checkpoint
   size (~200MB at the default model size) no matter how long the run goes. `open_loop_<step>.gif`
   files *do* accumulate, though: one every `run.video_every` steps, each a unique filename, never
   overwritten. At the defaults (`video_every=20000` over `total_steps=1000000`) that's up to 50
   GIFs, roughly 500KB each — worth knowing about, but nowhere near checkpoint-sized. Run
   `python scripts/cleanup.py --list` any time to check actual sizes.

5. **Know how you'll tell it's working**: `episode/best_x` climbing toward ≈3160 (the flag's
   x-position on 1-1) and `episode/flags` ticking above 0, in the log line or the dashboard.

### Tuning wall-clock speed vs. sample efficiency

`train.train_every` controls how many env steps happen per gradient step (default 16 → the
measured 11 fps above). Raising it, e.g. `--set train.train_every=32`, roughly halves how often
`agent.train_step()` runs — faster wall-clock, fewer gradient updates per env step, so slower
learning per env step. Other speed knobs: `env.grayscale=true` (smaller encoder input),
`model.cnn_depth` / `model.deter` (smaller network). These are architecture/env-shape knobs, so
only set them on a run's *first* invocation — see the resume caveat above.

### Sparse-reward A/B experiment

```bash
python scripts/train.py --name trial --set train.total_steps=1000000
python scripts/train.py --name trial-sparse --set env.sparse_reward=true --set train.total_steps=1000000
python scripts/dashboard.py --name trial --name trial-sparse
```

Identical env/model, only the reward signal changes (flag-only vs. dense x-progress) — see the
README's "Suggested experiments".

## `scripts/train.py` options

| Flag | Default | Meaning |
|---|---|---|
| `--name` | *(required)* | Run identifier; state lives in `<run.logdir>/<name>/`. Re-running the same name resumes it — see "Every run has a name" above. |
| `--config` | `configs/default.yaml` | YAML config for a fresh run. On resume, only its `run.logdir` value is used (to find the checkpoint); everything else comes from the checkpoint's own embedded config instead. |
| `--set KEY=VALUE` | *(none; repeatable)* | Dotted-key override, e.g. `--set train.total_steps=200000`. Repeat the flag for multiple overrides. Applied on top of `--config` (fresh run) or the checkpoint's config (resume). |
| `--logdir` | `None` (→ config's `run.logdir`, i.e. `runs`) | Shorthand for `--set run.logdir=...`. Unlike the same-named flag on `evaluate.py`/`dream.py`/`dashboard.py`/`cleanup.py` (a separate CLI-only default, not read from the YAML), this one *is* the config's `run.logdir` — see the caveat below. |
| `--device` | `None` (→ config's `run.device`, i.e. `auto`) | `auto` \| `cpu` \| `cuda[:N]` \| `mps` \| `tpu`. See "Choosing a device" above. |

**Caveat if you ever change where runs live**: `train.py --logdir` (or `--set run.logdir=...`) changes `run.logdir` itself, so a fresh or resumed run correctly goes wherever you point it. But `evaluate.py`, `dream.py`, `dashboard.py`, and `cleanup.py`'s `--logdir` flags are a *separate*, CLI-only default of `runs` — they don't read `run.logdir` from any config. If you ever redirect training output away from the default `runs/` directory, you'll need to also pass `--logdir` explicitly to those four scripts to match; they won't pick up the change automatically.
