# Dreamer plays Super Mario Bros

A from-scratch PyTorch implementation of the Dreamer algorithm (v1/v2/v3
lineage) learning to play Super Mario Bros (NES) from pixels — trained
entirely inside its own learned world model.

Layout of the showcase video produced by `scripts/dream.py`:
`[ real gameplay | the model's imagination | prediction error ]`

**Docs:**
[training pipeline](docs/training_pipeline.md) ·
[world model design](docs/design_world_model.md) ·
[actor-critic design](docs/design_actor_critic.md) ·
[architecture & file map](docs/architecture.md)

## Setup (macOS, Apple Silicon)

```bash
cd dreamer-mario
conda env create -f environment.yml
conda activate dreamer-mario
python scripts/smoke_test.py    # ~1-2 min; must print ALL SMOKE TESTS PASSED
```

Python must be 3.10 or 3.11 (`nes-py` does not build on 3.12 — the
environment.yml pins 3.11). The version pins in `requirements.txt` matter:
`gym==0.25.2` + `numpy<2` is the combination that keeps the old NES stack
working.

If `nes-py`'s C++ extension fails to build with an error like
`use of undeclared identifier '__builtin_clzg'`, your Xcode `clang++` is
older than the default SDK's libc++ headers expect. Updating Xcode to a
current version (App Store → Update) is the real fix. If you'd rather not
update Xcode right now, build against an older SDK explicitly instead, then
re-pin the versions nes-py's resolver bumps:

```bash
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk MACOSX_DEPLOYMENT_TARGET=14.0 \
  pip install nes-py==8.2.1
pip install "gym==0.25.2" "gym-super-mario-bros==7.4.0" "numpy<2" "torch>=2.2" \
  opencv-python pyyaml tensorboard imageio imageio-ffmpeg
```

If you ever hit a "not implemented for MPS" error, run with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py
```

All commands below assume you're running from the repo root (`dreamer-mario/`).

## Train

```bash
python scripts/train.py                                   # full run, default config
python scripts/train.py --set env.grayscale=true          # any config key is overridable
python scripts/train.py --set env.sparse_reward=true      # the hard flag-only experiment
python scripts/train.py --resume runs/<name>/ckpt.pt      # continue a run
tensorboard --logdir runs                                 # curves + metrics
```

Progress prints `best_x` (furthest x-position ever reached; the flag on 1-1
is around x=3160) and `flags` (level completions). Open-loop prediction GIFs
are written into the run directory as training progresses — watching the
model's imagination sharpen over time is half the fun. See
[docs/training_pipeline.md](docs/training_pipeline.md) for what actually
happens inside each training step.

**Expectations on an M2:** with the default small config, expect on the
order of days (not hours) to reach reliable flag captures — plan overnight
runs. Useful speed knobs: `train.train_every` (higher = faster wall-clock,
less sample-efficient), `env.grayscale=true`, and `model.cnn_depth`/`deter`.
Checkpoints save every 25k steps, so runs are resumable at any time.

## Evaluate & dream

```bash
python scripts/evaluate.py --ckpt runs/<name>/ckpt.pt --episodes 5 --video eval.mp4
python scripts/dream.py    --ckpt runs/<name>/ckpt.pt --out dream.mp4
```

`dream.py` is the showcase: the model watches 8 real frames, then predicts
~56 frames (about 22 seconds of NES time at frame-skip 4) purely in latent
space — Goombas, physics, scrolling and all — decoded back to pixels.
Checkpoints embed their training config, so these scripts always rebuild
the exact model that was trained.

## PPO baseline (sample-efficiency comparison)

```bash
pip install stable-baselines3 gymnasium
python baselines/ppo_baseline.py --steps 1000000
```

Uses the identical env wrapper, so env-frame counts are directly comparable.

## Suggested experiments

1. **Watch the world model learn:** compare `open_loop_*.gif` files across
   training — imagination goes from mush to crisp physics.
2. **Sample efficiency:** Dreamer vs PPO learning curves at equal env frames.
3. **Sparse reward:** `env.sparse_reward=true` for both — this is where
   imagination-based credit assignment should visibly win.
4. **v2/v3 ablations:** the config exposes `beta_dyn/beta_rep`, `free_bits`,
   `unimix`, and `train.imag_horizon` — turning v3 tricks off and watching
   what breaks is a great way to internalize why they exist.

## Project layout

See [docs/architecture.md](docs/architecture.md) for the full directory
tree, the reasoning behind the `dreamer/` / `scripts/` / `baselines/` split,
and a table of exactly where each Dreamer v1/v2/v3 paper idea lives in code.
