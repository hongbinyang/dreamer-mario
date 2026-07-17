# Dreamer plays Super Mario Bros

A from-scratch PyTorch implementation of the Dreamer algorithm (v1/v2/v3
lineage) learning to play Super Mario Bros (NES) from pixels — trained
entirely inside its own learned world model.

Layout of the showcase video produced by `scripts/dream.py`:
`[ real gameplay | the model's imagination | prediction error ]`

**Docs:**
[operations (how to actually run things)](docs/operations.md) ·
[training pipeline](docs/training_pipeline.md) ·
[world model design](docs/design_world_model.md) ·
[actor-critic design](docs/design_actor_critic.md) ·
[architecture & file map](docs/architecture.md)

## Setup (macOS, Apple Silicon)

```bash
cd dreamer-mario
conda env create -f environment.yml
conda activate dreamer-mario
python -m pytest                # ~1-2 sec, pure unit tests
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
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --name <name>
```

All commands below assume you're running from the repo root (`dreamer-mario/`).

## Train, evaluate, dream

Every run is identified by `--name` — it's required, and it's also how you resume: run the same
`--name` again and it picks up where it left off (same directory, continuous TensorBoard curve).
There's no separate "resume" command.

```bash
python scripts/train.py --name flag-run
```

See [docs/operations.md](docs/operations.md) for the full command reference: detached/overnight
runs, speed/sample-efficiency tuning, monitoring (`scripts/dashboard.py`), deleting old runs
(`scripts/cleanup.py`), evaluating a checkpoint, and generating the dream showcase video. See
[docs/training_pipeline.md](docs/training_pipeline.md) for what actually happens inside each
training step.

`scripts/dream.py` is the showcase: the model watches 8 real frames, then predicts ~56 frames
(about 22 seconds of NES time at frame-skip 4) purely in latent space — Goombas, physics,
scrolling and all — decoded back to pixels.

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
