# Dreamer plays Super Mario Bros

A from-scratch PyTorch implementation of the Dreamer algorithm (v1/v2/v3
lineage) learning to play Super Mario Bros (NES) from pixels — trained
entirely inside its own learned world model.

Layout of the showcase video produced by `scripts/dream.py`:
`[ real gameplay | the model's imagination | prediction error ]`

## Documentation

- [**Training**](docs/training.md) — starting/resuming a run, choosing a device, speed tuning,
  running a serious long training, the sparse-reward A/B experiment.
- [**Configuration**](docs/configuration.md) — every hyperparameter, what it does, and whether
  it's safe to change on a resumed run.
- [**Monitoring**](docs/monitoring.md) — watching a run's progress, listing and deleting old runs.
- [**Evaluation**](docs/evaluation.md) — evaluating a trained checkpoint, generating the dream
  showcase video.
- [**Baselines**](docs/baselines.md) — the PPO comparison run.
- [**Web GUI**](docs/webui.md) — start/stop/watch/evaluate/compare runs from a browser, a layer
  on top of the same CLI scripts, not a replacement for them.
- [**Testing**](docs/testing.md) — the unit test suite and the smoke test.
- [**Training pipeline**](docs/training_pipeline.md) — step-by-step walkthrough of what happens
  inside one training step: act → replay → world-model update → imagination rollout.
- [**World model design**](docs/design_world_model.md) — why the RSSM looks the way it does:
  discrete categorical latents, unimix, KL balancing, symlog/twohot regression.
- [**Actor-critic design**](docs/design_actor_critic.md) — why the actor-critic trains purely in
  imagination via REINFORCE, percentile return normalization, the EMA slow critic.
- [**Architecture & file map**](docs/architecture.md) — the full directory tree and a table of
  exactly where each Dreamer v1/v2/v3 paper idea lives in code.

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

Device selection (CPU/CUDA/MPS/TPU) is automatic and MPS's CPU-op-fallback is enabled by default —
see [docs/training.md](docs/training.md#choosing-a-device) if you need to override it.

All commands below assume you're running from the repo root (`dreamer-mario/`).

## Train, evaluate, dream

Every run is identified by `--name` — it's required, and it's also how you resume: run the same
`--name` again and it picks up where it left off (same directory, continuous TensorBoard curve).
There's no separate "resume" command.

```bash
python scripts/train.py --name trial
```

See [docs/training.md](docs/training.md) for the full training command reference — including how
to run a serious long training — [docs/monitoring.md](docs/monitoring.md) for watching progress
and cleaning up old runs, and [docs/evaluation.md](docs/evaluation.md) for evaluating a checkpoint
and generating the dream showcase video. See
[docs/training_pipeline.md](docs/training_pipeline.md) for what actually happens inside each
training step.

`scripts/dream.py` is the showcase: the model watches 8 real frames, then predicts ~56 frames
purely in latent space — Goombas, physics, scrolling and all — decoded back to pixels. The
default `--context 8 --horizon 56` covers 64 frames × `frame_skip=4` ≈ 4.3 seconds of NES time,
and the output video now plays back at that same real-time pace (`60 / frame_skip` fps).

## PPO baseline (sample-efficiency comparison)

```bash
pip install stable-baselines3 gymnasium
python baselines/ppo_baseline.py --name trial
```

Uses the identical env wrapper, so env-frame counts are directly comparable — see
[docs/baselines.md](docs/baselines.md) for the full command reference, config keys, and measured
throughput.

## Web GUI

```bash
pip install flask   # optional, not in requirements.txt/environment.yml
python scripts/webui.py
```

Then open `http://127.0.0.1:8000` — start/stop/watch training and PPO baseline runs, evaluate
and dream (with the resulting video playable right in the browser), and compare runs, all backed
by the exact same scripts above. See [docs/webui.md](docs/webui.md) for the full page walkthrough
and a "GUI action → CLI command" table.

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
