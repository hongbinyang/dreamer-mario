# Evaluation and the dream showcase

Both scripts here accept either `--name <run>` (resolves to `<logdir>/<name>/ckpt.pt`, matching
the naming used everywhere else — see [training.md](training.md)) or an explicit `--ckpt <path>`,
if you want to point at a checkpoint that isn't under `runs/`.

## Evaluate a checkpoint

```bash
python scripts/evaluate.py --name trial --episodes 5 --video eval.mp4
python scripts/evaluate.py --ckpt runs/trial/ckpt.pt --episodes 5   # equivalent
```

Rebuilds the exact training config from the checkpoint automatically — no need to pass matching
`model.*`/`env.*` flags here (unlike `train.py --resume`-by-name, which does need them to match).

Video plays back at **real-time by default** (`60 / env.frame_skip` fps — 15 fps at the default
`frame_skip=4`): each recorded frame is one `env.step()`, which already advances `frame_skip` real
NES frames, so real-time playback isn't 30 or any other arbitrary constant. Override with `--fps`
if you want slow motion, e.g. `--fps 5` for closely inspecting individual frames.

## Generate the dream showcase video

```bash
python scripts/dream.py --name trial --out dream.mp4 --context 8 --horizon 56
```

Also auto-rebuilds its config from the checkpoint, same as `evaluate.py`, and defaults to the same
real-time `60 / env.frame_skip` fps (overridable with `--fps`).

## Options

### `scripts/evaluate.py`

| Flag | Default | Meaning |
|---|---|---|
| `--name` | *(required; mutually exclusive with `--ckpt`)* | Run name under `--logdir`; resolves to `<logdir>/<name>/ckpt.pt`. |
| `--ckpt` | *(required; mutually exclusive with `--name`)* | Explicit checkpoint path. |
| `--logdir` | `runs` | Parent directory runs live under (used with `--name`). |
| `--config` | `configs/default.yaml` | Fallback only, used if the checkpoint has no embedded config. |
| `--episodes` | `5` | Number of greedy evaluation episodes to run. |
| `--video` | `None` | If set, save the first episode's real gameplay to this path (e.g. `eval.mp4`). |
| `--fps` | `None` (→ `60 / env.frame_skip`, real-time) | Video playback fps. Lower it for slow motion. |
| `--device` | `None` | See [training.md#choosing-a-device](training.md#choosing-a-device). |

### `scripts/dream.py`

| Flag | Default | Meaning |
|---|---|---|
| `--name` | *(required; mutually exclusive with `--ckpt`)* | Run name under `--logdir`; resolves to `<logdir>/<name>/ckpt.pt`. |
| `--ckpt` | *(required; mutually exclusive with `--name`)* | Checkpoint to dream from. |
| `--logdir` | `runs` | Parent directory runs live under (used with `--name`). |
| `--config` | `configs/default.yaml` | Fallback only, same as `evaluate.py`. |
| `--out` | `dream.mp4` | Output video path. |
| `--context` | `8` | Real frames the model watches before it starts imagining. |
| `--horizon` | `56` | Frames to imagine after the context window. |
| `--upscale` | `4` | Integer upscale factor for the output video (64×64 is tiny otherwise). |
| `--fps` | `None` (→ `60 / env.frame_skip`, real-time) | Video playback fps. Lower it for slow motion. |
| `--device` | `None` | See [training.md#choosing-a-device](training.md#choosing-a-device). |
