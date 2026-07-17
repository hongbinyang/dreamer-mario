# Evaluation and the dream showcase

Both scripts here accept either `--name <run>` (resolves to `<logdir>/<name>/ckpt.pt`, matching
the naming used everywhere else — see [training.md](training.md)) or an explicit `--ckpt <path>`,
if you want to point at a checkpoint that isn't under `runs/`.

## Evaluate a checkpoint

```bash
python scripts/evaluate.py --name flag-run --episodes 5 --video eval.mp4
python scripts/evaluate.py --ckpt runs/flag-run/ckpt.pt --episodes 5   # equivalent
```

Rebuilds the exact training config from the checkpoint automatically — no need to pass matching
`model.*`/`env.*` flags here (unlike `train.py --resume`-by-name, which does need them to match).

## Generate the dream showcase video

```bash
python scripts/dream.py --name flag-run --out dream.mp4 --context 8 --horizon 56
```

Also auto-rebuilds its config from the checkpoint, same as `evaluate.py`.

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
| `--device` | `None` | See [training.md#choosing-a-device](training.md#choosing-a-device). |
