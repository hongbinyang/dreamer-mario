# Monitoring and managing runs

## Monitor a run

```bash
python scripts/dashboard.py --name trial                        # one run
python scripts/dashboard.py --name trial --name trial-sparse    # compare runs side by side
```

Thin wrapper around `tensorboard --logdir runs/<name>` (or `--logdir_spec` for multiple named
runs). Key tags: `episode/best_x` (furthest x-position reached; the flag on 1-1 is ≈3160),
`episode/flags` (level completions), `wm/loss`, `wm/kl_dyn`/`wm/kl_rep` (should hover above the
1.0 free-bits floor, not collapse to it or blow up — see
[design_world_model.md](design_world_model.md)), `ac/return_scale`, `ac/entropy` (see
[design_actor_critic.md](design_actor_critic.md)).

## List and delete runs

```bash
python scripts/cleanup.py --list                       # name, step reached, disk size
python scripts/cleanup.py --name old-experiment          # dry-run: prints what would be deleted
python scripts/cleanup.py --name old-experiment --yes    # actually deletes runs/old-experiment/
python scripts/cleanup.py --name a --name b --yes        # delete several at once
```

Checkpoints are 100–200+ MB each, so this matters — deletion is a dry-run unless `--yes` is
passed.

## Options

### `scripts/dashboard.py`

| Flag | Default | Meaning |
|---|---|---|
| `--name` | *(required; repeatable)* | Run name(s) under `runs/` to show. Pass it more than once to compare runs side by side. |
| `--logdir` | `runs` | Parent directory runs live under. |

### `scripts/cleanup.py`

| Flag | Default | Meaning |
|---|---|---|
| `--list` | `False` | List all runs with step reached and disk size. Also the default action if `--name` isn't given. |
| `--name` | *(none; repeatable)* | Run name(s) to delete. |
| `--yes` | `False` | Actually delete. Without it, prints what *would* be deleted (dry-run). |
| `--logdir` | `runs` | Parent directory runs live under. |
