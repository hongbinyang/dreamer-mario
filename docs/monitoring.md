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

**Diagnostic tags**, added after the exploration-plateau investigation (see
[training.md](training.md#run-a-serious-long-training)) surfaced a few things the metrics above
didn't make visible:

- `wm/grad_norm`, `ac/actor_grad_norm`, `ac/critic_grad_norm` — pre-clip gradient norms (`torch.nn
  .utils.clip_grad_norm_`'s return value, previously discarded). Watch for a norm that's constantly
  pinned at the clip threshold (`train.model_grad_clip=1000` / `train.ac_grad_clip=100`) — a sign
  gradients want to be larger than the clip allows, not just occasional spikes.
- `train/model_lr`, `train/ac_lr` — the learning rate Adam is *actually* using this step. Exists
  specifically to make a known gotcha visible: `--set train.model_lr=...`/`ac_lr=...` are silently
  ignored on resume (`Adam.load_state_dict()` restores the checkpoint's old value) — see
  [configuration.md](configuration.md#is-it-safe-to-change-this-on-resume). If you set an override
  and this line doesn't move, that's why.
- `act/<action>_freq` (e.g. `act/right+A_freq`, `act/NOOP_freq`) — the real-environment action
  distribution over each log window (policy actions only, not `replay.prefill`'s random ones).
  A direct, concrete view of policy diversity — more legible than `ac/entropy` alone (which is
  computed on *imagined* rollouts) for spotting a policy that's collapsed onto repeating one path.
- `wm/open_loop_error` — pixel MSE between imagined and real frames, restricted to the
  truly-imagined portion of each `open_loop_<step>.gif` (excludes the posterior-reconstructed
  context, which isn't a fair test of imagination). Gives the "imagination sharpening over time"
  story a scalar trend line instead of only being eyeballable frame-by-frame in the GIFs.

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
