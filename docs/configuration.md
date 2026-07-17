# Configuration

Every hyperparameter lives in [configs/default.yaml](../configs/default.yaml), loaded by
`dreamer/config.py`. Two ways to change something:

```bash
python scripts/train.py --name flag-run --set model.cnn_depth=16 --set train.total_steps=200000
```

- `--set dotted.key=value`, repeatable. The value is cast to match the *existing* YAML value's
  type (bool/int/float/string) — `--set env.grayscale=true` becomes a real bool, not the string
  `"true"`. There's no validation beyond that: a typo'd key (`--set trian.total_steps=1`) fails
  with a plain `KeyError`, not a helpful message.
- Or copy `configs/default.yaml` to your own file (e.g. `configs/sparse.yaml`) and pass
  `--config configs/sparse.yaml` — better for a config you'll reuse across many runs.

## Is it safe to change this on resume?

Resuming (running `train.py` again with the same `--name`) rebuilds the model from **the
checkpoint's own embedded config**, then applies whatever `--set` overrides you pass on top — see
[training.md](training.md). Not every key behaves the same way once a checkpoint already exists:

1. **Shape-affecting — mismatched values make `agent.load()` fail immediately** with a state-dict
   size error (a safe, loud failure, not silent corruption): all of `model.*`, plus
   `env.grayscale`, `env.action_set`. `env.size` belongs here too, but for a stronger reason — see
   below.
2. **Silently ignored on resume** — `train.model_lr` and `train.ac_lr`. Verified directly: Adam's
   `load_state_dict()` restores each param group's saved hyperparameters, learning rate included,
   from the checkpoint — so a fresh `--set train.model_lr=...` on a resumed run has *no effect at
   all*, even though nothing errors or warns you. If you actually need to change a learning rate
   mid-training, that's not currently supported by the resume path.
3. **Won't crash, but the model was trained under different semantics** — `env.id` (the Mario
   level), `env.frame_skip`, `env.sparse_reward`. Loading succeeds because none of these affect
   network shapes, but resuming a dense-reward run with `env.sparse_reward=true` (say) just
   confuses a reward head that already learned to predict the old signal.
4. **Always safe, freely adjustable every run** — everything under `train.*` except the two
   learning rates above, and everything under `run.*`.

**`env.size` is a special case, not just a resume concern**: `ConvEncoder`'s flattened output size
(`dreamer/networks.py`) is hardcoded as `8 * cnn_depth * 4 * 4`, which is only correct because four
stride-2 convolutions take a 64×64 input down to exactly 4×4. Verified directly — setting
`env.size=32` produces a real encoder output of `8*depth*2*2`, silently mismatched against that
hardcoded formula, and the first forward pass through the RSSM's `post_net` crashes with a
shape error. This isn't resume-specific; **64 is currently the only value of `env.size` that
works at all**, fresh run or not.

## `env`

| Key | Default | Meaning |
|---|---|---|
| `id` | `SuperMarioBros-1-1-v0` | Any `gym-super-mario-bros` level id. |
| `action_set` | `right_only` | `right_only` (5 actions) \| `simple` (7 actions, adds pure jump and left) — see `ACTION_SETS` in [dreamer/envs/mario.py](../dreamer/envs/mario.py). Sizes the actor's output layer and the RSSM's action input. |
| `frame_skip` | `4` | Raw NES frames per env step; the last two are max-pooled together (standard anti-flicker). |
| `size` | `64` | Resize resolution. See the `env.size` note above — 64 is currently the only value that actually works. |
| `grayscale` | `false` | 1-channel vs. 3-channel observations; sizes the encoder/decoder's `in_channels`. |
| `sparse_reward` | `false` | `true` = reward is `100.0` on flag capture and `0.0` otherwise, replacing the dense x-progress/time/death shaping. The hard A/B experiment — see [training.md](training.md#sparse-reward-ab-experiment). |

## `model`

All of these size network layers — see [design_world_model.md](design_world_model.md) for *why*
these particular choices (discrete categorical latents, unimix, etc.).

| Key | Default | Meaning |
|---|---|---|
| `cnn_depth` | `32` | Base channel width of the conv encoder/decoder; channels double at each of the 4 stride-2 layers (32→64→128→256). |
| `deter` | `512` | GRU deterministic recurrent state size (`h_t`). |
| `stoch` | `32` | Number of categorical latent groups. |
| `classes` | `32` | Classes per categorical group — so the full discrete latent is `stoch * classes` = 1024 dims by default. |
| `hidden` | `512` | MLP width used throughout (RSSM's prior/posterior nets, all heads). |
| `head_layers` | `2` | Hidden-layer count in each MLP head (reward, continue, actor, critic — not the RSSM's own prior/posterior nets, which are fixed at 1 layer in `rssm.py`). |
| `num_bins` | `255` | Bins for the twohot symlog reward/value heads — see [design_world_model.md](design_world_model.md#symlog--twohot--one-set-of-hyperparameters-for-any-reward-scale). |
| `unimix` | `0.01` | Uniform-mixture floor on categorical probabilities (RSSM latents *and* the actor's action distribution) — see [design_world_model.md](design_world_model.md#unimix-never-let-a-latent-collapse-to-zero-probability). |

## `replay`

| Key | Default | Meaning |
|---|---|---|
| `capacity` | `200000` | Ring-buffer size in env steps. `200000 * 64*64*3` bytes (uint8 RGB) ≈ 2.3GB RAM — the dominant memory cost of training. Lower this on a lower-RAM machine. |
| `batch_size` | `16` | Sequences per training batch. |
| `seq_len` | `64` | Length of each sampled training sequence, in env steps. |
| `prefill` | `5000` | Random-action env steps before the first gradient step — see [training.md](training.md#understand-the-two-speed-regimes-before-you-launch-a-real-run). Also re-run in full on every resume, since the replay buffer itself isn't checkpointed. |

## `train`

| Key | Default | Meaning |
|---|---|---|
| `total_steps` | `1000000` | Env steps (post frame-skip) to train for — the loop's stopping condition. Always safe to raise on resume to keep going further. |
| `train_every` | `16` | Env steps between gradient steps — the main wall-clock/sample-efficiency dial, see [training.md](training.md#tuning-wall-clock-speed-vs-sample-efficiency). |
| `model_lr` | `1.0e-4` | World-model Adam learning rate. **Ignored on resume** — see the resume-safety note above. |
| `ac_lr` | `3.0e-5` | Actor/critic Adam learning rate. **Ignored on resume**, same reason. |
| `model_grad_clip` | `1000.0` | Gradient norm clip for the world model. Read fresh from config every step (not baked into the optimizer like the LRs), so this one *is* respected on resume. |
| `ac_grad_clip` | `100.0` | Gradient norm clip for actor+critic. Same — respected on resume. |
| `free_bits` | `1.0` | KL floor (nats) below which the dynamics/representation losses stop being pushed further — see [design_world_model.md](design_world_model.md#kl-balancing--free-bits--training-prior-and-posterior-at-different-rates). |
| `beta_dyn` | `0.5` | Weight on the dynamics KL loss (trains the prior to predict the posterior). |
| `beta_rep` | `0.1` | Weight on the representation KL loss (regularizes the posterior toward the prior); kept lower than `beta_dyn` so imagination stays learnable. |
| `imag_horizon` | `15` | Steps rolled out in imagination per world-model batch — see [design_actor_critic.md](design_actor_critic.md). |
| `gamma` | `0.997` | Discount factor for λ-returns. |
| `lam` | `0.95` | The λ in λ-returns — how much to trust the full imagined trajectory vs. a short-horizon TD estimate. |
| `entropy_coef` | `3.0e-4` | Fixed actor entropy bonus — only works as a fixed constant *because* `ac/return_scale` normalizes returns first, see [design_actor_critic.md](design_actor_critic.md#fixed-entropy-coefficient-not-adaptive). |
| `retnorm_decay` | `0.99` | EMA decay for the return-normalization percentile tracker (`utils.Moments`). |
| `slow_critic_decay` | `0.98` | EMA decay for the slow/target critic. |
| `slow_critic_reg` | `1.0` | Weight on the critic's regularizer toward its own slow-EMA copy. |

## `run`

| Key | Default | Meaning |
|---|---|---|
| `device` | `auto` | `auto` \| `cpu` \| `cuda[:N]` \| `mps` \| `tpu` — see [training.md](training.md#choosing-a-device). Overridable per-invocation via `--device`, always safe including on resume. |
| `seed` | `0` | Seeds `torch`, `numpy`, the env, and the replay buffer's sampler at the start of each invocation — including resumes, so changing it mid-training does change the RNG stream going forward (doesn't break anything, just changes reproducibility). |
| `logdir` | `runs` | Parent directory all named runs live under. Used to *locate* an existing checkpoint by `--name`, so it needs to stay consistent with wherever the run's files actually are — not a shape/learning concern, just a "point at the right place" one. |
| `log_every` | `500` | Env steps between TensorBoard scalar logs. |
| `video_every` | `20000` | Env steps between open-loop prediction GIFs — these accumulate (unique filename per step), unlike the checkpoint. |
| `checkpoint_every` | `25000` | Env steps between checkpoint saves. Always overwrites the same `ckpt.pt`. |
