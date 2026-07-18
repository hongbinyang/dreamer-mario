# Training pipeline

One call to `agent.train_step(batch)` ([dreamer/agent.py](../dreamer/agent.py)) does both halves
of Dreamer's loop. Everything below happens inside the `while step < total_steps` loop in
[scripts/train.py](../scripts/train.py).

## 1. Act in the real environment

```
obs --encoder--> embed --RSSM.obs_step--> posterior state --actor--> action
```

- The current posterior RSSM state (`deter`, `stoch`) is carried across steps in `policy_state`.
- The actor ([dreamer/networks.py](../dreamer/networks.py) `ActorHead`) samples a categorical
  action from the *real* posterior state — never from imagination — during acting.
- Below `replay.prefill` (5000 steps by default) actions are uniform random instead, just to
  seed the replay buffer before the model has learned anything.
- One environment step = `frame_skip` (4) raw NES frames, max-pooled over the last two to reduce
  flicker, then resized to 64×64. See [dreamer/envs/mario.py](../dreamer/envs/mario.py).

## 2. Store the transition

`(obs, action, reward, is_first, is_terminal)` goes into the flat ring-buffer
([dreamer/replay.py](../dreamer/replay.py)). Sequences sampled later may cross old episode
boundaries; the RSSM simply resets its state wherever `is_first` is set, so the buffer never
needs to segment episodes explicitly.

## 3. Every `train_every` env steps, run one training step

### 3a. World model update

```
batch (B,T) --encoder--> embed --RSSM.observe--> posterior/prior sequence
                                                 --> decoder, reward head, cont head
loss = recon + reward + continue + KL(post, prior)
```

- `RSSM.observe` walks the sampled sequence once, producing a **posterior** (uses the real frame)
  and a **prior** (predicts blind, from `h_t` alone) at every timestep.
- The decoder reconstructs pixels from the posterior feature; reward/continue heads predict from
  it too.
- The KL term pulls prior and posterior together in both directions at once (KL balancing, see
  [design_world_model.md](design_world_model.md)).
- One Adam step on every world-model parameter (encoder, RSSM, decoder, reward head, continue
  head) together, gradient-clipped.

### 3b. Actor-critic update, purely in imagination

```
posterior states (detached) --RSSM.imagine, horizon=15--> imagined trajectory
                                                          --> reward head, continue head, critic
                                                          --> λ-returns --> actor loss, critic loss
```

- Every posterior state produced in step 3a becomes a **starting point** for a 15-step imagined
  rollout — no real frames or replay data are touched again for this half.
- At each imagined step the actor picks an action, `RSSM.img_step` (prior only) advances the
  latent state, and the reward/continue heads predict what would happen — all inside the learned
  world model, nothing rendered.
- λ-returns are computed backward over the imagined trajectory, bootstrapped by the critic's own
  value estimate at the horizon.
- Actor and critic are optimized with **separate** Adam optimizers and gradient clips (see
  [design_actor_critic.md](design_actor_critic.md) for why the actor loss is REINFORCE, not
  backprop-through-dynamics).
- The EMA "slow critic" is updated last, after the critic's own weights have moved.

## 4. Log, checkpoint, and periodically render a dream

- Every `log_every` steps: scalar metrics (losses, KL, entropy, `best_x`, flag rate, env fps,
  gradient norms, learning rates in use, real-action frequency) go to TensorBoard — full tag list
  in [monitoring.md](monitoring.md).
- Every `video_every` steps: an open-loop prediction GIF is written — real context frames, then
  pure imagination, decoded to pixels — the same mechanism used by `scripts/dream.py`, just run
  automatically during training so you can watch the world model's imagination sharpen over time.
  A matching `wm/open_loop_error` scalar (pixel MSE on the imagined portion) gives that trend a
  number, not just a GIF to eyeball.
- Every `checkpoint_every` steps: `agent.save()` writes weights, optimizer state, and the exact
  training config into one `.pt` file, so `--resume` and `scripts/evaluate.py` /
  `scripts/dream.py` always reconstruct the identical model.

## Loop summary

```
┌─────────────────────────── repeat until total_steps ───────────────────────────┐
│                                                                                  │
│  act (real env) ──> replay buffer ──> sample batch                             │
│                                          │                                      │
│                            ┌─────────────┴─────────────┐                       │
│                            ▼                             ▼                      │
│                     world-model step             (uses posteriors from above)   │
│                  (recon+reward+cont+KL)                   │                     │
│                                                             ▼                    │
│                                                imagination rollout (15 steps)   │
│                                                actor loss + critic loss         │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```
