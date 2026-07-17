# Design: the world model (RSSM)

Code: [dreamer/rssm.py](../dreamer/rssm.py), [dreamer/world_model.py](../dreamer/world_model.py),
[dreamer/networks.py](../dreamer/networks.py).

## The core idea

Learn a compact recurrent state that can be rolled forward *without* new observations, accurately
enough that an agent can practice inside it. Two things fall out of that goal:

- **Recurrence** (`h_t`, a GRU): summarizes everything seen so far. This is what gives Mario's
  side-scrolling, partially-observable camera a chance — a single frame can't tell you if a Goomba
  just went offscreen, but `h_t` can remember.
- **Stochastic latent** (`z_t`): captures what *can't* be predicted from `h_t` alone (which enemy
  spawns next, exact pixel noise). Split into prior `p(z_t | h_t)` — the model's blind guess,
  used during imagination — and posterior `q(z_t | h_t, x_t)` — corrected using the real frame,
  used during training.

## Why discrete categorical latents, not Gaussian

`z_t` is 32 independent categorical variables of 32 classes each, sampled with a
straight-through estimator (forward pass = hard one-hot, backward pass = gradient as if it were
the softmax). Three reasons this beats a continuous Gaussian latent here:

1. Mario's dynamics are genuinely discrete/combinatorial (which tile is under Mario's feet, is a
   Goomba alive or dead) — a categorical is a more natural fit than a continuous distribution.
2. Straight-through discrete sampling gives cleaner multi-step KL balancing than reparameterized
   Gaussians tend to in practice (this was DreamerV2's central finding).
3. `unimix` (below) gives discrete latents an easy, bounded way to guarantee non-zero support
   everywhere, which is harder to enforce cleanly on a Gaussian.

## Unimix: never let a latent collapse to zero probability

Both the RSSM's categorical latents and the actor's action distribution mix in 1% uniform
probability: `probs = 0.99 * softmax(logits) + 0.01 / num_classes`. This puts a floor under how
confident (and therefore how un-recoverable) any single categorical class or action can become,
which stabilizes training against the "the model became certain and it was wrong" failure mode —
without needing a separate exploration bonus or temperature schedule.

## KL balancing + free bits — training prior and posterior at different rates

Two separate KL terms, not one:

- **Dynamics loss**: `KL[sg(posterior) ‖ prior]` — trains the *prior* to predict what the
  posterior already figured out. This is what makes imagination possible: by the time you roll
  forward without observations, the prior has learned to approximate the posterior it would have
  computed if it *could* see the frame.
- **Representation loss**: `KL[posterior ‖ sg(prior)]`, weighted 5x lower — keeps the posterior
  from drifting arbitrarily far from what the dynamics can predict, so it doesn't overfit to
  single-frame idiosyncrasies the prior could never reconstruct during imagination.

**Free bits**: each of those KL terms is clamped to a minimum of 1 nat *before* averaging, so the
optimizer stops pushing once a latent is "good enough" — without this, KL tends to collapse toward
zero (the posterior stops using the stochastic latent at all) long before reconstruction quality
plateaus.

## symlog + twohot — one set of hyperparameters for any reward scale

Mario's rewards (x-progress deltas, time penalties, death penalty, occasional +100 for the flag)
span a much wider range than typical control-suite rewards, and the range is different again
between the dense and sparse reward variants. Two DreamerV3 tricks make the same fixed
hyperparameters work regardless:

- **symlog** (`sign(x)·log(1+|x|)`) compresses large magnitudes logarithmically while staying
  ≈linear near zero — squashes outlier rewards without needing per-task reward scaling.
- **twohot regression**: instead of predicting a scalar reward/value directly, the reward and
  critic heads predict a categorical distribution over 255 bins spaced across symlog space, and
  the scalar target is spread across its two nearest bins ("two-hot"). This turns an unbounded
  regression problem into cross-entropy over a fixed, bounded output space — much better behaved
  gradients than raw MSE on a heavy-tailed target.

## Reconstruction loss: no symlog on pixels

Unlike reward/value, the image decoder is trained with plain Gaussian(mean, 1) NLL
(`0.5 * MSE`) directly on the `[-0.5, 0.5]`-normalized pixels — symlog exists to handle *unbounded*
scalar targets, and pixel intensities are already bounded and well-scaled, so it isn't needed
there.
