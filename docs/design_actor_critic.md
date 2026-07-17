# Design: actor-critic in imagination

Code: [dreamer/actor_critic.py](../dreamer/actor_critic.py), [dreamer/utils.py](../dreamer/utils.py).

## The core idea

Once the world model can roll forward accurately (see
[design_world_model.md](design_world_model.md)), train the policy entirely on *imagined*
trajectories instead of real environment steps. Every posterior state seen during a world-model
update becomes the start of a 15-step imagined rollout; the actor and critic never touch a real
frame during this half of training. This is what makes Dreamer sample-efficient: thousands of
"practice" rollouts happen per real environment step, all inside the learned model.

## λ-returns: a bias/variance dial over the imagined horizon

The value target at each imagined step is a TD(λ) return — a weighted blend of "trust the
1-step reward + next value" and "trust the further-out imagined trajectory" — bootstrapped by the
critic's own value estimate at the horizon. `lam=0.95` leans toward using more of the imagined
trajectory (lower bias, more variance) rather than truncating to a short-horizon TD estimate.
Steps after a *predicted* episode end (the continue head's output) are down-weighted by the
cumulative product of predicted discounts, so an imagined "death" doesn't keep contributing
imaginary reward afterward.

## Why REINFORCE, not backprop-through-dynamics, for the actor

Earlier Dreamer versions (v1) backpropagated the actor's gradient straight through the imagined
trajectory — since the RSSM transition is differentiable, you can literally take
`d(imagined_value)/d(actor_params)` end to end. DreamerV3 drops this for discrete action spaces
(Mario's action set) in favor of a REINFORCE/score-function estimator:

```
advantage = sg((R_t^λ − v(s_t)) / max(1, S))
actor_loss = −(advantage · log π(a_t|s_t) + η · entropy)
```

Why REINFORCE wins here: backprop-through-dynamics needs the *discrete* categorical action sample
itself to be differentiable (via straight-through), which means differentiating through 15 chained
straight-through estimators — the gradient variance compounds badly over a long, discrete,
learned (imperfect) dynamics function. REINFORCE sidesteps this entirely: it only needs
`log π(a_t | s_t)`, a plain differentiable function of the actor's own output, regardless of how
noisy or discrete the environment dynamics are. The price is the usual REINFORCE variance, which
DreamerV3 controls with the next two tricks.

## Percentile return normalization: why not just divide by std

The advantage is scaled by `S = EMA(percentile(R, 95) − percentile(R, 5), 0.99)`, floored at 1.
This keeps the entropy coefficient and effective learning rate meaningful across wildly different
reward scales (dense vs. sparse Mario reward, or different reward shaping) *without* retuning per
task — the stated goal of DreamerV3's "one fixed hyperparameter set everywhere." Percentiles
rather than mean/std specifically because they're robust to the occasional huge outlier return
(e.g. a lucky imagined flag-grab) that would otherwise dominate a std-based normalizer and shrink
every other gradient to near zero.

## EMA slow critic: a moving target that doesn't move too fast

The critic is regressed toward both the λ-return target *and* its own exponentially-averaged past
self (`slow_critic`, decay 0.98). Bootstrapping value estimates from a value function that is
itself being updated every step is a well-known source of instability (the target moves as fast
as the predictor). Regularizing toward a slowly-updating copy — the same idea as a target network
in DQN — damps that feedback loop without freezing the target outright.

## Fixed entropy coefficient, not adaptive

`η = 3e-4` is a constant, not tuned per environment and not adapted online (e.g. no
target-entropy scheme like SAC). This only works *because* the return normalization above already
absorbs reward-scale differences — with returns kept in a roughly fixed range, a fixed entropy
coefficient behaves consistently across tasks instead of needing to be re-scaled alongside the
reward.
