# Architecture: file map and project layout

## Directory tree

```
dreamer-mario/
├── README.md                setup, run commands, entry point for new readers
├── environment.yml           conda env spec (pinned versions that matter — see README)
├── requirements.txt          same pins, for pip-only installs
├── pytest.ini                 points pytest at tests/
├── conftest.py                 puts the repo root on sys.path so `import dreamer` works in tests
├── configs/
│   └── default.yaml          every hyperparameter, M2-sized defaults
├── tests/                    pytest unit tests for the pure/deterministic layer — see operations.md
│   ├── test_utils.py           symlog/twohot round-trip, lambda_return, return normalization
│   ├── test_replay.py          ring-buffer wraparound and temporal contiguity
│   ├── test_rssm.py            unimix floor, KL free bits, KL-balancing stop-gradient placement
│   ├── test_actor_critic.py    actor/critic/world-model gradient isolation, slow-critic EMA
│   └── test_networks.py        shape and zero-init checks for the network heads
├── docs/
│   ├── architecture.md        this file
│   ├── operations.md          practical commands: training recipes, resuming, monitoring
│   ├── training_pipeline.md   step-by-step collect → world model → imagination loop
│   ├── design_world_model.md  why RSSM / discrete latents / symlog-twohot / KL balancing
│   └── design_actor_critic.md why imagination-only, REINFORCE, return normalization, EMA critic
├── dreamer/                  the algorithm — importable package, no side effects on import
│   ├── config.py              YAML + dotted-key CLI override loader
│   ├── networks.py            CNN encoder/decoder, MLP heads (twohot, Bernoulli, actor)
│   ├── rssm.py                 the recurrent state-space model: prior, posterior, imagination, KL
│   ├── world_model.py          encoder + RSSM + heads wired together, joint loss, video_pred
│   ├── actor_critic.py         imagination rollout, λ-returns, actor/critic losses
│   ├── replay.py               uint8 ring-buffer sequence sampler
│   ├── agent.py                ties world model + actor-critic together: act(), train_step(), save/load
│   └── envs/
│       └── mario.py            gym-super-mario-bros wrapper: frame skip, resize, reward variants
├── scripts/                  entry points — run these, from the repo root
│   ├── smoke_test.py           fast end-to-end sanity check (~1-2 min)
│   ├── train.py                the main training loop; --name required, auto-resumes by name
│   ├── dashboard.py            tensorboard wrapper for one or more named runs
│   ├── cleanup.py              list / delete named runs under runs/
│   ├── evaluate.py             greedy rollouts + gameplay video from a checkpoint
│   └── dream.py                the real-vs-imagined showcase video
├── baselines/                 alternative algorithms for comparison, not part of Dreamer itself
│   └── ppo_baseline.py         stable-baselines3 PPO on the identical env wrapper
└── runs/                      one subdirectory per --name: ckpt.pt, tfevents, GIFs — gitignored
```

`dreamer/` is the reusable core: nothing in it parses CLI args, touches `sys.path`, or writes
files. `scripts/` holds thin entry points that do the argument parsing and I/O and then call into
`dreamer/`. `baselines/` is deliberately separate from `dreamer/` — it's a different algorithm
family (model-free) sharing only the env wrapper, so it has no reason to live inside the Dreamer
package. This split is what makes the project extensible: a second world-model variant would be a
new module in `dreamer/` plus a new script in `scripts/`; a second baseline (DQN, say) is a new
file in `baselines/`; a second environment would be a new file in `dreamer/envs/`.

## Where each paper idea lives in code

| Idea | Paper | File |
|---|---|---|
| RSSM (deterministic GRU + stochastic latent) | Dreamer v1 | `dreamer/rssm.py` |
| Actor-critic learned in latent imagination, λ-returns | Dreamer v1 | `dreamer/actor_critic.py`, `utils.lambda_return` |
| Discrete categorical latents, straight-through gradients | DreamerV2 | `rssm._dist`, `_sample` |
| KL balancing (dyn vs rep losses) | DreamerV2/V3 | `rssm.kl_loss` |
| symlog + twohot reward/value regression | DreamerV3 | `utils.symlog`, `utils.TwoHotDistSymlog` |
| Free bits, unimix, percentile return normalization | DreamerV3 | `rssm.kl_loss`, `networks.ActorHead`, `utils.Moments` |
| EMA slow critic regularizer | DreamerV3 | `actor_critic.update_slow_critic` |

See [design_world_model.md](design_world_model.md) and [design_actor_critic.md](design_actor_critic.md)
for *why* each of these exists, not just where.
