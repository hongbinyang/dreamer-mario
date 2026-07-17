"""Train Dreamer on Super Mario Bros. Every run is identified by --name;
running the same --name again automatically resumes it (same logdir, same
config it was started with, continuous TensorBoard curve).

    python scripts/train.py --name flag-run
    python scripts/train.py --name flag-run --set env.grayscale=true
    python scripts/train.py --name sparse-ablation --set env.sparse_reward=true

Ctrl-C any time; state is safe up to the last checkpoint (run.checkpoint_every
steps). Re-run the same command to continue. Run from the repo root so the
default --config path resolves. See docs/operations.md.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.agent import DreamerAgent  # noqa: E402
from dreamer.config import apply_overrides, dict_to_ns, load_yaml, pick_device  # noqa: E402
from dreamer.envs.mario import make_env  # noqa: E402
from dreamer.replay import ReplayBuffer  # noqa: E402
from dreamer.utils import RunningMean  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                         help="run identifier; state lives in <run.logdir>/<name>/")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[],
                         help="dotted.key=value overrides")
    args = parser.parse_args()

    # A fresh read of --config/--set tells us where runs live (run.logdir),
    # which is all we need to check whether this name already has a
    # checkpoint. If it does, the checkpoint's own embedded config (not this
    # read) becomes the source of truth below — --set overrides still apply
    # on top of it, so loop knobs like total_steps remain changeable, but
    # model/env shape must match the original run or agent.load() will fail.
    raw = load_yaml(args.config, args.set)
    logdir = pathlib.Path(raw["run"]["logdir"]) / args.name
    ckpt_path = logdir / "ckpt.pt"
    resuming = ckpt_path.exists()
    if resuming:
        raw = apply_overrides(torch.load(ckpt_path, map_location="cpu")["cfg"], args.set)
    cfg = dict_to_ns(raw)

    device = pick_device(cfg.run.device)
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    print(f"device: {device}")

    env = make_env(cfg, seed=cfg.run.seed)
    agent = DreamerAgent(env.num_actions, cfg, device)
    replay = ReplayBuffer(
        capacity=cfg.replay.capacity,
        obs_shape=env.obs_shape,
        num_actions=env.num_actions,
        seq_len=cfg.replay.seq_len,
        batch_size=cfg.replay.batch_size,
        seed=cfg.run.seed,
    )

    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)
    print(f"run: {args.name}  logdir: {logdir}")

    step = 0
    if resuming:
        step = agent.load(str(ckpt_path))
        print(f"resumed '{args.name}' at step {step}")
    # The replay buffer itself is not checkpointed, so every resume goes
    # through a fresh replay.prefill warm-up (random actions) before
    # training resumes, same as a brand-new run.

    obs = env.reset()
    replay.add(obs, action=0, reward=0.0, is_first=True, is_terminal=False)
    policy_state = agent.init_policy_state()
    is_first = True
    episode_returns, episode_lens, best_x = [], [], 0
    flags = 0
    episodes = 0
    meters = {}
    t0, step0 = time.time(), step

    while step < cfg.train.total_steps:
        # ------------------------------------------------------------ act
        if len(replay) < cfg.replay.prefill:
            action = int(np.random.randint(env.num_actions))
            # Keep the policy state in sync even during prefill.
            _, policy_state = agent.act(obs, policy_state, is_first)
            policy_state["action"] = torch.nn.functional.one_hot(
                torch.tensor([action], device=device), env.num_actions).float()
        else:
            action, policy_state = agent.act(obs, policy_state, is_first)
        is_first = False

        obs, reward, done, info = env.step(action)
        step += 1
        replay.add(obs, action, reward, is_first=False, is_terminal=done)

        if done:
            episodes += 1
            episode_returns.append(info["episode_return"])
            episode_lens.append(info["episode_len"])
            best_x = max(best_x, int(info.get("x_pos", 0)))
            flags += int(bool(info.get("flag_get", False)))
            obs = env.reset()
            replay.add(obs, action=0, reward=0.0, is_first=True, is_terminal=False)
            policy_state = agent.init_policy_state()
            is_first = True

        # ---------------------------------------------------------- train
        if len(replay) >= cfg.replay.prefill and step % cfg.train.train_every == 0:
            batch = replay.sample()
            metrics = agent.train_step(batch)
            for k, v in metrics.items():
                meters.setdefault(k, RunningMean()).update(v)

        # ------------------------------------------------------------ log
        if step % cfg.run.log_every == 0:
            fps = (step - step0) / max(time.time() - t0, 1e-8)
            t0, step0 = time.time(), step
            for k, meter in meters.items():
                writer.add_scalar(k, meter.result(), step)
            if episode_returns:
                writer.add_scalar("episode/return", float(np.mean(episode_returns)), step)
                writer.add_scalar("episode/length", float(np.mean(episode_lens)), step)
                episode_returns, episode_lens = [], []
            writer.add_scalar("episode/best_x", best_x, step)
            writer.add_scalar("episode/flags", flags, step)
            writer.add_scalar("perf/env_fps", fps, step)
            print(f"step {step:>8d} | episodes {episodes:>5d} | best_x {best_x:>5d} "
                  f"| flags {flags} | {fps:5.1f} env fps")

        if (step % cfg.run.video_every == 0
                and len(replay) > cfg.replay.seq_len + 1):
            video = agent.wm.video_pred(
                {k: torch.as_tensor(v, device=device)
                 for k, v in replay.sample().items()})
            frames = video.numpy()
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            import imageio
            imageio.mimsave(logdir / f"open_loop_{step}.gif", list(frames), fps=10)
            try:  # TensorBoard video needs moviepy; the GIF above always works.
                writer.add_video("wm/open_loop",
                                 video.permute(0, 3, 1, 2).unsqueeze(0), step, fps=10)
            except ImportError:
                pass

        if step % cfg.run.checkpoint_every == 0:
            agent.save(logdir / "ckpt.pt", step)

    agent.save(logdir / "ckpt.pt", step)
    env.close()
    writer.close()
    print("done.")


if __name__ == "__main__":
    main()
