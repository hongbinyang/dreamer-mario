"""Train Dreamer on Super Mario Bros.

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --set train.total_steps=200000 --set env.grayscale=true
    python scripts/train.py --resume runs/<name>/ckpt.pt

Run from the repo root so the default --config path resolves.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.agent import DreamerAgent  # noqa: E402
from dreamer.config import load_config, pick_device  # noqa: E402
from dreamer.envs.mario import make_env  # noqa: E402
from dreamer.replay import ReplayBuffer  # noqa: E402
from dreamer.utils import RunningMean  # noqa: E402


def main():
    cfg = load_config()
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

    run_name = time.strftime("%Y%m%d-%H%M%S")
    logdir = pathlib.Path(cfg.run.logdir) / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)
    print(f"logdir: {logdir}  (tensorboard --logdir {cfg.run.logdir})")

    step = 0
    if cfg.resume:
        step = agent.load(cfg.resume)
        print(f"resumed from {cfg.resume} at step {step}")

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
