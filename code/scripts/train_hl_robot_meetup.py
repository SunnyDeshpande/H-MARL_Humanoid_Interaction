import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from two_robot_meetup_env import TwoRobotMeetupEnvCfg, CurriculumConfig


def make_env(curriculum_phase=0):
    """Environment factory with curriculum phase support."""
    def _init():
        return TwoRobotMeetupEnvCfg(render_mode=None, curriculum_phase=curriculum_phase)
    return _init


class EpisodeStatsCallback(BaseCallback):
    """Track per-episode return and length."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.num_envs = None
        self.ep_returns = None
        self.ep_lengths = None
        self.ep_successes = None
        self.episode_returns_hist = []
        self.episode_lengths_hist = []
        self.episode_success_hist = []
        self.total_episodes = 0

    def _on_training_start(self) -> None:
        self.num_envs = self.training_env.num_envs
        self.ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.ep_successes = np.zeros(self.num_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])

        if rewards is None or dones is None:
            return True

        rewards = np.array(rewards).reshape(-1)
        dones = np.array(dones).reshape(-1)

        self.ep_returns += rewards
        self.ep_lengths += 1

        for i in range(self.num_envs):
            if isinstance(infos, list) and i < len(infos):
                if infos[i].get("success", False):
                    self.ep_successes[i] = 1

        for i in range(self.num_envs):
            if dones[i]:
                ep_ret = float(self.ep_returns[i])
                ep_len = int(self.ep_lengths[i])
                ep_success = int(self.ep_successes[i])

                self.episode_returns_hist.append(ep_ret)
                self.episode_lengths_hist.append(ep_len)
                self.episode_success_hist.append(ep_success)
                self.total_episodes += 1

                # Rolling averages
                window = min(100, len(self.episode_returns_hist))
                avg_ret = float(np.mean(self.episode_returns_hist[-window:]))
                avg_len = float(np.mean(self.episode_lengths_hist[-window:]))
                success_rate = float(np.mean(self.episode_success_hist[-window:])) * 100

                print(
                    f"[EP {self.total_episodes:6d}] "
                    f"Step: {self.num_timesteps:>10,} | "
                    f"Return: {ep_ret:>8.2f} | "
                    f"Length: {ep_len:>4} | "
                    f"Success: {ep_success} | "
                    f"AvgRet({window}): {avg_ret:>8.2f} | "
                    f"AvgLen({window}): {avg_len:>6.1f} | "
                    f"SuccessRate({window}): {success_rate:>5.1f}%"
                )

                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0
                self.ep_successes[i] = 0

        return True


class RenderCallback(BaseCallback):
    """Periodically visualize the policy."""

    def __init__(
        self,
        render_env_fn,
        render_every_steps: int = 50_000,
        rollout_len: int = 200,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.render_env_fn = render_env_fn
        self.render_every_steps = render_every_steps
        self.rollout_len = rollout_len
        self._last_render_step = 0
        self._render_env = None

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_render_step < self.render_every_steps:
            return True

        self._last_render_step = self.num_timesteps

        if self._render_env is None:
            self._render_env = self.render_env_fn()

        obs, _ = self._render_env.reset()
        done = False
        truncated = False
        steps = 0
        ep_ret = 0.0
        successes = 0

        if self.verbose:
            print(f"\n[Render] Visualizing at step={self.num_timesteps:,}")

        while not (done or truncated) and steps < self.rollout_len:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self._render_env.step(action)
            ep_ret += float(reward)
            if info.get("success"):
                successes += 1
            steps += 1
            time.sleep(0.03)

        if self.verbose:
            print(f"[Render] Return: {ep_ret:.2f}, Success: {successes > 0}\n")

        return True

    def _on_training_end(self) -> None:
        if self._render_env is not None:
            self._render_env.close()


def find_latest_checkpoint(log_dir: str, prefix: str = "ppo_two_robot") -> str | None:
    path = Path(log_dir)
    candidates = sorted(
        path.glob(f"{prefix}_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def train_phase(
    phase: int,
    total_timesteps: int,
    log_dir: str,
    model=None,
    resume=False,
):
    """Train a single curriculum phase."""
    
    phase_name = CurriculumConfig.PHASES[phase]["name"]
    checkpoint_prefix = f"ppo_two_robot_phase{phase}"

    print(f"\n{'='*80}")
    print(f"PHASE {phase}: {phase_name.upper()} (HOLONOMIC ROBOTS)")
    print(f"{'='*80}\n")

    # Create vectorized environment with curriculum phase
    env = DummyVecEnv([make_env(curriculum_phase=phase) for _ in range(1)])

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ortho_init=True,
    )

    # Load or create model
    if model is None:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
            n_steps=4096,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
        )
    else:
        # Swap environment for new phase
        model.set_env(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50_000 // env.num_envs),
        save_path=log_dir,
        name_prefix=checkpoint_prefix,
    )

    episode_stats_callback = EpisodeStatsCallback(verbose=1)

    render_callback = RenderCallback(
        render_env_fn=lambda: TwoRobotMeetupEnvCfg(render_mode="human", curriculum_phase=phase),
        render_every_steps=50_000,
        rollout_len=200,
        verbose=1,
    )

    callbacks = CallbackList([episode_stats_callback, checkpoint_callback, render_callback])

    print(f"Training phase {phase} ({phase_name}) for {total_timesteps:,} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Saving checkpoint for phase {phase}...")
        model.save(os.path.join(log_dir, f"{checkpoint_prefix}_interrupted"))
        return model

    # Save phase checkpoint
    model.save(os.path.join(log_dir, f"{checkpoint_prefix}_final"))
    print(f"[PHASE {phase}] Saved to {checkpoint_prefix}_final.zip")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Curriculum phase to train (0=approach, 1=orient, 2=meetup).",
    )
    parser.add_argument(
        "--phase0_steps",
        type=int,
        default=150_000,
        help="Timesteps for phase 0 (approach).",
    )
    parser.add_argument(
        "--phase1_steps",
        type=int,
        default=200_000,
        help="Timesteps for phase 1 (orientation).",
    )
    parser.add_argument(
        "--phase2_steps",
        type=int,
        default=400_000,
        help="Timesteps for phase 2 (full meetup).",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs/two_robot_meetup_holonomic",
        help="Root directory for logs & checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint.",
    )
    args = parser.parse_args()

    os.makedirs(args.log_root, exist_ok=True)

    # Phase configurations
    phases_config = [
        (0, args.phase0_steps, "Approach"),
        (1, args.phase1_steps, "Orientation"),
        (2, args.phase2_steps, "Full Meetup"),
    ]

    model = None

    # If resuming, try to load latest checkpoint
    if args.resume:
        for phase_idx, _, _ in phases_config:
            ckpt = find_latest_checkpoint(args.log_root, f"ppo_two_robot_phase{phase_idx}")
            if ckpt:
                print(f"[INFO] Found checkpoint: {ckpt}")
                model = PPO.load(ckpt, device="cuda")
                print(f"[INFO] Loaded model from {ckpt}")
                break

    # Train requested phase(s)
    if args.phase == 0:
        model = train_phase(0, args.phase0_steps, args.log_root, model, args.resume)
    elif args.phase == 1:
        model = train_phase(1, args.phase1_steps, args.log_root, model, args.resume)
    elif args.phase == 2:
        model = train_phase(2, args.phase2_steps, args.log_root, model, args.resume)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()