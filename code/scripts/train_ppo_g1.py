#!/usr/bin/env python

"""
PPO training script for G1 flat locomotion task.
Supports curriculum phases: 'stand' (balance) and 'walk' (velocity tracking).
"""

import argparse
import signal
import sys
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher

# CLI Configuration
parser = argparse.ArgumentParser(description="Train PPO for G1 locomotion.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-OneG1-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mode", type=str, default="train", choices=["train", "play"])
parser.add_argument("--total_timesteps", type=int, default=None)
parser.add_argument("--phase", type=str, choices=["stand", "walk"], default="walk")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to resume from.")
parser.add_argument("--vecnorm_path", type=str, default=None, help="Path to VecNormalize stats.")

# Logging & Video
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)

# Hyperparameters
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--n_steps", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--n_epochs", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_range", type=float, default=0.1)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--vf_coef", type=float, default=2.0)
parser.add_argument("--max_grad_norm", type=float, default=0.5)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set default timesteps based on phase
if args_cli.total_timesteps is None:
    args_cli.total_timesteps = 1_000_000 if args_cli.phase == "stand" else 10_000_000

if args_cli.video:
    args_cli.enable_cameras = True

# Launch Simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Clean tqdm on interrupt
def cleanup_pbar(*_):
    import gc
    for obj in gc.get_objects():
        if "tqdm" in type(obj).__name__ and hasattr(obj, "close"):
            obj.close()
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, cleanup_pbar)

# RL Imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.one_g1_env_cfg import OneG1FlatEnvCfg

logger = logging.getLogger(__name__)


def safe_set_weight(rewards_cfg, attr_name, weight):
    if hasattr(rewards_cfg, attr_name):
        reward_term = getattr(rewards_cfg, attr_name)
        if hasattr(reward_term, "weight"):
            reward_term.weight = weight
            return True
    return False


class EpisodeStatsCallback(BaseCallback):
    """Log episode returns and lengths."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
        self.total_episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        if "episode" in infos[0]:
            for info in infos:
                if "episode" in info:
                    self.total_episodes += 1
                    self.episode_returns.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])

                    avg_ret = np.mean(self.episode_returns[-100:]) if self.episode_returns else 0.0
                    avg_len = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0

                    print(f"[Ep {self.total_episodes}] Step: {self.num_timesteps:,} | "
                          f"Ret: {info['episode']['r']:.2f} | Len: {info['episode']['l']} | "
                          f"AvgRet(100): {avg_ret:.2f}")
        return True


def load_vecnormalize(base_env, checkpoint_path, explicit_path):
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    
    if explicit_path:
        return VecNormalize.load(str(Path(explicit_path).expanduser().resolve()), base_env)

    # Search for vecnorm in checkpoint directories
    candidates = [ckpt_path.parent / "vecnormalize_final.pkl", ckpt_path.parent.parent / "vecnormalize_final.pkl"]
    for cand in candidates:
        if cand.exists():
            print(f"[INFO] Loading VecNormalize from: {cand}")
            return VecNormalize.load(str(cand), base_env)

    raise FileNotFoundError("Could not find VecNormalize stats. Pass --vecnorm_path or check checkpoint dir.")


def main():
    # Seeding
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # Environment Config
    env_cfg = OneG1FlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    if hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.export_io_descriptors:
        env_cfg.export_io_descriptors = True

    # Curriculum Phase Setup
    if args_cli.phase == "stand":
        print("[INFO] Configuring for STAND phase")
        
        # Zero commands
        try:
            cmd = env_cfg.commands.base_velocity
            cmd.ranges.lin_vel_x = (0.0, 0.0)
            cmd.ranges.lin_vel_y = (0.0, 0.0)
            cmd.ranges.ang_vel_z = (0.0, 0.0)
            if hasattr(cmd, "noise"): cmd.noise.std = 0.0
        except AttributeError:
            pass

        # Stand rewards
        safe_set_weight(env_cfg.rewards, "track_lin_vel_xy_exp", 3.0)
        safe_set_weight(env_cfg.rewards, "track_ang_vel_z_exp", 1.5)
        safe_set_weight(env_cfg.rewards, "base_height_l2", -2.0)
        safe_set_weight(env_cfg.rewards, "flat_orientation_l2", -2.0)
        safe_set_weight(env_cfg.rewards, "lin_vel_z_l2", -0.5)
        safe_set_weight(env_cfg.rewards, "ang_vel_xy_l2", -0.2)
        safe_set_weight(env_cfg.rewards, "joint_deviation_hip", -0.1)
        safe_set_weight(env_cfg.rewards, "termination_penalty", -50.0)
        # Relax controls for standing
        safe_set_weight(env_cfg.rewards, "dof_torques_l2", -1e-6)
        safe_set_weight(env_cfg.rewards, "action_rate_l2", -0.02)
        safe_set_weight(env_cfg.rewards, "feet_air_time", 0.0)

    else: # WALK phase
        print("[INFO] Configuring for WALK phase")
        try:
            cmd = env_cfg.commands.base_velocity
            cmd.ranges.lin_vel_x = (-0.5, 0.5)
            cmd.ranges.lin_vel_y = (-0.5, 0.5)
            cmd.ranges.ang_vel_z = (-0.5, 0.5)
        except AttributeError:
            pass

        # Walk rewards (upright posture focus)
        safe_set_weight(env_cfg.rewards, "joint_deviation_torso", -0.4)
        safe_set_weight(env_cfg.rewards, "flat_orientation_l2", -4.0)
        safe_set_weight(env_cfg.rewards, "joint_deviation_hip", -0.3)
        safe_set_weight(env_cfg.rewards, "base_height_l2", -3.0)
        safe_set_weight(env_cfg.rewards, "joint_deviation_arms", -0.3)

    # Initialize Environment
    base_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        base_env = gym.wrappers.RecordVideo(
            base_env,
            video_folder=os.path.join("videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    base_env = Sb3VecEnvWrapper(base_env)

    # Logging Directories
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", f"ppo_one_g1_{args_cli.phase}", args_cli.task, run_name))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    
    print(f"[INFO] Log directory: {log_dir}")

    # VecNormalize
    if args_cli.load_checkpoint:
        env = load_vecnormalize(base_env, args_cli.load_checkpoint, args_cli.vecnorm_path)
        env.training = True
    else:
        env = VecNormalize(base_env, training=True, norm_obs=True, norm_reward=True, clip_obs=100.0, gamma=args_cli.gamma)

    # Policy Setup
    policy_kwargs = dict(
        activation_fn=torch.nn.ELU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ortho_init=False,
    )

    if args_cli.load_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args_cli.load_checkpoint}")
        model = PPO.load(args_cli.load_checkpoint, env=env, device=args_cli.device or "auto")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args_cli.learning_rate,
            n_steps=args_cli.n_steps,
            batch_size=args_cli.batch_size,
            n_epochs=args_cli.n_epochs,
            gamma=args_cli.gamma,
            gae_lambda=args_cli.gae_lambda,
            clip_range=args_cli.clip_range,
            ent_coef=args_cli.ent_coef,
            vf_coef=args_cli.vf_coef,
            max_grad_norm=args_cli.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=args_cli.device or "auto",
            tensorboard_log=log_dir,
            seed=args_cli.seed,
        )

    # Callbacks
    callbacks = CallbackList([
        EpisodeStatsCallback(verbose=1),
        CheckpointCallback(
            save_freq=max(1, 500_000 // args_cli.num_envs),
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix=f"ppo_{args_cli.phase}",
            save_vecnormalize=True,
        )
    ])

    # Training Loop
    print(f"[INFO] Starting training for {args_cli.total_timesteps:,} steps...")
    try:
        model.learn(
            total_timesteps=args_cli.total_timesteps,
            callback=callbacks,
            log_interval=args_cli.log_interval,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted.")

    # Save Artifacts
    final_path = os.path.join(log_dir, f"ppo_{args_cli.phase}_final")
    model.save(final_path)
    if isinstance(env, VecNormalize):
        env.save(os.path.join(log_dir, "vecnormalize_final.pkl"))

    print(f"[INFO] Training done. Model saved to {final_path}.zip")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
