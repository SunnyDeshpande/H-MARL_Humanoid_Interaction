#!/usr/bin/env python

"""
Evaluation script for G1 flat locomotion task using a trained PPO policy.
"""

import argparse
import sys
import random
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# CLI setup
parser = argparse.ArgumentParser(description="Evaluate G1 locomotion policy.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-OneG1-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--phase", type=str, choices=["stand", "walk"], default="walk", help="Evaluation phase.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint (.zip).")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode.")
parser.add_argument("--num_episodes", type=int, default=10, help="Total episodes to evaluate.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# IsaaceLab and RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.one_g1_env_cfg import OneG1FlatEnvCfg


def main():
    # Seeding
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # Configure Environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    if hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Set command ranges based on phase
    try:
        cmd = env_cfg.commands.base_velocity
        if args_cli.phase == "stand":
            print("[INFO] Phase: STAND (Zero Commands)")
            cmd.ranges.lin_vel_x = (0.0, 0.0)
            cmd.ranges.lin_vel_y = (0.0, 0.0)
            cmd.ranges.ang_vel_z = (0.0, 0.0)
        else:
            print("[INFO] Phase: WALK (Motion Commands)")
            cmd.ranges.lin_vel_x = (-1.0, 1.0)
            cmd.ranges.lin_vel_y = (-1.0, 1.0)
            cmd.ranges.ang_vel_z = (-1.6, 1.6)

        # Reduce noise for evaluation
        noise_cfg = getattr(cmd, "noise", None)
        if noise_cfg is not None:
            std_val = getattr(noise_cfg, "std", None)
            if isinstance(std_val, (tuple, list)):
                noise_cfg.std = tuple(0.5 * float(x) for x in std_val)
            elif isinstance(std_val, (float, int)):
                noise_cfg.std = 0.5 * float(std_val)

    except AttributeError as e:
        print(f"[WARN] Failed to modify command ranges: {e}")

    # Create Environment
    print(f"[INFO] Creating environment: {args_cli.task}")
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    vec_env = Sb3VecEnvWrapper(base_env)

    # Load Normalization Stats
    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    vn_path = checkpoint_path.parent / "vecnormalize_final.pkl"

    if vn_path.exists():
        print(f"[INFO] Loading VecNormalize stats from: {vn_path}")
        env = VecNormalize.load(str(vn_path), vec_env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[WARN] VecNormalize not found at {vn_path}. Proceeding with raw observations.")
        env = vec_env

    # Load Model
    print(f"[INFO] Loading PPO model: {checkpoint_path}")
    model = PPO.load(
        str(checkpoint_path),
        env=env,
        device=args_cli.device if hasattr(args_cli, "device") else "auto"
    )

    # Evaluation Loop
    print(f"\n[INFO] Starting evaluation: {args_cli.num_episodes} episodes, max {args_cli.max_steps} steps.")
    
    obs = env.reset()
    episode_returns = np.zeros(args_cli.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args_cli.num_envs, dtype=np.int32)
    total_finished = 0

    while total_finished < args_cli.num_episodes:
        actions, _ = model.predict(obs, deterministic=True)

        if np.any(~np.isfinite(obs)):
            print("[ERROR] NaN/Inf detected in observations. Terminating.")
            break

        obs, rewards, dones, infos = env.step(actions)
        episode_returns += rewards
        episode_lengths += 1

        for i in range(args_cli.num_envs):
            if dones[i]:
                total_finished += 1
                print(f"Episode {total_finished:>4} | Env {i:>2} | Return: {episode_returns[i]:>8.2f} | Length: {episode_lengths[i]:>5}")
                
                episode_returns[i] = 0.0
                episode_lengths[i] = 0

                if total_finished >= args_cli.num_episodes:
                    break
        
        # Guard for max steps mismatch
        if np.any(episode_lengths >= args_cli.max_steps):
            idxs = np.where(episode_lengths >= args_cli.max_steps)[0]
            for i in idxs:
                print(f"[WARN] Env {i} reached max_steps without termination.")

    print("\n[INFO] Evaluation complete.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
