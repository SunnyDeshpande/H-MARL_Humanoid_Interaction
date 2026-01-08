#!/usr/bin/env python

"""
Two-G1 meetup play script with split observations:

- Env:    Isaac-TwoG1-Flat-Sunny-v0  (SunnyTwoG1FlatEnvCfg)
- Robot0: driven by PPO (Sunny's walking policy)
- Robot1: driven by PPO (same weights, independent obs)
- Base vels: driven by meetup PPO (2D holonomic policy), *separate*
             commands for each robot via:
               - commands.base_velocity      -> Robot
               - commands.base_velocity_1    -> Robot_1
"""

import argparse
import signal

from isaaclab.app import AppLauncher

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Two G1 meetup play script (split obs).")

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-TwoG1-Flat-Sunny-v0",
    help="Gym task id (two-G1 env).",
)

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed.",
)

parser.add_argument(
    "--g1_checkpoint",
    type=str,
    required=True,
    help="Path to Sunny's G1 PPO checkpoint (.zip).",
)

parser.add_argument(
    "--g1_vecnorm",
    type=str,
    required=True,
    help="Path to VecNormalize stats (.pkl) matching the G1 checkpoint.",
)

parser.add_argument(
    "--meetup_checkpoint",
    type=str,
    required=True,
    help="Path to meetup PPO checkpoint (.zip).",
)

parser.add_argument(
    "--max_steps",
    type=int,
    default=5000,
    help="Max steps for this rollout.",
)

# AppLauncher adds --device, --headless, etc.
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# -------------------------------------------------------------------------
# Launch Omniverse BEFORE importing isaaclab_tasks / managers
# -------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def handle_sigint(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, handle_sigint)

# -------------------------------------------------------------------------
# Heavy imports AFTER SimulationApp is up
# -------------------------------------------------------------------------
import math
import logging

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)

# Import the two-G1 env cfg so we can pass it explicitly
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.two_g1_task_cfg import (
    SunnyTwoG1FlatEnvCfg,
)


# -------------------------------------------------------------------------
# Utility: load VecNormalize stats in a dummy env
# -------------------------------------------------------------------------
def load_vecnorm_stats(vecnorm_path: str, obs_dim: int = 123) -> VecNormalize:
    """
    Load VecNormalize stats from disk using a dummy VecEnv of the right obs dim.
    We do NOT wrap the Isaac env with this; we just call normalize_obs() manually.
    """

    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class DummySingleObsEnv(gym.Env):
        def __init__(self, dim):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-1e10, high=1e10, shape=(dim,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return (
                self.observation_space.sample(),
                0.0,
                False,
                False,
                {},
            )

    dummy_env = DummyVecEnv([lambda: DummySingleObsEnv(obs_dim)])
    vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    print(f"[INFO] VecNormalize stats loaded from: {vecnorm_path}")
    return vecnorm


# -------------------------------------------------------------------------
# Meetup obs projector (2D kinematics)
# -------------------------------------------------------------------------
def angle_diff(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return diff


class MeetupObsProjector:
    def __init__(self):
        self.room_half_size = 5.0
        self.max_speed = 1.5
        self.max_yaw_rate = 2.0
        self.robot_radius = 0.5
        self.target_dist = 2 * self.robot_radius - 0.1

    def build_obs(self, pose1, pose2) -> np.ndarray:
        x1, y1, theta1, vx1, vy1, omega1 = pose1
        x2, y2, theta2, vx2, vy2, omega2 = pose2

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        heading_to_other = math.atan2(dy, dx)
        rel_heading_1 = angle_diff(heading_to_other, theta1)

        heading_to_me = math.atan2(-dy, -dx)
        rel_heading_2 = angle_diff(heading_to_me, theta2)

        obs = np.array(
            [
                np.clip(dx / self.room_half_size, -1.0, 1.0),
                np.clip(dy / self.room_half_size, -1.0, 1.0),
                np.clip(rel_heading_1 / math.pi, -1.0, 1.0),
                np.clip(rel_heading_2 / math.pi, -1.0, 1.0),
                np.clip(
                    (dist - self.target_dist) / self.room_half_size, -1.0, 1.0
                ),
                np.clip(vx1 / self.max_speed, -1.0, 1.0),
                np.clip(vy1 / self.max_speed, -1.0, 1.0),
                np.clip(omega1 / self.max_yaw_rate, -1.0, 1.0),
                np.clip(vx2 / self.max_speed, -1.0, 1.0),
                np.clip(vy2 / self.max_speed, -1.0, 1.0),
                np.clip(omega2 / self.max_yaw_rate, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs


# -------------------------------------------------------------------------
# Isaac helpers: unwrap env and extract kinematics
# -------------------------------------------------------------------------
def unwrap_env(e):
    """Strip gym wrappers until we reach ManagerBasedRLEnv."""
    while hasattr(e, "env"):
        e = e.env
    return e


def quat_to_yaw(q: torch.Tensor) -> float:
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return float(yaw.item())


def quat_to_rot_mat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    # standard quaternion → rotation formula
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = torch.empty((3, 3), device=q.device, dtype=q.dtype)
    R[0, 0] = ww + xx - yy - zz
    R[0, 1] = 2 * (xy - wz)
    R[0, 2] = 2 * (xz + wy)
    R[1, 0] = 2 * (xy + wz)
    R[1, 1] = ww - xx + yy - zz
    R[1, 2] = 2 * (yz - wx)
    R[2, 0] = 2 * (xz - wy)
    R[2, 1] = 2 * (yz + wx)
    R[2, 2] = ww - xx - yy + zz
    return R


def extract_base_pose_vel(robot, pelvis_idx: int = 0):
    """
    Return (x, y, yaw, vx_world, vy_world, wz_world) for the pelvis.
    """
    state = robot.data.body_state_w  # (num_envs, num_bodies, 13)
    base = state[0, pelvis_idx]
    pos = base[0:3]
    quat = base[3:7]
    lin_vel = base[7:10]
    ang_vel = base[10:13]

    x = float(pos[0].item())
    y = float(pos[1].item())
    yaw = quat_to_yaw(quat)
    vx = float(lin_vel[0].item())
    vy = float(lin_vel[1].item())
    wz = float(ang_vel[2].item())
    return x, y, yaw, vx, vy, wz


def project_gravity_to_base(base_quat: torch.Tensor) -> np.ndarray:
    """
    Approximate 'projected_gravity' term:
        gravity vector expressed in base frame.
    """
    w, x, y, z = base_quat

    g_world = torch.tensor([0.0, 0.0, -1.0], device=base_quat.device)
    q_conj = torch.tensor([w, -x, -y, -z], device=base_quat.device)

    def qmul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return torch.tensor(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            device=base_quat.device,
        )

    v = torch.cat([torch.zeros(1, device=base_quat.device), g_world])
    qgv = qmul(q_conj, qmul(v, base_quat))
    g_body = qgv[1:]
    return g_body.detach().cpu().numpy().astype(np.float32)


def world_to_yaw(vx_world: float, vy_world: float, yaw: float):
    """
    Rotate world-frame velocities into the robot's yaw frame.

    yaw-frame = R(-yaw) * world
    R(-yaw) = [[ cos(yaw),  sin(yaw)],
               [-sin(yaw),  cos(yaw)]]
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_yaw = c * vx_world + s * vy_world
    vy_yaw = -s * vx_world + c * vy_world
    return vx_yaw, vy_yaw


def build_single_robot_obs(
    isaac_env,
    robot,
    last_action: np.ndarray,
    cmd_term_name: str,
) -> np.ndarray:
    """
    Build a 123-dim obs vector in the same structure as the single-G1 env:
        [ base_lin_vel(3)   -- in BASE frame
          base_ang_vel(3)   -- in BASE frame
          projected_gravity(3),
          velocity_commands(3),
          joint_pos_rel(37),
          joint_vel(37),
          actions(37) ]
    """
    # Base state
    body_state = robot.data.body_state_w[0, 0]  # (13,)
    lin_vel_world = body_state[7:10]  # world
    ang_vel_world = body_state[10:13]
    quat = body_state[3:7]

    # Transform velocities into BASE frame (approx matches mdp.base_*_vel)
    R = quat_to_rot_mat(quat)
    lin_vel_body = R.T @ lin_vel_world
    ang_vel_body = R.T @ ang_vel_world

    base_lin_vel = lin_vel_body.detach().cpu().numpy().astype(np.float32)
    base_ang_vel = ang_vel_body.detach().cpu().numpy().astype(np.float32)
    proj_gravity = project_gravity_to_base(quat)

    # Command term: 'base_velocity' or 'base_velocity_1'
    cmd_mgr = getattr(isaac_env, "command_manager", None)
    if cmd_mgr is None:
        raise RuntimeError("Isaac env has no command_manager; cannot build obs.")

    term = cmd_mgr.get_term(cmd_term_name)

    cmd = None
    for attr in ["command", "commands", "target", "targets"]:
        if hasattr(term, attr):
            buf = getattr(term, attr)
            if isinstance(buf, torch.Tensor) and buf.ndim >= 2 and buf.shape[1] >= 3:
                cmd = buf[0, 0:3].detach().cpu().numpy().astype(np.float32)
                break
    if cmd is None:
        cmd = np.zeros(3, dtype=np.float32)

    # Joints: use RELATIVE positions (joint_pos - default_pos) to match mdp.joint_pos_rel
    joint_pos = robot.data.joint_pos[0]
    joint_vel = robot.data.joint_vel[0]

    if hasattr(robot.data, "default_joint_pos") and robot.data.default_joint_pos is not None:
        default_pos = robot.data.default_joint_pos[0]
        joint_pos_rel = (joint_pos - default_pos).detach().cpu().numpy().astype(np.float32)
    else:
        joint_pos_rel = joint_pos.detach().cpu().numpy().astype(np.float32)

    joint_vel_np = joint_vel.detach().cpu().numpy().astype(np.float32)

    if last_action is None:
        last_action = np.zeros_like(joint_pos_rel, dtype=np.float32)

    obs = np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            proj_gravity,
            cmd,
            joint_pos_rel,
            joint_vel_np,
            last_action.astype(np.float32),
        ],
        axis=0,
    )
    assert obs.shape[0] == 123, f"Expected 123-dim obs, got {obs.shape}"
    return obs[None, :]  # (1, 123)


def sync_robot1_from_robot0(robot0, robot1):
    """
    Clone ONLY joint states from robot0 to robot1 at reset.

    Root poses (positions + yaw) are left to the env cfg, so the robots
    stay separated in space and don't get stacked on top of each other.
    """
    joint_pos0 = robot0.data.joint_pos.clone()
    joint_vel0 = robot0.data.joint_vel.clone()
    robot1.write_joint_state_to_sim(joint_pos0, joint_vel0)

    print("[SYNC] Robot1 joint state cloned from Robot0 (joints only)")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    # Seeds
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # ---------------------------------------------------------------------
    # Create Isaac env directly (no SB3 wrappers)
    # ---------------------------------------------------------------------
    env_cfg = SunnyTwoG1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed

    if hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    print(f"[INFO] Creating Isaac env: {args_cli.task}")
    base_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    # unwrap to get the actual ManagerBasedRLEnv and its device
    isaac_env = unwrap_env(base_env)
    device = isaac_env.device
    print("[INFO] Raw Isaac env:", type(isaac_env))
    print(f"[INFO] Isaac env device: {device}")
    print("[INFO] Env action space:", isaac_env.action_space)

    # ---------------------------------------------------------------------
    # Grab robot handles
    # ---------------------------------------------------------------------
    scene = isaac_env.scene
    try:
        robot0 = scene["robot"]
        robot1 = scene["robot_1"]
    except KeyError as e:
        raise RuntimeError(
            "Expected scene['robot'] and scene['robot_1'] in SunnyTwoG1FlatEnvCfg."
        ) from e

    # Command terms (we assume you added base_velocity_1 in the cfg)
    cmd_mgr = isaac_env.command_manager
    term0 = cmd_mgr.get_term("base_velocity")
    term1 = cmd_mgr.get_term("base_velocity_1")

    # ---------------------------------------------------------------------
    # Load VecNormalize stats
    # ---------------------------------------------------------------------
    vecnorm = load_vecnorm_stats(args_cli.g1_vecnorm, obs_dim=123)

    # ---------------------------------------------------------------------
    # Load G1 PPO (once) and meetup PPO
    # ---------------------------------------------------------------------
    print(f"[INFO] Loading G1 PPO checkpoint from: {args_cli.g1_checkpoint}")
    model_g1 = PPO.load(
        args_cli.g1_checkpoint,
        device=args_cli.device if hasattr(args_cli, "device") else "auto",
        print_system_info=True,
    )
    print("[INFO] G1 PPO loaded.")

    print(f"[INFO] Loading meetup PPO checkpoint from: {args_cli.meetup_checkpoint}")
    model_meetup = PPO.load(
        args_cli.meetup_checkpoint,
        device=args_cli.device if hasattr(args_cli, "device") else "auto",
    )
    print("[INFO] Meetup PPO loaded.")

    meetup_proj = MeetupObsProjector()

    # ---------------------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------------------
    obs_env, _ = base_env.reset()

    # After reset, sync robot1 to robot0
    sync_robot1_from_robot0(robot0, robot1)

    step_count = 0
    last_action_r0 = None
    last_action_r1 = None

    print("[INFO] Starting two-G1 meetup rollout...")

    # stabilization steps
    for stab_step in range(150):
    # Write zero commands to the command terms
        if hasattr(term0, "command"):
            term0.command[0, 0] = 0.0
            term0.command[0, 1] = 0.0
            term0.command[0, 2] = 0.0
        if hasattr(term1, "command"):
            term1.command[0, 0] = 0.0
            term1.command[0, 1] = 0.0
            term1.command[0, 2] = 0.0
        
        # Build obs for each robot
        obs_r0_raw = build_single_robot_obs(isaac_env, robot0, None, "base_velocity")
        obs_r1_raw = build_single_robot_obs(isaac_env, robot1, None, "base_velocity_1")
        
        # Normalize
        obs_r0_norm = vecnorm.normalize_obs(obs_r0_raw.copy())
        obs_r1_norm = vecnorm.normalize_obs(obs_r1_raw.copy())
        
        # Get actions
        a0, _ = model_g1.predict(obs_r0_norm, deterministic=True)
        a1, _ = model_g1.predict(obs_r1_norm, deterministic=True)
        
        last_action_r0 = a0[0].copy()
        last_action_r1 = a1[0].copy()
        
        # Execute
        joint_actions_np = np.concatenate([a0, a1], axis=1)
        joint_actions = torch.as_tensor(joint_actions_np, device=device, dtype=torch.float32)
        obs_env, _, _, _, _ = base_env.step(joint_actions)


    try:
        while step_count < args_cli.max_steps:
            # --- Extract base poses for meetup vel policy ---
            x0, y0, yaw0, vx0_w, vy0_w, wz0 = extract_base_pose_vel(robot0, pelvis_idx=0)
            x1, y1, yaw1, vx1_w, vy1_w, wz1 = extract_base_pose_vel(robot1, pelvis_idx=0)

            pose0 = (x0, y0, yaw0, vx0_w, vy0_w, wz0)  # ← Use actual wz0
            pose1 = (x1, y1, yaw1, vx1_w, vy1_w, wz1)  # ← Use actual wz1
            
            meetup_obs = meetup_proj.build_obs(pose0, pose1)
            meetup_obs = meetup_obs[None, :]  # (1, 11)
            a_meetup, _ = model_meetup.predict(meetup_obs, deterministic=True)

            # a_meetup: [v1_raw, w1_raw, v2_raw, w2_raw] from your 2D env
            # v*_raw: forward speed (in robot's heading direction)
            # w*_raw: yaw rate
            v1_raw, w1_raw, v2_raw, w2_raw = a_meetup[0]

            # Scaling factors
            LIN_SCALE_MEETUP = 1.0
            YAW_SCALE_MEETUP = 1.5

            # G1's training command ranges
            G1_LIN_VEL_MIN = -0.5
            G1_LIN_VEL_MAX = 1.0

            # ----- Robot 0 -----
            # v1_raw is forward speed in robot 0's heading direction
            # Convert to world-frame x,y by rotating by robot's yaw
            vx0_world = v1_raw * LIN_SCALE_MEETUP * np.cos(yaw0)
            vy0_world = v1_raw * LIN_SCALE_MEETUP * np.sin(yaw0)
            w0_world = w1_raw * YAW_SCALE_MEETUP

            # Now convert world-frame to yaw-frame (what G1 PPO expects in commands)
            vx0_yaw, vy0_yaw = world_to_yaw(vx0_world, vy0_world, yaw0)

            # Map to G1's expected command range [-0.5, 1.0]
            vx0_cmd = float(np.interp(vx0_yaw, [-1.5, 1.5], [G1_LIN_VEL_MIN, G1_LIN_VEL_MAX]))
            vy0_cmd = float(np.clip(vy0_yaw, -0.2, 0.2))  # G1 doesn't do lateral
            w0_cmd = float(np.clip(w0_world, -1.5, 1.5))

            # ----- Robot 1 -----
            # Same process for robot 1
            vx1_world = v2_raw * LIN_SCALE_MEETUP * np.cos(yaw1)
            vy1_world = v2_raw * LIN_SCALE_MEETUP * np.sin(yaw1)
            w1_world = w2_raw * YAW_SCALE_MEETUP

            vx1_yaw, vy1_yaw = world_to_yaw(vx1_world, vy1_world, yaw1)

            vx1_cmd = float(np.interp(vx1_yaw, [-1.5, 1.5], [G1_LIN_VEL_MIN, G1_LIN_VEL_MAX]))
            vy1_cmd = float(np.clip(vy1_yaw, -0.2, 0.2))
            w1_cmd = float(np.clip(w1_world, -1.5, 1.5))

            # Write into the two command terms' buffers
            if hasattr(term0, "command"):
                term0.command[0, 0] = vx0_cmd
                term0.command[0, 1] = vy0_cmd
                term0.command[0, 2] = w0_cmd

            if hasattr(term1, "command"):
                term1.command[0, 0] = vx1_cmd
                term1.command[0, 1] = vy1_cmd
                term1.command[0, 2] = w1_cmd

            # Log commands occasionally
            if step_count % 50 == 0:
                print(
                    f"[STEP {step_count:05d}] "
                    f"R0: yaw={yaw0:+.3f} | meetup_v={v1_raw:+.3f} | world=({vx0_world:+.3f},{vy0_world:+.3f}) | "
                    f"cmd=({vx0_cmd:+.3f},{vy0_cmd:+.3f},{w0_cmd:+.3f}) | "
                    f"R1: yaw={yaw1:+.3f} | meetup_v={v2_raw:+.3f} | world=({vx1_world:+.3f},{vy1_world:+.3f}) | "
                    f"cmd=({vx1_cmd:+.3f},{vy1_cmd:+.3f},{w1_cmd:+.3f})"
                )


            # --- Build obs for each robot (123-dim each) ---
            obs_r0_raw = build_single_robot_obs(
                isaac_env, robot0, last_action_r0, "base_velocity"
            )
            obs_r1_raw = build_single_robot_obs(
                isaac_env, robot1, last_action_r1, "base_velocity_1"
            )

            # Normalize with VecNormalize stats
            obs_r0_norm = vecnorm.normalize_obs(obs_r0_raw.copy())
            obs_r1_norm = vecnorm.normalize_obs(obs_r1_raw.copy())

            # --- Run the same PPO twice ---
            a0, _ = model_g1.predict(obs_r0_norm, deterministic=True)
            a1, _ = model_g1.predict(obs_r1_norm, deterministic=True)

            last_action_r0 = a0[0].copy()
            last_action_r1 = a1[0].copy()

            # --- Concatenate actions for the env (74-dim, numpy) ---
            joint_actions_np = np.concatenate([a0, a1], axis=1)  # (1, 74)

            # Convert to torch on the Isaac env's device
            joint_actions = torch.as_tensor(
                joint_actions_np,
                device=device,
                dtype=torch.float32,
            )

            obs_env, reward, terminated, truncated, info = base_env.step(joint_actions)

            done = bool(terminated) or bool(truncated)
            if done:
                print(f"[INFO] Episode done at step {step_count}, resetting env.")
                obs_env, _ = base_env.reset()
                sync_robot1_from_robot0(robot0, robot1)
                last_action_r0 = None
                last_action_r1 = None

            step_count += 1

        print("[INFO] Rollout complete.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    base_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
