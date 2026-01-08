# one_g1_env_cfg.py
#
# Custom G1 locomotion configurations (flat + rough) with specialized reward shaping.
# Designed for manager-based velocity tasks.

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .rough_env_cfg import G1RoughEnvCfg
from .flat_env_cfg import G1FlatEnvCfg

import torch

def safe_set_reward_weight(rewards_cfg, attr_name, weight):
    """Safely updates a reward term weight."""
    if hasattr(rewards_cfg, attr_name):
        reward_term = getattr(rewards_cfg, attr_name)
        if reward_term is not None and hasattr(reward_term, "weight"):
            reward_term.weight = weight
            return True
    return False

# -----------------------------------------------------------------------------
# Custom Reward Functions
# -----------------------------------------------------------------------------

def is_alive(env):
    return 1.0 - mdp.is_terminated(env)

def feet_symmetry_forward(env):
    """Penalize asymmetric forward placement of feet relative to pelvis."""
    robot = env.scene["robot"]
    body_state = robot.data.body_state_w
    
    # Indices: Pelvis=0, Left=24, Right=25 (verify with your asset)
    x_pelvis = body_state[:, 0, 0]
    xL_rel = body_state[:, 24, 0] - x_pelvis
    xR_rel = body_state[:, 25, 0] - x_pelvis

    return (xL_rel + xR_rel) ** 2

def leg_spread_penalty(env):
    """Penalize lateral leg spread distance."""
    robot = env.scene["robot"]
    yL = robot.data.body_state_w[:, 24, 1]
    yR = robot.data.body_state_w[:, 25, 1]
    spread = torch.abs(yL - yR)
    return spread * spread

def hip_symmetry(env, asset_cfg):
    """Penalize asymmetric hip joint angles."""
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    
    # Indices: Left hip=[0,1,2], Right hip=[3,4,5]
    diff = joint_pos[:, 0:3] - joint_pos[:, 3:6]
    return torch.sum(torch.abs(diff), dim=1)

def foot_landing_symmetry(env):
    """Penalize asymmetry in foot landing positions relative to pelvis."""
    robot = env.scene["robot"]
    contact = env.scene["contact_forces"]
    
    body_state = robot.data.body_state_w
    air_time = contact.data.last_air_time
    num_envs, device = body_state.shape[0], body_state.device

    # State allocation
    if not hasattr(env, "_foot_last_landing_x_rel"):
        env._foot_last_landing_x_rel = torch.zeros(num_envs, 2, device=device)
        env._foot_landing_initialized = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)

    x_pelvis = body_state[:, 0, 0]
    x_L_rel = body_state[:, 24, 0] - x_pelvis
    x_R_rel = body_state[:, 25, 0] - x_pelvis

    dt = getattr(env, "step_dt", 0.005)
    left_landing = air_time[:, 24] < (1.5 * dt) # Assuming idx 24 maps to sensor idx
    right_landing = air_time[:, 25] < (1.5 * dt)

    # Note: Ensure sensor indices match body indices or use correct mapping
    # Assuming standard G1 layout where sensor indices align or are mapped.
    # If sensor indices are different (e.g. 6 & 13), use those for `air_time`.
    
    if left_landing.any():
        env._foot_last_landing_x_rel[left_landing, 0] = x_L_rel[left_landing]
        env._foot_landing_initialized[left_landing, 0] = True
    if right_landing.any():
        env._foot_last_landing_x_rel[right_landing, 1] = x_R_rel[right_landing]
        env._foot_landing_initialized[right_landing, 1] = True

    both_inited = env._foot_landing_initialized[:, 0] & env._foot_landing_initialized[:, 1]
    diff = env._foot_last_landing_x_rel[:, 0] - env._foot_last_landing_x_rel[:, 1]
    cost = diff * diff
    cost[~both_inited] = 0.0
    return cost

def air_time_symmetry(env):
    """L1 penalty on air time difference between feet."""
    # Using hardcoded sensor indices 6 (left) and 13 (right)
    air_time = env.scene["contact_forces"].data.last_air_time
    return torch.abs(air_time[:, 6] - air_time[:, 13])

def torso_roll_l2(env):
    """Penalize pelvis roll angle."""
    quat = env.scene["robot"].data.body_state_w[:, 0, 3:7] # w, x, y, z
    w, x, y, z = quat.unbind(dim=1)
    
    # Roll calculation
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    return roll * roll


# -----------------------------------------------------------------------------
# Reward Configurations
# -----------------------------------------------------------------------------

@configclass
class OneG1Rewards(RewardsCfg):
    """Custom reward configuration for improved balance and gait."""

    # -- Task Rewards --
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"command_name": "base_velocity", "std": 0.8})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=5.0, params={"command_name": "base_velocity", "std": 0.5})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)
    alive_bonus = RewTerm(func=is_alive, weight=0.0)

    # -- Stability --
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-7.0, params={"target_height": 0.75, "asset_cfg": SceneEntityCfg("robot", body_names="pelvis")})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})
    torso_roll_l2 = RewTerm(func=torso_roll_l2, weight=-1.0)

    # -- Control Regularization --
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- Gait Shaping --
    feet_air_time = RewTerm(func=mdp.feet_air_time_positive_biped, weight=0.5, params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 0.4})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.2, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")})
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 1.0})

    # -- Posture & Joints --
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])})
    joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint"])})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])})
    dof_pos_limits_arms = RewTerm(func=mdp.joint_pos_limits, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])})
    joint_deviation_torso = RewTerm(func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")})

    # -- Symmetry (Experimental) --
    feet_symmetry_forward_term = RewTerm(func=feet_symmetry_forward, weight=-0.3)
    leg_spread_penalty = RewTerm(func=leg_spread_penalty, weight=-0.5)
    hip_symmetry = RewTerm(func=hip_symmetry, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")})
    air_time_symmetry = RewTerm(func=air_time_symmetry, weight=-4.0) # Disabled by default via weight tuning if needed
    foot_landing_symmetry = RewTerm(func=foot_landing_symmetry, weight=-4.0)


# -----------------------------------------------------------------------------
# Environment Configurations
# -----------------------------------------------------------------------------

@configclass
class OneG1RoughEnvCfg(G1RoughEnvCfg):
    """Rough terrain configuration using custom rewards."""
    rewards: OneG1Rewards = OneG1Rewards()

    def __post_init__(self):
        super().__post_init__()
        
        # Adjust weights for rough terrain
        safe_set_reward_weight(self.rewards, "track_lin_vel_xy_exp", 10.0)
        safe_set_reward_weight(self.rewards, "track_ang_vel_z_exp", 15.0)
        safe_set_reward_weight(self.rewards, "base_height_l2", -3.0)
        safe_set_reward_weight(self.rewards, "feet_air_time", 1.0)
        safe_set_reward_weight(self.rewards, "feet_slide", -0.3)
        safe_set_reward_weight(self.rewards, "dof_pos_limits", -1.0)


@configclass
class OneG1FlatEnvCfg(G1FlatEnvCfg):
    """Flat terrain configuration using custom rewards."""
    rewards: OneG1Rewards = OneG1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # Ensure sufficient episode length
        try:
            self.episode_length_s = 20.0
            if hasattr(self.terminations, "time_out"):
                self.terminations.time_out.time_out = 20.0
        except Exception as e:
            print(f"[WARN] Failed to set episode timeout: {e}")
