# two_g1_task_cfg.py
#
# Custom Two-G1 flat environment configuration.
#
#  - scene.robot    : First G1 robot (standard)
#  - scene.robot_1  : Second G1 robot (offset in +x)
#
#  Action Space (74D):
#  - actions.joint_pos     : 37 DOFs for robot
#  - actions.joint_pos_r1  : 37 DOFs for robot_1
#
#  Command Space:
#  - commands.base_velocity      : Velocity command for robot
#  - commands.base_velocity_1    : Velocity command for robot_1

from dataclasses import MISSING
import numpy as np

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    MySceneCfg,
    ActionsCfg,
)
# Inherit from your single-robot flat config
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.one_g1_env_cfg import (
    OneG1FlatEnvCfg,
)


def rpy_deg_to_quat_rad(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple:
    """Convert RPY (degrees) to Quaternion (radians)."""
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (w, x, y, z)


@configclass
class TwoG1SceneCfg(MySceneCfg):
    """Scene configuration with two G1 robots."""
    robot_1: ArticulationCfg = MISSING


@configclass
class TwoG1ActionsCfg(ActionsCfg):
    """Action configuration for two robots."""
    # Inherits `joint_pos` for the first robot
    joint_pos_r1 = mdp.JointPositionActionCfg(
        asset_name="robot_1",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class TwoG1CommandsCfg:
    """Command specifications for two G1 robots."""
    
    # Common velocity command parameters
    _VELOCITY_CMD_KWARGS = dict(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.5, 1.5),
            heading=(0.0, 0.0),
        ),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(asset_name="robot", **_VELOCITY_CMD_KWARGS)
    base_velocity_1 = mdp.UniformVelocityCommandCfg(asset_name="robot_1", **_VELOCITY_CMD_KWARGS)


@configclass
class TwoG1FlatEnvCfg(OneG1FlatEnvCfg):
    """Two-G1 flat locomotion environment."""

    scene: TwoG1SceneCfg = TwoG1SceneCfg(num_envs=4096, env_spacing=4.0)
    actions: TwoG1ActionsCfg = TwoG1ActionsCfg()
    commands: TwoG1CommandsCfg = TwoG1CommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Clone robot config for the second agent
        base_robot_cfg = self.scene.robot
        
        # Configure Robot 1 (the original one)
        base_robot_cfg.init_state.rot = rpy_deg_to_quat_rad(0.0, 0.0, 45.0)

        # Configure Robot 2 (the copy)
        robot1_cfg = base_robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
        
        z0 = base_robot_cfg.init_state.pos[2] if getattr(base_robot_cfg, "init_state", None) else 1.0
        robot1_cfg.init_state.pos = (4.0, 3.0, z0)
        
        # Face the first robot (yaw = 180 deg)
        robot1_cfg.init_state.rot = rpy_deg_to_quat_rad(0.0, 0.0, -180.0)

        self.scene.robot_1 = robot1_cfg

        # Interactive play adjustments
        self.scene.num_envs = 1024
        self.scene.env_spacing = 4.0
