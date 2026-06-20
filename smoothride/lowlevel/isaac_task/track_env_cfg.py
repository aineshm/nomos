"""Command-tracking task: drive the WheeledLab car to track a (velocity, heading)
setpoint — the exact interface the high-level coordination policy emits.

This is a manager-based RL task, structured like Isaac Lab's velocity-locomotion
tasks (ANYmal/velocity) but for an Ackermann car. It IS the contract between the
two policy layers:

    SETPOINT (command)  : target forward speed (m/s) + target heading (rad)
    ACTION   (2)        : [drive throttle, steer angle]   (normalized)
    OBSERVATION         : base lin/ang vel, gravity, the command, steering state,
                          last action  -> a frozen MLP policy after training.

Train with RSL-RL PPO (see agents/rsl_rl_cfg.py), freeze, export to controller.pt.
"""
from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .car_cfg import DRIVE_JOINTS, MAX_STEER_RAD, MAX_WHEEL_RAD_S, STEER_JOINTS, WHEELEDLAB_CAR_CFG


# ----------------------------------------------------------------------- scene
@configclass
class CarSceneCfg(InteractiveSceneCfg):
    # flat ground plane
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    # the car (cloned across num_envs by the scene)
    robot = WHEELEDLAB_CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # dome light (cheap; training is headless / no RTX render)
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2000.0))


# --------------------------------------------------------------------- command
@configclass
class CommandsCfg:
    """The SETPOINT: target forward velocity + a commanded heading.

    UniformVelocityCommand with heading_command=True samples a target speed and a
    target heading and turns the heading error into a yaw-rate target — i.e. the
    car is asked to reach a heading while holding a speed, exactly the high-level
    policy's (velocity, heading) setpoint."""
    setpoint = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 4.0),     # forward speed setpoint (m/s)
            lin_vel_y=(0.0, 0.0),      # nonholonomic car: no lateral command
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# --------------------------------------------------------------------- actions
@configclass
class ActionsCfg:
    drive = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=DRIVE_JOINTS, scale=MAX_WHEEL_RAD_S)
    steer = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=STEER_JOINTS, scale=MAX_STEER_RAD,
        use_default_offset=True)


# ---------------------------------------------------------------- observations
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        setpoint = ObsTerm(func=mdp.generated_commands, params={"command_name": "setpoint"})
        steer_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=STEER_JOINTS)})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -------------------------------------------------------------------- rewards
@configclass
class RewardsCfg:
    # track the commanded forward speed and yaw — the core of the skill
    track_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0,
                            params={"command_name": "setpoint", "std": 0.5})
    track_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.0,
                            params={"command_name": "setpoint", "std": 0.5})
    # keep it gentle so the learned setpoints track well when REPLAYED in the demo
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)        # punish lateral slip
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


# ----------------------------------------------------------------- terminations
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flipped = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)})


# ----------------------------------------------------------- events (domain rand)
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.1), "num_buckets": 64})
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*"),
                "mass_distribution_params": (-0.2, 0.4), "operation": "add"})
    push = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(6.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})
    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


# ----------------------------------------------------------------------- env cfg
@configclass
class CarTrackEnvCfg(ManagerBasedRLEnvCfg):
    scene: CarSceneCfg = CarSceneCfg(num_envs=4096, env_spacing=3.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005               # 200 Hz physics
        self.sim.render_interval = self.decimation
        # headless training on Modal: no RTX, just PhysX
        self.sim.device = "cuda:0"
