"""WheeledLab car as an Isaac Lab Articulation.

WheeledLab's platforms (MuSHR / HOUND) are ~1/10-scale Ackermann-steered RC cars:
two front STEERING joints + (rear or all) DRIVE wheel joints. We wrap the asset
as an `ArticulationCfg` so the task can spawn N of them and command throttle+steer.

TODO(wheeledlab): the exact USD path and joint names come from the WheeledLab
repo cloned into the image at /opt/WheeledLab. Fill `USD_PATH` and the joint
regexes from its asset (grep its *.usd / robot cfg). The values below are the
right SHAPE with placeholder names.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# TODO(wheeledlab): point at the real asset shipped in /opt/WheeledLab.
USD_PATH = "/opt/WheeledLab/wheeledlab_assets/mushr/mushr.usd"

# Joint name patterns (regex). TODO(wheeledlab): confirm against the USD.
STEER_JOINTS = ["steer_.*"]          # front-left / front-right steering
DRIVE_JOINTS = ["wheel_.*"]          # driven wheels
WHEEL_RADIUS = 0.05                  # m, MuSHR ~1/10 scale (TODO confirm)
MAX_STEER_RAD = 0.38                 # ~22 deg lock (TODO confirm)
MAX_WHEEL_RAD_S = 200.0              # caps top speed via wheel radius

WHEELEDLAB_CAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.06),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # position-controlled steering
        "steering": ImplicitActuatorCfg(
            joint_names_expr=STEER_JOINTS,
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=4.0,
        ),
        # velocity-controlled drive wheels (throttle == target wheel speed)
        "drive": ImplicitActuatorCfg(
            joint_names_expr=DRIVE_JOINTS,
            effort_limit=5.0,
            velocity_limit=MAX_WHEEL_RAD_S,
            stiffness=0.0,        # velocity control -> no position stiffness
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
