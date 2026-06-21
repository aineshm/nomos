"""Isaac Lab external task: WheeledLab car (velocity, heading) command tracking.

Importing this module registers the Gym env id so the standard Isaac Lab / RSL-RL
training scripts can pick it up with `--task Nomos-WheeledLab-Track-v0`.
Isaac is only available inside the Modal container, so the registration is guarded
— importing this on a plain laptop is a no-op instead of an ImportError.
"""
from __future__ import annotations

TASK_ID = "Nomos-WheeledLab-Track-v0"

try:
    import gymnasium as gym

    from .agents.rsl_rl_cfg import CarTrackPPORunnerCfg
    from .track_env_cfg import CarTrackEnvCfg

    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": CarTrackEnvCfg,
            "rsl_rl_cfg_entry_point": CarTrackPPORunnerCfg,
        },
    )
except ImportError:
    # Isaac / gym not present (e.g. this laptop) — registration is a no-op.
    pass
