"""Train the frozen low-level controller on Modal GPU, headless in Isaac Lab.

Run from a laptop (only `modal` needed locally):
    modal run smoothride/lowlevel/train_isaac.py            # full training
    modal run smoothride/lowlevel/train_isaac.py --iters 50 # quick smoke

The heavy imports (Isaac Sim / Isaac Lab / RSL-RL) happen INSIDE the remote
function, where the container provides them. Checkpoints land in the Modal Volume
at /ckpts/<experiment>/ — `export.py` reads from there.
"""
from __future__ import annotations

from .modal_image import CKPT_DIR, app, gpu_function, volume


@gpu_function()
def train(iters: int | None = None, num_envs: int = 4096, seed: int = 0):
    """Headless PhysX + RSL-RL PPO training of the command-tracking policy."""
    # 1) boot Isaac Sim headless (no RTX) BEFORE importing anything omni/isaaclab
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=False)
    simulation_app = app_launcher.app

    # 2) now Isaac Lab + RSL-RL + our task are importable
    import os

    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from smoothride.lowlevel.isaac_task import TASK_ID  # registers the gym id
    from smoothride.lowlevel.isaac_task.agents.rsl_rl_cfg import CarTrackPPORunnerCfg
    from smoothride.lowlevel.isaac_task.track_env_cfg import CarTrackEnvCfg

    env_cfg = CarTrackEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    runner_cfg = CarTrackPPORunnerCfg()
    if iters is not None:
        runner_cfg.max_iterations = iters

    log_dir = os.path.join(CKPT_DIR, runner_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    print(f"[train] task={TASK_ID} envs={num_envs} iters={runner_cfg.max_iterations}")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations)

    # persist to the Volume so export.py / later runs can read it
    runner.save(os.path.join(log_dir, "model_final.pt"))
    volume.commit()
    env.close()
    simulation_app.close()
    print(f"[train] saved -> {log_dir}/model_final.pt")
    return log_dir


@app.local_entrypoint()
def main(iters: int | None = None, num_envs: int = 4096, seed: int = 0):
    log_dir = train.remote(iters=iters, num_envs=num_envs, seed=seed)
    print(f"done. checkpoints in Volume: {log_dir}")
    print("export the frozen controller with:  modal run smoothride/lowlevel/export.py")
