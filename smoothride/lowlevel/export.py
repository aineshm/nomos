"""Freeze the trained controller to a portable TorchScript `controller.pt`.

Runs on Modal (reads the checkpoint from the Volume), rebuilds the actor, exports
the deterministic mean-action path as JIT TorchScript, and downloads it locally.
The result is GPU-free: one forward pass per car per control step at demo time.

    modal run smoothride/lowlevel/export.py        # -> runs/lowlevel/controller.pt

Inference contract (must match isaac_task/track_env_cfg.py):
    input : obs  (B, num_obs)  — base lin/ang vel, gravity, setpoint, steer, last action
    output: act  (B, 2)        — [drive throttle, steer angle], normalized
"""
from __future__ import annotations

import os

from .modal_image import CKPT_DIR, app, gpu_function

EXPERIMENT = "wheeledlab_track"
LOCAL_OUT = "runs/lowlevel/controller.pt"


@gpu_function()
def export_jit() -> bytes:
    import torch
    from rsl_rl.modules import ActorCritic

    from smoothride.lowlevel.isaac_task.agents.rsl_rl_cfg import CarTrackPPORunnerCfg

    ckpt = os.path.join(CKPT_DIR, EXPERIMENT, "model_final.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"{ckpt} not found — run train_isaac.py first")

    state = torch.load(ckpt, map_location="cpu")["model_state_dict"]
    # infer obs/action dims from the actor's first/last linear layers (version-robust)
    actor_w = [v for k, v in state.items() if k.startswith("actor.") and v.ndim == 2]
    num_obs = actor_w[0].shape[1]
    num_actions = actor_w[-1].shape[0]

    pcfg = CarTrackPPORunnerCfg().policy
    ac = ActorCritic(
        num_actor_obs=num_obs, num_critic_obs=num_obs, num_actions=num_actions,
        actor_hidden_dims=pcfg.actor_hidden_dims,
        critic_hidden_dims=pcfg.critic_hidden_dims,
        activation=pcfg.activation, init_noise_std=pcfg.init_noise_std,
    )
    ac.load_state_dict(state)
    ac.eval()

    # deterministic policy = the actor MLP's mean output; wrap + script it
    class Controller(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            return self.actor(obs)

    scripted = torch.jit.script(Controller(ac.actor))
    out = os.path.join(CKPT_DIR, EXPERIMENT, "controller.pt")
    scripted.save(out)
    print(f"[export] obs={num_obs} act={num_actions} -> {out}")
    with open(out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    blob = export_jit.remote()
    os.makedirs(os.path.dirname(LOCAL_OUT), exist_ok=True)
    with open(LOCAL_OUT, "wb") as f:
        f.write(blob)
    print(f"frozen controller -> {LOCAL_OUT} ({len(blob) / 1024:.0f} KB)")
    print("consumed by smoothride/demo/isaac/run_isaac_demo.py via --controller")
