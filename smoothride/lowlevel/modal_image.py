"""Modal App + image for headless Isaac Lab training of the low-level controller.

Training the controller needs Isaac Sim's PhysX, but NOT the RTX renderer — so it
runs headless on a datacenter GPU (A100/H100), which Modal offers. (The RTX demo
render is the separate, RT-Cores-gated step; this is just physics + RL.)

Prereqs (one-time):
  * Modal account + `pip install modal`, `modal token new`
  * An NGC API key to pull nvcr.io images, stored as a Modal secret:
        modal secret create ngc NGC_API_KEY=nvapi-xxxxxxxx
  * (optional) Weights & Biases secret `wandb` for logging.

This module ONLY defines infra (app, image, gpu, volume, secrets). The training
body lives in train_isaac.py so Isaac imports stay inside the remote function.
"""
from __future__ import annotations

import modal

APP_NAME = "smoothride-lowlevel"
GPU = "A100"          # H100 also fine; training is PhysX+PPO, no RT Cores needed
TIMEOUT_S = 6 * 60 * 60

app = modal.App(APP_NAME)

# Checkpoints persist here across runs (resume + export read from it).
volume = modal.Volume.from_name("smoothride-lowlevel-ckpts", create_if_missing=True)
CKPT_DIR = "/ckpts"

# Isaac Sim base image (pins match WheeledLab: Isaac Sim 4.5.0 / IsaacLab 2.0.2).
# nvcr.io requires the `ngc` secret (NGC_API_KEY) at build time.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/isaac-sim:4.5.0",
        setup_dockerfile_commands=[
            # the isaac-sim image ships its own python at /isaac-sim/python.sh;
            # expose a `python3` Modal can call.
            "RUN ln -sf /isaac-sim/python.sh /usr/local/bin/python3 || true",
        ],
        add_python=None,
    )
    .env({
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ISAACSIM_PATH": "/isaac-sim",
        "OMNI_KIT_ALLOW_ROOT": "1",
    })
    # IsaacLab (pinned) — installs into the isaac-sim python env.
    .run_commands(
        "git clone --depth 1 --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab",
        "cd /opt/IsaacLab && ./isaaclab.sh --install rsl_rl",
        # WheeledLab: the car asset + dynamics/DR recipe we build the task from.
        # TODO(wheeledlab): pin a release tag once chosen; --install if it ships a setup.
        "git clone --depth 1 https://github.com/UWRobotLearning/WheeledLab.git /opt/WheeledLab",
        "cd /opt/IsaacLab && ./isaaclab.sh -p -m pip install -e /opt/WheeledLab || true",
        secrets=[modal.Secret.from_name("ngc")],
    )
    # our task extension is added at runtime via add_local_python_source (below),
    # so editing the task doesn't trigger an image rebuild.
    .add_local_python_source("smoothride")
)


def gpu_function(**kwargs):
    """Decorator preset for the training/export functions."""
    return app.function(
        image=image,
        gpu=GPU,
        timeout=TIMEOUT_S,
        volumes={CKPT_DIR: volume},
        secrets=[modal.Secret.from_name("ngc")],
        **kwargs,
    )
