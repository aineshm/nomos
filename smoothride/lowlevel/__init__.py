"""Low-level locomotion controller — trained ONCE, then FROZEN.

A single-agent command-tracking policy that turns the high-level coordination
setpoints (target velocity / heading) into steering + throttle for a rigid-body
WheeledLab car in Isaac. It is never part of the multi-agent coordination
training loop, so there is no body-transfer problem: coordination is learned on
top of a fixed locomotion skill.

Layout (an Isaac Lab "external task" extension, launched on Modal GPU):
    modal_image.py        Modal App: Isaac Sim + IsaacLab + WheeledLab image, GPU, Volume
    train_isaac.py        Modal entrypoint -> Isaac Lab + RSL-RL headless training
    isaac_task/           the new command-tracking task (env cfg, car cfg, RSL-RL cfg)
    export.py             frozen checkpoint -> controller.pt (TorchScript) for the demo

Nothing here imports Isaac at module load (Isaac only exists in the Modal
container), so `modal run` works from a plain laptop. The contract the rest of
the project depends on is in isaac_task/track_env_cfg.py: obs/action layout +
the setpoint definition.
"""
