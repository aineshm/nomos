"""WheeledLab / Isaac physics demo pipeline (the cinematic hero shot).

Architecture (from the project plan):
    trained high-level coordination policy  (kinematic env, Modal)
        -> per-step SETPOINTS (target waypoint / velocity / heading)
        -> FROZEN low-level WheeledLab controller (Isaac, single-agent, trained once)
        -> rigid-body car physics + RTX render

This package has two halves:
  * export_setpoints.py — runs ON THIS MACHINE (no GPU): replays a checkpoint and
    dumps the setpoint stream the low-level controller will track.
  * run_isaac_demo.py   — runs ON A GPU/RTX BOX: loads Isaac Sim + WheeledLab,
    spawns rigid-body cars, tracks the setpoints, renders. Stubbed with a real
    preflight check so it fails loudly-and-clearly off-GPU rather than silently.
"""
