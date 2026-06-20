# Isaac / WheeledLab physics demo (cinematic upgrade)

The **rigid-body** hero shot: the same trained coordination policy that drives the
web viewer, replayed with real car physics in Isaac Sim using WheeledLab's car
asset and a frozen low-level controller.

```
trained coordination policy  ──setpoints──▶  frozen WheeledLab low-level
(kinematic env / Modal)      (waypoint/vel/  controller (Isaac, trained once)
                              heading)        │
                                              ▼
                                  steering+throttle ─▶ rigid-body car ─▶ RTX frame
```

This is the **gated** path — it needs a GPU/RTX box. The
[`web/`](../web) deck.gl viewer is the always-works demo; build/show that first.

## Two halves

| step | where it runs | command |
|---|---|---|
| 1. export setpoints | **any machine** (no GPU) | `python -m smoothride.demo.isaac.export_setpoints --ckpt runs/trained.msgpack` |
| 2. render in Isaac  | **RTX/L40S box** | `python -m smoothride.demo.isaac.run_isaac_demo --setpoints runs/isaac/setpoints.npz --controller runs/lowlevel/controller.pt` |

Step 1 works here today. Step 2 runs a **preflight check** and, off-GPU, prints
exactly what's missing and what it would do — instead of a confusing crash.

## Hardware / software (the gated box)

- NVIDIA RTX or L40S GPU **with RT Cores** (for the path-traced render)
- Isaac Sim **v4.5.0**, IsaacLab **v2.0.2** (the versions WheeledLab pins)
- WheeledLab (BSD-3): the car asset (MuSHR / HOUND) + its dynamics/DR recipe
- A trained-and-**frozen** low-level controller checkpoint (`--controller`)

## Status: structured stub

`run_isaac_demo.py` lays out the full orchestration; the Isaac-specific calls are
marked `TODO(isaac)` and implemented against your installed Isaac/WheeledLab on
the box. The setpoint contract (`export_setpoints.py`), the georef metadata
(origin + CRS travel with the `.npz`), and `config.yaml` are all real now, so the
remaining work is wiring the TODOs — not redesigning the handoff.

## Why this split is safe (no body-transfer problem)

The low-level controller is trained **once** (single-agent) and then **frozen**;
it is never part of the multi-agent coordination training. We learn coordination
*on top of* a fixed locomotion skill, so the policy that flows smoothly in the
kinematic env also flows when its setpoints are tracked by real car physics —
keep the learned driving gentle (the kinematic↔physics gap grows with aggressive
maneuvers) and the setpoints track well.
