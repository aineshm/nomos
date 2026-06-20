# Low-level controller — train once on Modal (Isaac Lab + WheeledLab), then freeze

The locomotion skill the demo cars stand on: a single-agent policy that tracks a
**(forward-velocity, heading) setpoint** and emits **[throttle, steer]** for a
rigid-body WheeledLab car. Trained **once**, **frozen**, and never part of the
multi-agent coordination loop — so coordination is learned on top of a fixed
locomotion skill (no body-transfer problem).

Training needs Isaac's PhysX but **not** the RTX renderer, so it runs **headless
on a Modal A100/H100** — which is why "we're not doing WheeledLab *training* on
RT Cores" is true: this is PhysX + PPO, the RT-Cores box is only for the demo
render.

## The contract (what the rest of the project depends on)

Defined in [`isaac_task/track_env_cfg.py`](isaac_task/track_env_cfg.py):

| | |
|---|---|
| **setpoint** (command) | target forward speed (m/s) + target heading (rad) |
| **action** (2) | `[drive throttle, steer angle]`, normalized |
| **observation** | base lin/ang vel, projected gravity, setpoint, steering pos, last action |

The high-level coordination policy already emits waypoint/velocity/heading; that
maps straight onto this setpoint. `export.py` produces `controller.pt` with
exactly this input/output, consumed by
[`../demo/isaac/run_isaac_demo.py`](../demo/isaac/run_isaac_demo.py).

## One-time setup

```bash
pip install -e ".[lowlevel]"          # local: just the Modal client
modal token new                        # auth
modal secret create ngc NGC_API_KEY=nvapi-xxxxxxxx   # to pull nvcr.io/nvidia/isaac-sim
```

## Run

```bash
# quick smoke (small iter count) — first run also builds the ~10GB Isaac image
modal run smoothride/lowlevel/train_isaac.py --iters 50 --num-envs 1024

# full training -> checkpoint in the Modal Volume
modal run smoothride/lowlevel/train_isaac.py

# freeze -> runs/lowlevel/controller.pt (TorchScript, CPU-runnable)
modal run smoothride/lowlevel/export.py
```

## Files

| file | role |
|---|---|
| `modal_image.py` | Modal App: Isaac Sim 4.5.0 + IsaacLab 2.0.2 + WheeledLab image, A100, Volume |
| `train_isaac.py` | Modal entrypoint → headless PhysX + RSL-RL PPO; Isaac imports stay in the remote fn |
| `isaac_task/track_env_cfg.py` | the command-tracking task (scene, command, actions, obs, rewards, DR) |
| `isaac_task/car_cfg.py` | WheeledLab car as an Articulation |
| `isaac_task/agents/rsl_rl_cfg.py` | compact PPO actor-critic config |
| `export.py` | frozen checkpoint → `controller.pt` |

## Honest status

I scaffolded this on a Mac, so **none of it is runnable/verifiable here** (no
Isaac, no GPU). The Isaac Lab 2.0 / RSL-RL APIs and the Modal wiring are written
faithfully; the parts that need the actual WheeledLab repo are marked
**`TODO(wheeledlab)`** — specifically in `car_cfg.py`:

- `USD_PATH` — the real car asset path inside `/opt/WheeledLab`
- `STEER_JOINTS` / `DRIVE_JOINTS` regexes, `WHEEL_RADIUS`, `MAX_STEER_RAD`

Grep the cloned WheeledLab repo on the first Modal run to fill those, then the
task is complete. The image pins (Isaac Sim 4.5.0, IsaacLab v2.0.2) and the
WheeledLab clone/install in `modal_image.py` may also need a release-tag tweak
once you pick one.
