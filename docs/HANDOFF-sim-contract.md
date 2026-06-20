# Handoff — what to expect from the sim (the RL contract)

**Audience:** whoever builds the **deterministic verifier, reward system, and CMDP training** on top of the sim.
**Companion to:** the design spec (`docs/superpowers/specs/2026-06-20-rl-env-reframe-design.md`) and the 3D-sim plan (`docs/superpowers/plans/2026-06-20-3d-sim-setup.md`).
**Status of each interface is tagged:** ✅ exists today · 🔜 added by the 3D-sim plan · 🧩 yours to build (this doc is the contract you build against).

> TL;DR: The sim is a pure `reset`/`step` JAX env over N cars. It hands you per-step **observations**, a scalar **reward**, and a **cost/constraint channel**. You optimize travel time subject to the costs (CMDP). Validity and reward come from a **deterministic verifier that reads a logged trace** — never from re-running physics, never from an LLM.

---

## 1. Mental model (read this first)

- **The simulator is the dataset.** No real-car data, no imitation. You generate experience by stepping the env.
- **Two numbers, kept separate:** `reward` (what you maximize = efficiency / travel time) and `cost` (constraints you must keep ≤ 0 = crash / off-road / rule). Do **not** fold cost back into reward as a fixed penalty — that's the failure mode we're leaving behind. Use a Lagrangian multiplier (already wired in `rl/ppo.py`).
- **Validity ≠ training signal.** "Invalid run" (any violation) is an **eval/demo metric**. For training, a violation is a **dense per-agent cost on the offending transition**; other agents keep going. Never discard the whole episode.
- **Verify the trace, not the sim.** The verifier is a pure function over logged arrays. Same trace → same verdict, regardless of GPU/float nondeterminism.

---

## 2. Coordinate frames & units

| Quantity | Frame | Unit |
|---|---|---|
| `pos` (car x,y) | local metric (UTM, origin-shifted to bbox min) | meters |
| `z` 🔜 | same metric frame, ground elevation | meters |
| `heading` | CCW from +x (east) | radians, wrapped to (−π, π] |
| `speed` | — | m/s |
| `dt` | — | seconds (default `0.2`) |
| render coords (scene.json) | WGS84 lon/lat + z | degrees, meters |

Reprojection metric↔lon/lat is handled in `smoothride/demo/export_web.py` (`_to_lonlat`, `_lonlat_transformer`). You work in the **metric frame**; only the renderer cares about lon/lat.

---

## 3. Env API ✅ (`smoothride/env/kinematic.py`)

```python
env = K.make_env(pool, world_min, world_max, n_agents=24, n_peds=12, max_steps=300, **kw)
state, obs = K.reset(env, key)                       # obs: (N, obs_dim)
state, obs, reward, done, info = K.step(env, state, action, key)
#   action: (N, act_dim) in [-1, 1]
#   reward: (N,) float32        done: scalar bool (t >= max_steps)
#   info:   dict of per-step signals (see §6)
```

- Pure functions. `jax.jit`-able; `jax.vmap` over worlds (batch axis B → leaves `(B, N, ...)`).
- `env` is a pytree; scalar config fields are static. See `rl/ppo.py::collect` for the canonical rollout (`jax.lax.scan` over `max_steps`).
- `env.obs_dim`, `env.act_dim` are properties — **always read them, never hardcode**.

---

## 4. State — the abstract car ✅🔜 (`State` dataclass)

Per-agent arrays, shape `(N,)` unless noted. ✅ today; 🔜 added by the reframe.

| Field | Type | Meaning |
|---|---|---|
| `pos` | (N,2) f32 | position, meters ✅ |
| `heading` | (N,) f32 | radians ✅ |
| `speed` | (N,) f32 | m/s ✅ |
| `route_idx` | (N,) i32 | index into the route pool ✅ |
| `wp_ptr` | (N,) i32 | current waypoint along the route ✅ |
| `lane` | (N,) i32 | discrete lane index ✅ |
| `just_crashed` | (N,) bool | collided **this** step (then respawns) ✅ → **primary cost signal** |
| `crashes` | (N,) i32 | cumulative collisions ✅ |
| `spawn_grace` | (N,) i32 | merge-in immunity countdown ✅ |
| `goals` | (N,) i32 | cumulative trips completed ✅ → **throughput** |
| `ped_pos` | (M,2) f32 | pedestrian positions ✅ |
| `t` | scalar i32 | step counter ✅ |
| `z` 🔜 | (N,) f32 | ground elevation under the car |
| `off_road` 🔜 | (N,) bool | center left the drivable polygon |
| `rule_violation` 🔜 | (N,) i32 | bitmask: speed/​wrong-way/​yield (enum below) |
| `travel_time` 🔜 | (N,) f32 | seconds since spawn (for the reward) |
| `dist_remaining` 🔜 | (N,) f32 | meters left to destination |

State updates are **immutable** (`state.replace(...)`); follow that pattern in any wrapper you write.

---

## 5. Observation layout ✅ (decentralized — local only, this is the "agency, no hive mind" guarantee)

`obs_dim = 6 + 1 + k_neighbors*4 + 3` (with default `k_neighbors=4` → 26). Each car sees **only** this; no global state at execution.

| Slice | Contents |
|---|---|
| `[0:6]` ego | `speed/vmax`, `sin(herr)`, `cos(herr)`, `clip(dist_to_wp/100)`, `progress`, `clip(lead_gap/lead_cone)` |
| `[6:7]` | `lane_frac` (lane / (lanes−1)) |
| `[7:7+4K]` neighbors | per nearest neighbor: `nx/50`, `ny/50`, `nvel_x/vmax`, `nvel_y/vmax` (ego frame) |
| `[last 3]` pedestrian | nearest ped `px/50`, `py/50`, `pmin/50` |

🔜 the reframe will append a **grade** scalar and **route/nav** context (`dist_remaining`, coarse local traffic). When it does, `obs_dim` changes — that's why you read `env.obs_dim`.

---

## 6. Action + cost channel (the CMDP seam)

**Action** ✅ — `act_dim = 3`, each in `[-1, 1]`:
`action[:,0]` = accel/brake · `action[:,1]` = steer · `action[:,2]` = lane-change intent (`>0.5` right, `<−0.5` left).

**`info` dict from `step`** ✅ today:
`just_crashed (N,) bool`, `crashes (N,) i32`, `goals (N,) i32`, `total_goals`, `crashes_per_car`, `ped_hits`, `mean_speed`.

**The cost vector you build the constraint on** 🧩 — assemble per step, shape `(N,)`:
```python
cost = info["just_crashed"].astype(f32)          # ✅ available now
     + info["off_road"].astype(f32)              # 🔜
     + (info["rule_violation"] > 0).astype(f32)  # 🔜
```
`rl/ppo.py` already subtracts `lam * cost` (Lagrangian). Today it uses `just_crashed`; widen it to the sum above as the 🔜 signals land.

**Rule-violation enum** 🔜 (bitmask in `rule_violation`):
`1 = over speed limit · 2 = wrong-way on one-way · 4 = entered occupied uncontrolled junction without yielding`.

---

## 7. The run trace — what the verifier consumes 🧩 (schema is the contract; define once, both sides import)

A rollout is logged to a **trace**: a manifest + per-step/per-car arrays. The deterministic verifier reads *only* this — it does not touch the env. Suggested location: `smoothride/rl/trace.py`.

```python
# smoothride/rl/trace.py  (define this; the sim/rollout wrapper fills it)
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TraceManifest:
    run_id: str
    seed: int
    scenario_id: str
    policy_checkpoint_id: str
    config_hash: str            # env params + map version + code version
    dt: float
    n_steps: int
    n_agents: int
    n_peds: int

@dataclass(frozen=True)
class Trace:
    manifest: TraceManifest
    # timeline, shape (T, N) unless noted
    pos: np.ndarray             # (T, N, 2) meters
    z: np.ndarray               # (T, N)
    heading: np.ndarray         # (T, N)
    speed: np.ndarray           # (T, N)
    lane: np.ndarray            # (T, N) i32
    action: np.ndarray          # (T, N, 3)
    wp_ptr: np.ndarray          # (T, N) i32
    dist_remaining: np.ndarray  # (T, N)
    # events
    crashed: np.ndarray         # (T, N) bool  — collision this step
    off_road: np.ndarray        # (T, N) bool
    rule_violation: np.ndarray  # (T, N) i32   — bitmask (§6)
    arrived: np.ndarray         # (T, N) bool  — reached destination this step
    # static
    speed_limit: np.ndarray     # (T, N) m/s   — edge limit under each car (for the speed rule)
    collision_radius: float
    road_polygon_ref: str       # how off_road was/should be judged
```

The matching JSON the **renderer** consumes (`scene.json`) is a lon/lat projection of this — already specified in `smoothride/demo/scene.py` (schema v1). Trace = metric truth for the verifier; scene.json = lon/lat for the eyes.

---

## 8. The deterministic verifier — your main build 🧩

Pure function over a `Trace`. **Must:** be deterministic, no randomness, no wall-clock, no network/LLM, no physics replay. **Must not** import the env or call Cosmos.

```python
# smoothride/rl/verifier.py  (yours to build)
from dataclasses import dataclass

@dataclass(frozen=True)
class CarVerdict:
    arrived: bool
    travel_time: float | None     # seconds, None if never arrived
    crashed: bool
    off_road: bool
    rule_violations: int          # count over the run
    valid: bool                   # no crash/off-road/rule the whole run

@dataclass(frozen=True)
class RunVerdict:
    valid_run: bool               # all cars valid (the eval headline)
    throughput: int               # total arrivals
    mean_travel_time: float
    crash_count: int
    violation_count: int
    per_car: list[CarVerdict]

def verify(trace: Trace) -> RunVerdict: ...
```

Predicates (all geometric, all over the trace):
- **crash** = `trace.crashed[t,i]` (already a footprint-overlap event) — or recompute from `pos` + `collision_radius` for an independent check.
- **off_road** = `trace.off_road[t,i]` (center outside `road_polygon_ref`).
- **rule** = `trace.rule_violation[t,i]` bits; speed-rule cross-checks `speed > speed_limit`.
- **arrived / travel_time** = first `arrived[t,i]` → `t * dt`.
- **valid_run** = no crash/off-road/rule for any car at any step.

Reward and validity come **only** from here.

---

## 9. The reward 🧩 (CMDP objective)

Maximize efficiency; constraints handled by the cost channel (§6), not here.

```python
# per step, per agent (dense shaping that sums to ~ -travel_time):
reward = w_progress * progress_along_route        # forward m this step
       + w_goal * arrived_this_step               # sparse arrival bonus
       - w_time                                   # small per-step time cost
# NOTHING about crashes here — those are the CONSTRAINT (cost), not the reward.
```
`progress` and the goal bonus already exist in `kinematic.py::step`; the change is **removing crash/proximity terms from `reward`** and routing them through `cost` instead. Keep `w_*` in `env` config (already fields).

---

## 10. Determinism guarantees (what you can rely on) ✅

- Same `seed` + same `config_hash` → identical rollout on the same hardware (JAX is deterministic per-device).
- The verifier is hardware-independent because it reads the trace, not the sim.
- The manifest's four IDs (`seed`, `scenario_id`, `policy_checkpoint_id`, `config_hash`) make any run replayable — store them with every trace.
- **Cosmos-Reason is NOT in this path.** It's an optional, separate, *qualitative* critic for curriculum mining / demo QA. It never sets reward or validity.

---

## 11. Where things live / where to put yours

| Concern | Path | Owner |
|---|---|---|
| Env, state, obs, dynamics | `smoothride/env/kinematic.py` ✅ | sim |
| Elevation/grade, buildings | `smoothride/data/` 🔜 | sim (3D plan) |
| Scene schema (render contract) | `smoothride/demo/scene.py` 🔜 | sim (3D plan) |
| Trace dataclasses | `smoothride/rl/trace.py` 🧩 | **you** |
| Deterministic verifier | `smoothride/rl/verifier.py` 🧩 | **you** |
| Reward (CMDP objective) | `smoothride/env/kinematic.py` reward + `rl/` 🧩 | **you** |
| Lagrangian training | `smoothride/rl/ppo.py` ✅ (extend cost) | shared |

## 12. What's frozen vs. in flux

**Frozen (build against these):** the `reset`/`step` signature; obs is decentralized/local; `reward` vs `cost` separation; the verifier-reads-trace principle; the Trace schema in §7; metric-frame units.

**In flux (will grow, read dynamically):** `obs_dim`/`act_dim` (read the properties); the exact `rule_violation` bits may gain entries; `z`/`off_road`/`dist_remaining` land with the 3D + reframe work. None of these change the *shape* of the contract — only widen fields.

---

**Start here:** define `smoothride/rl/trace.py` (§7), then write `verify()` (§8) with unit tests over hand-built `Trace` fixtures (you can fabricate a 3-step, 2-car trace by hand — no sim needed). That unblocks the whole RL side before the 3D sim is even finished.
