# Handoff — what to expect from the sim (the RL contract)

**Audience:** whoever builds the **deterministic verifier, reward system, and CMDP training** on top of the sim.
**Companion to:** the design spec (`docs/superpowers/specs/2026-06-20-rl-env-reframe-design.md`) and the 3D-sim plan (`docs/superpowers/plans/2026-06-20-3d-sim-setup.md`).
**Status of each interface is tagged:** ✅ exists today · 🔜 added by the 3D-sim plan · 🧩 yours to build (this doc is the contract you build against).

> TL;DR: The sim is a pure `reset`/`step` JAX env over N cars. It hands you per-step **observations**, a scalar **reward**, and a **cost/constraint channel**. You optimize travel time subject to the costs (CMDP). Validity and reward come from a **deterministic verifier that reads a logged trace** — never from re-running physics, never from an LLM.

---

## 0. Status update (2026-06-20) — plan changes agreed after main `e2cd4cc`

A large feature landed on `origin/main` (`e2cd4cc` "Worldsim 3D physics path, traffic-law RL, and respawn fixes", DEVLOG 49–52) that overlaps this contract. We are **not** pulling/merging it (the Cesium viewers have diverged into incompatible architectures). Instead we adopt its *findings* as decisions. Three changes to the plan:

**① Spawn fix — DONE on main, dropped here.** Our diagnosed "crash floor" was ~94% a spawn artifact (47/96 cars flagged crashed on frame 0). Main fixed exactly this in `kinematic.step` (DEVLOG 52): spawn-immune cars are now **excluded as collision partners** (`neighbor_ok = valid & ~immune[cand]`), and respawns **merge in at 0.6× the edge speed limit** instead of a dead stop. We will **not** rebuild this — adopt main's approach when/if integrating. Effect on this contract: `just_crashed`/`crashes` (§4, §6) now reflect genuine car-to-car contact, not spawn overlap. Main also *tried and reverted* respawn-at-route-start (piled cars onto shared start nodes, worse at 6000 cars) — don't retry it.

**② Remove-on-arrival (finite cohort) — TO BUILD here, complementary.** Not on main (main kept continuous respawn). Behavior: when a car reaches its destination it **freezes at the destination and is masked out of collision / reward / cost** from that step on; we log the arrival step + location; **no respawn**. This yields one clean origin→destination trajectory per car (no teleport-streak data pollution) and naturally decreasing density. It refines the semantics below: `arrived[t,i]` (§7) latches True once and stays; an arrived car stops contributing to other cars' costs. ⚠️ This edits `kinematic.py` (training-env code) — sequenced as the next env change, pending explicit go. Chosen over "park-on-arrival" because a frozen-but-collidable car becomes a phantom obstacle that re-pollutes the data.

**③ `legality.py` — available reference, NOT a mandate.** Main ships `env/legality.py`: a pure `(env, state) → per-car` check — **OFF-LANE** (>1.5 lane-widths from the nearest lane centerline of the segment the car is *currently* on; point-to-segment so legal lane-changes / corner-cuts don't false-trip) and **WRONG-WAY** (heading against the route while moving; respawn-grace exempt). On main it is baked into the PPO reward (`w_offlane` / `w_wrongway`), **not** split into a CMDP cost channel. It is the natural source for this contract's `off_road` / `rule_violation` (§6) and the verifier's off-road / wrong-way predicates (§8) — **but the verifier is now being built independently** (separate `rl-verifier` worktree, branched from `dfc67b9`), so whether to consume `legality.py` or re-derive the geometry is **the verifier author's call**. Treat it as a validated reference, not a required dependency.

> Note on "road limits": speed limits **are already hard-enforced** — `kinematic.step` clips `speed` to `min(v_max, routes_speed[edge])`, so the §6 rule-bit `1 = over speed limit` is currently *unreachable* in the kinematic env. The visible "cars don't follow the road" behavior is **lane-keeping**, not speed: our `runs/trained.msgpack` was trained with **no** off-lane penalty (the legality reward lives only on main), so it cuts corners / drifts. ② + adopting ③'s signal are what close that gap.

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
| `just_crashed` | (N,) bool | collided **this** step (then respawns) ✅ → **primary cost signal**; genuine car-to-car contact only since main's spawn fix (§0①) |
| `crashes` | (N,) i32 | cumulative collisions ✅ |
| `spawn_grace` | (N,) i32 | merge-in immunity countdown ✅; immune cars excluded as collision partners (§0①) |
| `goals` | (N,) i32 | cumulative trips completed ✅ → **throughput** (semantics shift under remove-on-arrival, §0②) |
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
`rl/ppo.py` already subtracts `lam * cost` (Lagrangian). Today it uses `just_crashed`; widen it to the sum above as the 🔜 signals land. **`off_road` / wrong-way already exist on main** as `env/legality.py` (reward-shaped via `w_offlane`/`w_wrongway`, not yet a cost) — see §0③; it's a reference, not a required import.

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
| Trace dataclasses | `smoothride/rl/trace.py` 🧩 | **you** (in `rl-verifier` worktree, §0③) |
| Deterministic verifier | `smoothride/rl/verifier.py` 🧩 | **you** (in `rl-verifier` worktree, §0③) |
| Off-lane / wrong-way signal | `smoothride/env/legality.py` (on `origin/main`) | reference for the verifier — §0③ |
| Reward (CMDP objective) | `smoothride/env/kinematic.py` reward + `rl/` 🧩 | **you** |
| Lagrangian training | `smoothride/rl/ppo.py` ✅ (extend cost) | shared |

## 12. What's frozen vs. in flux

**Frozen (build against these):** the `reset`/`step` signature; obs is decentralized/local; `reward` vs `cost` separation; the verifier-reads-trace principle; the Trace schema in §7; metric-frame units.

**In flux (will grow, read dynamically):** `obs_dim`/`act_dim` (read the properties); the exact `rule_violation` bits may gain entries; `z`/`off_road`/`dist_remaining` land with the 3D + reframe work. None of these change the *shape* of the contract — only widen fields.

---

**Start here:** define `smoothride/rl/trace.py` (§7), then write `verify()` (§8) with unit tests over hand-built `Trace` fixtures (you can fabricate a 3-step, 2-car trace by hand — no sim needed). That unblocks the whole RL side before the 3D sim is even finished.
