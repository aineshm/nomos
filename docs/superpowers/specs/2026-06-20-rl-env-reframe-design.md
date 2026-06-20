# SmoothRide — RL Environment Reframe (Design Spec)

**Date:** 2026-06-20
**Status:** Draft for review
**Supersedes framing in:** `STACK.md`, `REALISM.md` (architecture intact; this sharpens the RL formulation, the eval loop, and the tool roles)

---

## 1. Motivation

The current env (`smoothride/env/kinematic.py`) hit a **crash-rate floor** (~1.13–1.24 crashes/car at density) that reward tuning, PPO-Lagrangian, and runtime shields could not break past. Root-cause analysis (DEVLOG #15–17, #39–46) showed the ceiling lives in the **representation and reward framing**, not the optimizer.

This reframe makes three changes:

1. **Constrained MDP, not weighted-sum reward.** Optimize *travel time*; forbid crashes / rule / lane-geometry violations as **constraints**, not penalty terms.
2. **A deterministic, trace-based verifier** defines reward and run validity — separated cleanly from any learned/stochastic critic.
3. **A strict separation of concerns** across tools: *simulation ≠ rendering*, and *in-loop prediction ≠ foundation-model reasoning*. Every external tool gets exactly one non-overlapping job.

The thesis is unchanged: **decentralized self-driving agents coordinate without traffic signals, with zero crashes, at realistic SF density.**

---

## 2. Core principles (the decisions that resolve the design)

| Principle | What it means |
|---|---|
| **Agency, no hive mind** | Each car runs the (shared-weight) policy on its **own local observation**. No inter-agent communication, no shared runtime state. Same brain, separate bodies, separate perceptions → independent behavior. |
| **Simulation ≠ Rendering** | The *compute* layer (dynamics + collisions + RL) produces per-car poses. The *render* layer just draws them. These are different engines and never conflated. |
| **Two world models** | (a) An **in-loop, microsecond** predictor the policy uses to anticipate neighbors. (b) **Cosmos 3**, an **out-of-loop, seconds-scale** foundation model for scenario generation, evaluation, and photoreal demo. Cosmos never sits in the per-step driving loop. |
| **Verify the trace, don't re-simulate** | The verifier is a pure function over the *recorded trajectory*. No physics replay, no randomness, no network — so it is deterministic regardless of GPU/float non-determinism. |
| **Every tool earns an artifact** | No tool enters the stack without a visible job and output. |

---

## 3. The RL formulation — Constrained MDP (CMDP)

```
maximize   E[ reward ]      where  reward = −travel_time  (+ arrival bonus)
subject to crashes        ≤ 0
           off_road        = never
           rule_violations ≤ 0
```

- **Objective (reward):** efficiency — minimize time to destination. Dense per-step shaping = forward progress along the route; sparse arrival bonus on reaching the destination.
- **Constraints (cost channel):** crash, off-road (left the drivable polygon), traffic-rule violation (speed-limit, wrong-way, uncontrolled-junction yield). Carried as a **dense per-agent, per-transition cost**, not a binary episode flag.
- **Solver:** MAPPO / shared-param actor + centralized critic (CTDE, training-only), with **PPO-Lagrangian** dual ascent on the cost constraint (already prototyped in `rl/ppo.py`).

### Why constraints, not a binary "invalid run"

A whole-run invalidation produces **zero gradient** early in training (with many agents, almost every run has a crash), so nothing learns. Instead:

- **Training:** crash = a large terminal cost on the *offending agent's* transition; other agents continue. The policy learns *which action caused the failure* across iterations.
- **Eval / demo:** "valid run = zero violations across all agents" — the binary bar is a **reporting metric**, computed by the verifier, not the training signal.

---

## 4. The abstract car (agent data model)

Immutable record, updated functionally each step (consistent with the existing `State` and the project immutability rule).

```
Car:
  id
  origin, destination               # route context (per-agent)
  route: [waypoints]                # path from OSM A* (or Google-cached route pool)
  pose:        (x, y, z), heading
  kinematics:  speed, steer_angle, lane
  progress:    wp_ptr, dist_remaining, travel_time
  status:      ACTIVE | ARRIVED | CRASHED
  costs:       crash, off_road, rule_violation   # CMDP constraint channel
  spawn_grace                       # merge-in immunity (existing mechanism)

Action: [accel/brake ∈ [-1,1], steer ∈ [-1,1], lane_change ∈ {-1,0,+1}]
```

Each car carries **its own** origin/destination/route and perceives **its own** local FOV. That is the operational definition of agency.

---

## 5. Perception model (how a car "detects surroundings")

No pixels in the policy loop. Perception = **geometric sensors** in the sim engine:

- **Field-of-view + occlusion:** ray-cast / range queries; a building or vehicle blocks line-of-sight to whatever is behind it.
- **Detection list (structured obs):** for each visible object — relative position, relative velocity, class (car / pedestrian / static). Microsecond-fast.
- **Buildings:** from **OSM building footprints** (`ox.features_from_bbox(bbox, tags={"building": True})`), extruded to `height` (or `building:levels × 3 m`), placed at the draped ground `z`. Used as occluders + colliders + render meshes.
- **Route/nav context (per-agent, app-style):** distance/time remaining on route, and a **coarse traffic-volume signal derived from the sim's own density** (not Google's live feed — that would be phantom traffic the agent isn't actually in). School/construction zones from OSM tags.

This delivers true 3D perception (FOV + occlusion) with no vision model and no Cosmos in the loop.

---

## 6. Episode rules

- **Fixed time horizon** (`max_steps`) — bounded rollouts for vectorized training.
- **Per-agent done-on-arrival** — a car that reaches its destination is done (then respawns for persistent-flow training, or finishes for a finite demo cohort).
- **No "all agents must arrive" gate** — that makes episode length unbounded (one stuck car stalls the batch). "All arrive" is a *demo* framing, not a training rule.
- **Headline metrics:** throughput (arrivals/horizon), travel time, crash count, validity.

---

## 7. Environment setup pipeline

**Build once (offline):**
1. OSM road graph for the SF bbox → metric (UTM) frame (existing).
2. Elevation → per-node `z`, per-edge `grade` (USGS 3DEP via `py3dep`, **or** Cesium World Terrain).
3. OSM building footprints → occluder/collider geometry.
4. **Route pool** — origin→destination paths (OSM A*, or Google Directions cached once into a pool, then sampled).
5. *(optional)* Antim/Gizmo SimReady scene/asset variety; Cosmos-Predict hard scenarios.

**Per step (inner loop):**
```
for each car (parallel):
  obs    = perceive(FOV+occlusion, own kinematics, route_remaining, nav traffic)
  action = policy(obs)                      # decentralized, local obs only
  pose'  = dynamics(pose, action)           # kinematic bicycle (train) / rigid-body (hi-fi)
  costs  = check(collision, off_road, rule) # constraint signals
  reward = −Δtime (+ arrival bonus)
```

---

## 8. The experiment → result → reward → fine-tune loop

```
 ┌─ EXPERIMENT ─ roll out current policy on a seeded world cohort ───┐
 │     (seed + scenario_id + checkpoint → full trajectory)           │
 │                          ↓                                        │
 │  RESULT ──── the TRACE: every car's per-step pose + events        │
 │                          ↓                                        │
 │  VERIFIER ── deterministic pass over the trace →                  │
 │              per-car: arrived? travel_time? valid?                │
 │              + constraint costs (crash / off-road / rule)         │
 │                          ↓                                        │
 │  REWARD ──── −travel_time  s.t.  costs ≤ 0   (CMDP)               │
 │                          ↓                                        │
 │  FINE-TUNE ─ MAPPO update + Lagrangian λ update                   │
 └────────────  next round: harder scenarios (Cosmos / Antim curriculum)
```

"Fine-tune" = the RL **policy update** (gradient step), not LLM fine-tuning. Cosmos is **not** in this loop — it feeds the curriculum (Predict) and provides a *secondary* qualitative critique (Reason); it never defines reward.

---

## 9. Trace schema (what we record per run)

**Manifest (once per run) — makes any run replayable bit-for-bit:**
`run_id, seed, scenario_id, policy_checkpoint_id, config_hash` (env params + map version + code version).

**Timeline (per car, per step):**
`t, id, x, y, z, heading, speed, steer, lane, action, wp_ptr, dist_remaining`.

**Events:** `crash` (who / where / closing-speed), `off_road`, `rule_violation` (type), `arrival` (travel_time), `respawn`.

**Aggregates:** throughput, mean/median travel time, crash count, violation count, mean speed, **validity flag**.

Stored as structured JSONL per run. Candidate tooling: **HUD** as the experiment/eval harness, **Raindrop** for trace inspection, **Modal** for running rollouts + offline Cosmos fan-out at scale.

---

## 10. The deterministic verifier (reward + validity source of truth)

**Principle: verify the trace, don't re-simulate.** A verifier that re-runs physics could disagree with itself (GPU/float non-determinism). A verifier that applies geometric predicates to logged poses is deterministic by construction.

Predicates (pure booleans over numbers already in the trace):

- **crash** = min footprint distance `< collision_radius` at any step
- **off_road** = car center outside the drivable road polygon
- **rule_violation** = speed > edge limit, wrong-way on a one-way, failed uncontrolled-junction yield, etc.
- **arrived** = destination waypoint reached within horizon
- **valid_run** = all constraints satisfied for all cars at all steps

Reward and validity come **only** from this verifier.

### Verifier vs Cosmos-Reason (must stay separate)

| | Deterministic verifier (geometry) | Cosmos-Reason (VLM critic) |
|---|---|---|
| Defines reward / validity? | ✅ the ground truth | ❌ never |
| Deterministic? | ✅ always | ❌ stochastic, networked |
| Role | the judge | secondary plausibility / "did this look real" reviewer for curriculum mining & demo QA |

---

## 11. Compute vs. render (and where Cesium / Isaac sit)

| Concern | Engine | Notes |
|---|---|---|
| **Compute** (dynamics + collisions + RL) | kinematic JAX (train) / Isaac (hi-fi) | physics + millions of steps |
| **Geometry / ground truth** | OSM + Cesium World Terrain + Cesium OSM Buildings | real SF roads, hills, buildings |
| **Render** (the demo) | **Cesium** (browser) / Isaac RTX (cinematic) | fed by per-car poses `(id,t,x,y,z,heading)` |

Cesium is the **render + geometry-source** layer, not a physics engine. Caveat: Cesium tiles serve *visuals*; the *sim* still needs collision/occlusion geometry derived from OSM in the engine's own format.

---

## 12. Tool-role map (one job each)

| Tool | Single job |
|---|---|
| **OSMnx / OSM** | real SF road graph, building footprints, school/construction tags, routing base |
| **USGS 3DEP / py3dep** *or* **Cesium World Terrain** | elevation → node `z`, edge `grade` |
| **Cesium (CesiumJS)** | ground-truth terrain + 3D buildings; **browser demo viewer** |
| **Kinematic JAX env** | fast coordination-policy **training** substrate |
| **Isaac Sim / Isaac Lab** | hi-fi physics **execution**; frozen low-level controller training; cinematic render |
| **WheeledLab** | car asset + dynamics recipe for the frozen low-level controller |
| **Cosmos-Predict** | **offline** hard/adversarial scenario generation → curriculum (retires the DIY DDPM) |
| **Cosmos-Reason** | **secondary** plausibility critic + run QA (never the reward) |
| **Cosmos-Transfer** | photoreal sim→real demo render |
| **Antim Labs / Gizmo** | **SimReady** static scene/asset variety → spatial domain randomization |
| **Google Maps** | real route pool, **cached once at setup** (never live, never traffic) |
| **Modal** | GPU compute — training rollouts + offline Cosmos fan-out |

---

## 13. Open decisions (to confirm before planning)

1. **Training substrate** — (A) keep fast kinematic JAX, execute/demo in Isaac *[recommended for hackathon]*; (B) train in Isaac Lab with lightweight kinematic actors at moderate counts; (C) full rigid-body MARL in Isaac *[budget risk]*. "Use Isaac for sims" most naturally means A (Isaac = execution/demo) — **confirm**.
2. **Elevation source** — USGS 3DEP vs Cesium World Terrain.
3. **Primary render target** — Cesium browser (always-works) vs Isaac RTX (cinematic) vs both tiers.
4. **Agents** — shared-weight policy (default) vs heterogeneous per-car parameters.
5. **Antim** — critical path or stretch (only if OSM geometry isn't varied enough).

---

## 14. Non-goals (YAGNI)

- No real-car data, no imitation learning, no deployment to hardware (the sim is the dataset).
- No live Google Maps calls in the loop.
- No Cosmos in the per-step control loop.
- No three parallel scenario generators — Cosmos-Predict supersedes the DIY DDPM.

---

## 15. Risks

- **Many-agent Isaac training** is the expensive corner → mitigate via substrate decision A/B.
- **Vendor integration tax** (Modal + Cosmos + Cesium + Antim + Isaac + OSM) → each must earn an artifact; sequence by ROI.
- **Building-height gaps** in OSM → fall back to `levels × 3 m` / default.
- **Verifier/sim drift** → verifier reads the trace only; never couples to physics determinism.
