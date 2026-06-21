# Pedestrian-yield environment — design spec

**Date:** 2026-06-20
**Branch:** `worktree-3d-sim-setup`
**Status:** approved design, pre-implementation

---

## 1. Vision & framing (the "why")

The pitch is a **2040 scenario where AVs are the only drivers**. Low-level autonomy
(detect, avoid, adapt speed) is assumed solved — Waymo-level commodity. What today's
AVs *don't* do is shed the human-era traffic system: they still obey lights, painted
lanes, and posted limits because they share the road with humans. The bet: once *every*
car is autonomous, a coordinated swarm can negotiate right-of-way directly, so the human
coordination layer (traffic signals especially) becomes unnecessary.

This project learns that **swarm-coordination / behavioral layer** with RL, targeting
something map-agnostic that could drop onto any AV anywhere.

**Pedestrians are the irreducible uncoordinated agent.** Everything else runs the shared
policy; people who choose to walk do not. The whole point of the environment is to show
the swarm **gracefully accommodating an agent that does not follow the protocol** — by
slowing for it.

### Staged thesis (what we are actually proving)
- Design for **today's roads** — 14 years is not enough to repaint the world, and proving
  this on existing infrastructure is what *earns* future change.
- Keep the human road layout (lanes, directions) as the swarm's **coordination contract**
  for now.
- The dial we turn is **speed**: run **slow and provably safe first** (near-zero crash
  risk). That result is the license to scale speed up later — and eventually to justify
  changing the roads themselves.
- **Safety is the gate; speed is the reward you unlock by passing it.**

### Scale
City-scale density — "SF's population in this system." Dense pedestrians (300–400) on the
existing named SF regions (`map_loader.SF_REGIONS`: downtown / chinatown_fidi / mission /
nopa). This density is *why* the perception change below is required, not optional.

---

## 2. What does NOT change

- **Car dynamics.** The kinematic-bicycle model, route/lane/waypoint machinery, spatial
  hash, and finite-cohort freeze logic are untouched. The env *already* supports both
  capabilities the behavior needs (verified in code):
  - **Brake to a full stop mid-route and resume.** `speed = clip(st.speed + accel*dt,
    0.0, vmax)` clamps to 0; a merely-stopped car is not `done`, so it is not frozen and
    can accelerate again next step. A stopped car earns only `−w_time` (lost-time
    opportunity cost), so stopping is *allowed but costs time* — the exact tradeoff we
    want learned. A stationary car is never falsely flagged wrong-way (`speed > idle_speed`
    gate).
  - **Per-step perception loop.** `_observe` already runs every step and the policy
    conditions each action on current surroundings — so temporary stopping/yielding is
    expressible today. (The *content* of that observation is what §4 changes; the per-step
    loop itself is unchanged.)
  - Conclusion: today's policy never brakes purely because **nothing in the reward/cost
    pays for slowing** — not an env limitation. We change only **observation** and the
    **cost channel**, never the physics.
- **Reward.** Stays §9 efficiency-only: `w_progress·progress + w_goal·arrival − w_time`.
  All constraints flow through the deterministic verifier's cost channel (CMDP).
- **Kept constraints (coordination contract on today's roads):** on-roadway, lane-keeping
  (`off_lane`), `wrong_way`. These remain in the cost channel.
- **Viewer.** Already renders pedestrians (amber cylinders). No change.

---

## 3. Pedestrian motion model (deterministic, hard-coded paths)

Replaces the current random-walk `kinematic.py::_ped_step` (turn + dart, bounce off world
bounds).

- **Host-side `build_ped_paths(net, n_peds, seed)`** produces fixed arrays:
  - `ped_paths (M, P, 2)` — per-ped polyline = a **sidewalk run** + **one perpendicular
    crossing**. Sidewalks are offset from a road centerline by ≈ road-half-width + ~1.5 m.
  - `ped_cum (M, P)` — cumulative arc length per polyline (for arc interpolation).
  - `ped_starts (M,)` — a random **start step** per ped (staggered).
  - `ped_crossing (M, P)` (or an arc-interval) — marks which portion of each path is the
    in-roadway crossing segment. Used to gate the yield cost.
- **Env step (deterministic, no per-step RNG → JAX/vmap-friendly):**
  - `walked = max(0, t − ped_start) · ped_speed · dt`
  - `ped_pos = arc_interpolate(path, walked)`; before start → `path[0]`, after end →
    `path[-1]`.
  - `ped_vel` / `ped_dir` derived from the path tangent (for both rendering and the new
    observation features).
  - `ped_crossing_now (M,)` — boolean: is the ped's current arc position inside its
    crossing segment.
- Pedestrians are **somewhat unpredictable** (cross perpendicular at an arbitrary point,
  staggered starts) — the uncoordinated human element, *not* polite crosswalk users. No
  traffic signals anywhere.

---

## 4. Perception — permutation-invariant set encoders (Deep Sets)

### Why (settled by research, 3 independent scans — see §4.4)
At SF-scale density, the current **"1 nearest ped, 3 features"** block — and the fixed-K
neighbor MLP in general — is the **documented failure mode**, not a scaling choice:
- **Truncation:** a fixed K silently drops the (K+1)-th agent. Fatal at 300–400 peds;
  K=1 ped is the extreme case.
- **Rank-flip discontinuity:** sort-by-distance makes the input vector jump when two agents
  swap rank → non-smooth value/policy surface.
- It is also blind to ped *motion* and to whether a ped is *crossing*.

An AV-specific set encoder (ESC, arXiv:2105.11299) reports a **62.2% approximation-error
reduction vs. a sorted-list baseline**; **V-Max** (JAX/Waymax RL — our exact stack) ships
only vectorized encoders and its MLP baseline underperforms (~0.68 vs ~0.84–0.87). For the
stated cross-map ("any AV anywhere") goal, ego-relative vectorized representations
generalize best.

### Decision — Deep Sets, per entity type, over radius-capped masked sets
Replace the flat fixed-K observation with **two permutation-invariant Deep Sets encoders**
(one for cars, one for pedestrians):

`pooled_type = pool_i ( φ_type(neighbor_i) )`, mean⊕max pool, then
`obs_embedding = MLP( concat[ ego_vec, pooled_cars, pooled_peds ] )`.

- **Per-neighbor features (ego frame):** relative position, relative **velocity**, type
  flag; for peds also the **crossing bit**.
- **Radius-capped candidate sets (the perf-critical detail):** do **not** pad to 400 peds
  per agent — that blows up trajectory memory (`B×T×N×400×feat`). Use the existing spatial
  hash to gather only neighbors within a radius, **capped at ~16–32** per type, padded +
  masked to the cap. Bounded memory, still permutation-invariant and cardinality-agnostic
  locally (cap ≫ K=1, so it rarely truncates). This mirrors GPUDrive's radius/view-cone
  observation.
- **Masking (JAX correctness):** zero out invalid slots before pooling and divide by the
  **live count** (`clip(mask.sum, 1)`), never by the padded N. (For phase-2 attention: add
  `-1e9` not `-inf` pre-softmax and guarantee ≥1 valid key, else NaNs when a world has zero
  neighbors.)
- **Observation becomes structured** (a pytree): `(ego, cars[C,feat], cars_mask,
  peds[C,feat], peds_mask)` instead of a flat vector. `obs_dim` concept is replaced; full
  retrain from scratch (expected).

### `Movers` view (perception-only unification)
Cars (`State`) and peds keep their **separate** dynamics & storage (routes/lanes vs.
polylines). A lightweight `@flax.struct.dataclass` **view** (struct-of-arrays — *not*
per-agent objects, which would break `vmap`/`jit`) is built by concatenating them *only for
the perception/neighbor layer*, so candidate gathering and the set encoders treat "nearby
moving things" uniformly. A full unified state table is explicitly **not** done.

### Phasing
- **Phase 1 (this work): Deep Sets (mean⊕max) for both types.** Minimal correct change,
  ~same effort as the MLP patch, permanently removes truncation + rank-flip.
- **Phase 2 (next iteration): swap the *car* pool for ego-query attention** (social-
  attention, Leurent & Mercat arXiv:1911.12250) — a drop-in replacement of the pooling
  step that models pairwise interaction, relevant to signal-free intersection negotiation.

### 4.4 Research provenance
Three independent literature scans (2026-06-20) converged on the above. Key sources:
Deep Sets (arXiv:1703.06114), ESC permutation-invariant AV state (arXiv:2105.11299),
Social Attention for dense traffic (arXiv:1911.12250), Set Transformer PMA/ISAB
(arXiv:1810.00825), V-Max JAX RL framework (arXiv:2503.08388), VectorNet (arXiv:2005.04259),
GPUDrive (arXiv:2408.01584), CADRL socially-aware DRL (arXiv:1703.08862), Frenet domain
normalization for cross-city transfer (arXiv:2305.17965).

---

## 5. Speed — low cruise cap as the tunable dial

- Add a **configurable cruise-speed cap** (e.g. ~6–8 m/s initially), the "scale up later"
  lever for the staged rollout. Cars ride at the cap by default.
- The **ped-yield cost** (below) makes them dip *below* the cap near a crossing ped and
  re-accelerate after — this produces the visible **braking-for-pedestrians** behavior
  that is the demo.
- **No separate global speed-penalty term.** The cap handles cruise speed; ped-yield
  handles braking; the tight crash target handles safety. (Per-edge speed limits stay as
  an `over_cap`/`over_speed` constraint; the operative ceiling is the lower cruise cap.)

---

## 6. Safety radii — asymmetric & layered

Today `collision_radius` (car-car) and `ped_radius` are both 2.2 m — wrong for the thesis.
Hitting a person is categorically worse; AVs give a human a wide berth. Layered zones:

| Zone | Radius | Effect |
|---|---|---|
| Car–car hard collision | 2.2 m (unchanged) | crash event |
| **Car–ped hard collision** | **~3.5–4 m** (raised) | crash / safety fail → drive to ~0 |
| **Car–ped yield zone (outer, continuous)** | **~8–10 m** | ped-yield cost ramps |

Gradient, not a cliff:

```
distance to crossing ped:
  > R_yield        → 0 cost (cruise at cap)
  R_yield → r_ped  → ped-yield cost ramps with (proximity × speed)   ["slow for the person"]
  ≤ r_ped          → hard collision / safety fail                    [rare if outer zone works]
```

The wide outer zone forces early, gentle slowing so the inner hard zone is almost never
breached → high accuracy, low crashes, visibly cautious-around-people behavior. The
vestigial `prox_radius` field (from the retired proximity reward) is **repurposed** as
`R_yield`.

---

## 7. Cost channel — single channel, ped-yield folded in

The cost channel stays a **single scalar** with **one** Lagrange multiplier ascending
toward **one** tight (low) crash-oriented target. The only continuous term added is
ped-yield; speed is handled by the cap (not a cost), so there is no second term that would
fight the crash target. Dual-multiplier machinery is **not** needed.

```
cost = crash + off_lane + wrong_way + over_cap + ped_yield
```

**Ped-yield term (continuous hinge, gated to crossing peds):**
```
p    = clip((R_yield − d) / (R_yield − r_ped), 0, 1)     # proximity ramp, d = car→ped dist
yield_cost = p · (speed / cruise_cap)                     # zero when stopped or far
# summed/maxed over peds with crossing_now == True, near the ego car
```
- Continuous and graded → optimum is to **slow**, not freeze. Explicitly **not** a hard
  0/1 flag (that risks the "car learns to stop dead" failure).
- Near-zero-crash and small-ped-yield both point the same way (drive carefully / slower),
  so a single multiplier balances them.

**Plumbing:**
- Log **ped positions (+ velocity + crossing flag)** in the rollout `batch` (`ppo.collect`)
  and add `ped_pos`/ped fields to the offline `Trace` (`rl/trace.py`) so both the training
  relabel (`ppo.verifier_cost`) and the offline verifier (`verifier.cost_signal`) compute
  the same ped-yield term.
- Add the ped-yield term inside `verifier.step_cost` (one rulebook for training and grading).

---

## 8. Retrain & evaluate

- Retrain on Modal, verifier-driven (`--verifier --cost-target`), dense peds
  (`--peds 300-400`), low cruise cap, on a training region; then eval on a held-out region
  (`scripts/eval_policy.py --region …`) and render in the viewer.
- Expected new metrics to watch: braking events near peds, min car–ped distance
  distribution, ped-yield cost trend, and that crash rate (esp. car–ped) drives toward 0
  while throughput stays acceptable.

---

## 9. Out of scope (recorded, deferred)

- **Ego-query attention over cars** (social-attention) — phase-2 perception upgrade; phase 1
  uses Deep Sets pooling for both types.
- GNN/GAT and BEV-raster CNN observation — overkill for our setting (heavy to vectorize /
  need a renderer we lack / generalize worse cross-map).
- Frenet/heading canonicalization of lane geometry — optional cross-map transfer booster;
  defer until generalization is measured.
- Full unified entity *state* table — we do perception-view unification only.
- On-device (JAX-scan) port of `step_cost` — separate perf item; host relabel stays for now.
- Global speed-penalty cost term — not needed; cruise cap is the speed dial.

---

## 10. File-level change surface

- `smoothride/env/kinematic.py` — new `build_ped_paths` (host), deterministic arc-interp
  `_ped_step`, ped vel/crossing state, `Movers` view, **radius-capped candidate gathering
  for cars+peds**, **structured observation** (ego + padded/masked car & ped sets) replacing
  the flat 26-dim vector, cruise cap, raised ped radius, repurposed `R_yield`.
- `smoothride/rl/networks.py` — **Deep Sets front-end** in `ActorCritic`: shared per-
  neighbor `φ` MLP + masked mean⊕max pool per type → concat with ego → existing trunk.
  (Phase-2 hook: car pool swappable for ego-query attention.)
- `smoothride/rl/trace.py` — add ped fields (`ped_pos`/vel/crossing) to `Trace`/manifest.
- `smoothride/rl/verifier.py` — ped-yield term in `step_cost` (+ `cost_signal` wiring).
- `smoothride/rl/ppo.py` — structured-obs plumbing (obs is now a pytree); log ped fields in
  `collect`; ped-yield in `verifier_cost`.
- `smoothride/rl/modal_train.py` — `--peds`, cruise-cap, ped-radius, candidate-cap flags.
- Tests — ped path build, deterministic arc-interp, **Deep Sets masking (live-count divide,
  empty-set → zero, permutation invariance)**, ped-yield hinge (graded not binary, zero when
  stopped/far, ramps with speed×proximity), Trace shape with ped fields.
