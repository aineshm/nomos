# Nomos — reinforcement-learned traffic coordination for a driverless San Francisco

> *nomos* (νόμος) — Greek for "law" or "custom." When every car is autonomous, the
> law of the road stops being painted lines and signal timing and becomes a learned
> behavioral policy. That policy is Nomos.

**Thesis.** In a world where every car drives itself, the human-era traffic system —
signals, stop signs, rigid lane law — becomes unnecessary. A coordinated swarm can
negotiate right-of-way directly. Nomos **learns that swarm-coordination / behavioral
layer** with multi-agent reinforcement learning on the **real San Francisco road
graph**. Low-level autonomy (perception, throttle/brake/steer control) is assumed
solved; Nomos is the layer above it — *how the fleet behaves*. The one irreducible
human element is **pedestrians**, whom cars must detect and slow for.

This README is the entry point: the inspiration, the idea, how it works, the hard
parts and how we got through them, and how to run it. The full blow-by-blow build
log is in [`DEVLOG.md`](DEVLOG.md).

---

## Inspiration

Two observations collided:

1. **Almost everything about today's roads exists for human limitations.** Traffic
   lights, stop signs, lane discipline, speed limits — they're a protocol for slow,
   distractible, uncoordinated human drivers who can't talk to each other. Remove the
   humans and the protocol is legacy overhead. Autonomous cars can negotiate an
   intersection by *agreement*, not by waiting for a light.

2. **The behavioral layer is the unsolved part.** Perception and low-level control
   are largely solved problems with huge teams behind them. What *isn't* solved is the
   collective behavior — how thousands of self-driving cars share a real city without a
   central traffic controller and without crashing. That's a multi-agent coordination
   problem, and it's exactly what RL is for.

So we set the scene in a near-future **San Francisco where every vehicle is
autonomous**, tore out the signals, and asked: can a single shared policy, trained on
the real street graph, drive the whole fleet safely and efficiently?

## The idea

- **One shared policy, many cars.** Every car runs the *same* decentralized policy on
  its own local view of the world (the cars around it, the pedestrians, its route).
  Train one brain; deploy it on any number of cars.
- **Safety is a constraint, not a reward.** We don't bribe the policy to avoid crashes
  with a penalty term and hope the weights balance. We formulate the whole thing as a
  **Constrained MDP**: the reward is *only* efficiency (make progress, complete trips),
  and all safety lives in a separate **cost channel** that a **deterministic verifier**
  computes from the rules of the road. A Lagrangian trainer drives that cost toward
  zero. (Why this matters is the whole story of the project — see *Challenges*.)
- **Real map, real demand.** The world is the actual OSM street graph of SF
  neighborhoods (downtown, Mission, NoPa, Chinatown/FiDi), with car counts grounded in
  real SF travel data, not made-up numbers.
- **A demo you can watch.** Trained checkpoints render into a **3D Cesium viewer** of
  San Francisco — cars flowing through real streets, slowing for pedestrians, with a
  live telemetry dashboard.

---

## How it works

The pipeline, end to end:

```
real SF OSM graph ──▶ JAX kinematic-bicycle multi-agent env ──▶ Deep-Sets observation
                                                                        │
            deterministic verifier (the "rules") ──┐                    ▼
            efficiency reward ────────────────────┐ │       PPO + dual-Lagrangian trainer
                                                  ▼ ▼                    │
                                       reward − λ_hard·cost_hard         │  (on Modal GPUs)
                                              − λ_soft·cost_soft         ▼
                                                              trained checkpoint (.msgpack)
                                                                        │
                                              eval (held-out region) ◀──┤
                                                                        ▼
                                            3D Cesium viewer scene  ◀── render
```

### The environment (the RL substrate)

`smoothride/env/kinematic.py` — a **vectorized JAX kinematic-bicycle** multi-agent sim
on the real SF graph. It's deliberately *not* a rigid-body physics sim: a cheap,
`jit`/`vmap`-able kinematic model lets us run **thousands of cars per step on CPU** and
hundreds of parallel worlds for fast RL.

- **Cars** follow routes (precomputed shortest paths over the OSM graph, handed to the
  env as fixed-size waypoint arrays), holding a lane offset to the right of centerline,
  capped at each road's real speed limit, able to brake to a full stop and resume.
- **Pedestrians** are a second agent class following **deterministic hard-coded paths**
  (a sidewalk run plus a perpendicular crossing *at an intersection*), with only their
  start time randomized — so the policy faces a reproducible yielding problem.
- **Observation** (per car, ego-relative, permutation-invariant): the car's own state
  plus **Deep-Sets encoders** over the nearby cars and pedestrians (padded + masked
  sets). Permutation-invariance over a *set* of neighbors is what lets one policy
  generalize across maps and densities it never trained on.

### Reward vs. cost (the CMDP)

- **Reward** (`smoothride/rl/ppo.py`, efficiency only): `progress + arrival − time`.
  Nothing about safety lives here.
- **Cost** (`smoothride/rl/verifier.py`, pure & deterministic — the rules of the road),
  a **dual channel**:
  - **hard** = actual collisions (`car_crash` + `ped_hit`, pedestrian hits weighted
    heavier) → Lagrangian target **0**.
  - **soft** = graded terms: a **car-collision-risk hinge** (a dense "back off before
    you hit" gradient — the single biggest lever for eliminating car–car crashes), a
    **pedestrian-yield hinge** (slow down near crossers), lane-keeping, wrong-way, and
    over-speed.

Because the verifier is pure and deterministic, the safety signal is auditable: a crash
or a yield violation is a fact computed from positions, not a number baked into a reward.

### The trainer

PPO with a **dual-Lagrangian** objective: `reward − λ_hard·cost_hard − λ_soft·cost_soft`,
where the λ multipliers *ascend automatically* (dual gradient) toward their cost targets.
The policy is shared across all cars (homogeneous agents); a centralized critic sees a
scene summary (CTDE). `--regions a,b,c` round-robins neighborhoods per iteration so the
policy generalizes leave-one-out to a region it never saw. Training runs on **Modal
GPUs**; eval and rendering run locally.

---

## Challenges & iterations

The honest arc (full log in [`DEVLOG.md`](DEVLOG.md)). Almost every real gain came from
fixing the *environment* or the *problem formulation*, not from tuning reward weights.

1. **Reward hacking.** Early on, adding a respawn-and-keep-driving loop made throughput
   climb — and crashes climb with it. The policy happily traded collisions for progress
   reward. Classic multi-objective failure: a sparse crash penalty can't outweigh dense
   progress reward.
2. **Some crashes were geometry, not policy.** Every car tracked the road *centerline*,
   so two cars going opposite ways on a two-way street aimed down the identical line — a
   guaranteed head-on no policy could dodge. **Fix: a lateral lane offset.** Crashes
   dropped immediately. Lesson: if the world makes a collision physically unavoidable,
   no amount of training fixes it.
3. **The density wall.** A policy trained at 24 cars crashed 76% of the time at
   2,000-car density. Several "crashes" turned out to be *bugs* — crashed cars freezing
   into permanent obstacles, a collision radius wider than a lane, respawns teleporting
   cars on top of each other. Fixing those revealed the true baseline was ~1.3
   crashes/car, not ~4.
4. **Hand-built safety filters hit a wall.** We tried wrapping the policy in classical
   safety filters — a braking shield, then a reciprocal velocity-obstacle (ORCA) filter,
   then a higher-order **CBF-QP**. The brake shield was *net-negative* (sudden stops
   propagate into pile-ups); the CBF-QP helped a little. The deep reason: cars are
   **non-holonomic** — at low speed they can't execute the lateral dodge a filter
   computes. A runtime filter can only ever be a last-resort backstop; **the policy
   itself has to learn to avoid conflict states.**
5. **PPO-Lagrangian broke the plateau (the lever).** Pure penalty-PPO plateaued. Treating
   crashes as a **constraint** with an adaptive multiplier — zeroing the fixed crash
   weight and letting λ climb via dual ascent toward a target crash rate — drove crashes
   down steadily while throughput held. The policy *itself* learned to avoid conflicts.
6. **The CMDP reframe + deterministic verifier.** We pulled *all* safety out of the
   reward into a pure, deterministic verifier (the "rules") feeding a dual cost channel
   (hard collisions → target 0; soft graded risk/yield). The graded **collision-risk
   hinge** — turning the binary crash signal into a dense gradient — was the biggest
   single lever for car–car crashes.
7. **Generalization, not memorization.** Deep-Sets observations + multi-region
   round-robin training got us a policy that holds up **leave-one-out** on a neighborhood
   it never trained on (Mission), ~12× safer than the v1 baseline.

---

## Results (v2)

| Metric | Result | Model (checkpoint in Modal volume `smoothride-nav-ckpts`) |
|---|---|---|
| **In-distribution crashes** | **0.07% / car** (~1 per 1,400) | `trained_v5c96p5x` (96 cars / 5 peds, downtown) |
| **Held-out generalization (leave-one-out)** | **~1% / car** on unseen Mission | `trained_v4loo` (trained downtown+nopa+chinatown) |
| v1 baseline (for contrast) | 12% on Mission | `trained_peds` |

≈ **12× cross-map safety improvement** over v1. Both the ≤0.5% in-distribution target
and cross-region generalization were achieved.

**The density frontier (key planning takeaway).** Near-zero crash requires **≤ ~96 cars
AND ≤ ~5 pedestrians** in the downtown bbox. There are two independent walls — car–car
(~300 cars saturates the graph → ~50% crash) and car–ped (~300 crossing pedestrians →
high car–ped crashes). The honest safe operating point is **~80–100 cars + a handful of
pedestrians**. Scaling density is an environment/map-scale problem, not a tuning fix.

---

## How to run

Prereqs: a `python3` env with JAX/Flax/Optax (`pip install -e '.[rl]'`), plus `modal`
(authenticated) for training. **Training runs on Modal GPUs; eval, rendering, and the
viewer run locally.**

**Train** (the in-distribution champion config):
```bash
modal run --detach -m smoothride.rl.modal_train \
  --iters 400 --worlds 16 --agents 96 --n-peds 5 --steps 250 \
  --crash-target 0.0 --soft-target 0.05 --w-carped 8.0 --cruise-cap 4.0 \
  --region downtown --tag _demo --snapshot-every 100
```
Multi-region leave-one-out: swap `--region` for `--regions downtown,nopa,chinatown_fidi`.

**Evaluate on a held-out region** (reports arrivals + crash / off-lane / wrong-way):
```bash
modal volume get smoothride-nav-ckpts trained_v4loo.msgpack runs/trained_v4loo.msgpack
cp runs/trained_v4loo.msgpack runs/untrained_v4loo.msgpack   # eval compares trained vs untrained
python3 scripts/eval_policy.py --region mission --agents 96 --peds 10 --steps 250 \
  --trained runs/trained_v4loo.msgpack --untrained runs/untrained_v4loo.msgpack
```

**Run the 3D viewer locally** (see the next section for what you're looking at):
```bash
# serve from smoothride/demo so the viewer's data path (../web/public/) resolves
python3 -m http.server 8141 --directory smoothride/demo
# open http://127.0.0.1:8141/cesium/index.html   (landing page: /landing.html)
```
The Cesium ion token is embedded in `cesium/app.js`, so 3D terrain + buildings render
with zero setup. Named regions live in `smoothride/data/map_loader.py::SF_REGIONS`
(downtown, mission, nopa, chinatown_fidi).

**Deploy** (Vercel, static — no build step): `vercel.json` at the repo root serves
`smoothride/demo/` (`/` → landing page, `/sim` → 3D viewer). See
[`smoothride/demo/DEPLOY.md`](smoothride/demo/DEPLOY.md).

---

## The 3D viewer (Nomos Cesium)

`smoothride/demo/cesium/` — a static Cesium site rendering San Francisco in 3D
(Cesium World Terrain + OSM Buildings), with cars driven along the exported RL
trajectories and a live telemetry dashboard (trips, crashes/car, fleet status, speed).
Cars are colored by state (red = crashed, lingering 3 s before removal; green =
arrived, ghosting out once the trip is done; blue = en-route, brighter = faster);
pedestrians are small 3D figures (amber dots from altitude).

Two data paths feed it (both selectable in the viewer's *Policy checkpoint* dropdown):
- **`public/scene_*.json` + `public/manifest.json`** — self-contained scenes (roads +
  buildings + full trajectory, including **real crossing pedestrians**) produced by
  `scripts/export_snapshots.py`. The viewer **defaults to the champion Mission scene**
  (96 cars / 10 real peds, leave-one-out) — the headline demo — and the dropdown
  scrubs the downtown training progression (iter 0 → 299). See
  [`smoothride/demo/cesium/SCENES.md`](smoothride/demo/cesium/SCENES.md).
- **`web/public/trajectories.json`** — a synthetic lane-following ambient-traffic
  demo (not an RL rollout), kept as the explicitly-labeled last dropdown entry.

Useful URL params: `?lite=1` (meeting mode: lighter caches + plain buildings),
`?cars=N` (cap the fleet), `?scene=champion` (initial scene). Full list in
[`smoothride/demo/cesium/README.md`](smoothride/demo/cesium/README.md).

---

## Repo map

- `smoothride/env/` — the env (`kinematic.py`), pedestrian paths (`ped_paths.py`),
  routing, spatial hash, map loader.
- `smoothride/rl/` — `verifier.py` (rules / cost), `ppo.py` (dual-Lagrangian trainer),
  `networks.py` (Deep Sets + attention), `modal_train.py` (Modal entry + CLI).
- `smoothride/data/` — OSM map loader + SF travel-demand model.
- `smoothride/demo/` — the Cesium 3D viewer, landing page, and scene exporters.
- `smoothride/worldsim/` — experimental rigid-body (Newton/MuJoCo) physics path.
- `scripts/` — `eval_policy.py` (held-out eval), `export_snapshots.py`, smoke tests.
- `tests/` — pytest suite (env, rl, data, demo); ~170+ tests.
- `DEVLOG.md` — the full build log, newest at the bottom. `REALISM.md`,
  `RESEARCH_SAFETY.md`, `STACK.md` — design notes. `docs/internal/` — archived
  experiment/handoff logs.

---

## What's next

- Re-export `trajectories.json` *with* the env's real crossing pedestrians so the default
  viewer shows yielding behavior directly (today it falls back to a synthetic crowd).
- Push held-out crash below ~1% or support higher density — the lever is
  environment/map scale, not more cost tuning (the frontier is density-bound).

---

## References

The architecture composes a handful of well-established methods. Each row maps a piece
of Nomos to the paper it's based on.

**Policy network** (`smoothride/rl/networks.py`) — a shared actor + centralized critic
(CTDE) over permutation-invariant sets of neighbors:

| Piece | Paper |
|---|---|
| `DeepSets` neighbor encoder (masked mean+max pool — the production architecture) | Zaheer et al., *Deep Sets*, NeurIPS 2017 — [1703.06114](https://arxiv.org/abs/1703.06114) |
| `AttentionPool` ego-query set encoder (PMA / social-attention; tested, not better than Deep Sets) | Lee et al., *Set Transformer*, ICML 2019 — [1810.00825](https://arxiv.org/abs/1810.00825); Vaswani et al., *Attention Is All You Need*, 2017 — [1706.03762](https://arxiv.org/abs/1706.03762) |
| Centralized critic / CTDE (scene summary at train time, local obs at run time) | Lowe et al., *MADDPG*, NeurIPS 2017 — [1706.02275](https://arxiv.org/abs/1706.02275) |

**Trainer** (`smoothride/rl/ppo.py`) — multi-agent PPO with a dual-Lagrangian constraint:

| Piece | Paper |
|---|---|
| MAPPO / IPPO (shared-parameter multi-agent PPO) | Yu et al., *The Surprising Effectiveness of PPO in Cooperative MARL*, NeurIPS 2022 — [2103.01955](https://arxiv.org/abs/2103.01955) |
| Base PPO | Schulman et al., *Proximal Policy Optimization*, 2017 — [1707.06347](https://arxiv.org/abs/1707.06347) |
| Constrained MDP — crashes as a constraint, adaptive λ via dual ascent | Achiam et al., *Constrained Policy Optimization*, ICML 2017 — [1705.10528](https://arxiv.org/abs/1705.10528); Ray et al., *Benchmarking Safe Exploration* (Safety Gym / PPO-Lagrangian), 2019; Stooke et al., *Responsive Safety via PID-Lagrangian*, ICML 2020 — [2007.03964](https://arxiv.org/abs/2007.03964) |
| JaxMARL — the JAX multi-agent RL framework Nomos is built on | Rutherford et al., *JaxMARL*, 2023 — [2311.10090](https://arxiv.org/abs/2311.10090) |

The offline adversarial scenario generator (`smoothride/diffusion/`) follows the
Safe-Sim / MotionDiffuser line of diffusion-based traffic-scenario synthesis; the
constrained-safety design rationale (CBF-QP / RSS as a runtime backstop vs. learned
avoidance) is written up in [`RESEARCH_SAFETY.md`](RESEARCH_SAFETY.md).
