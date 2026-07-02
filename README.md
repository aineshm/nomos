# Nomos — reinforcement-learned traffic coordination for a driverless San Francisco

![Nomos — orbiting the trained fleet over downtown San Francisco](docs/media/nomos-aerial.gif)

![Nomos — street-level dolly through the fleet](docs/media/nomos-street-dolly.gif)

*The trained policy driving the live [Cesium viewer](smoothride/demo/cesium/) on the real
SF grid — 172 cars, per-car colors, building facades, and a live telemetry dashboard.*

When every vehicle drives itself, traffic lights and lane law are legacy overhead for
human limitations. Nomos rips them out and learns the **behavioral layer** instead — how
a whole fleet shares the real San Francisco road graph without a central controller and
without crashing. Low-level autonomy (perception, throttle/brake/steer) is assumed
solved; the one irreducible human element is **pedestrians**, whom cars must yield to.

---

## The RL, in brief

**Formulation — decentralized multi-agent control.** Each car is an agent with a *local*
observation (its own kinematic state, route waypoints, and nearby cars/pedestrians as
masked sets). Actions are **(accelerate/brake, steer, lane-change)** on a kinematic
bicycle model, capped by each road's real speed limit. It's a Dec-POMDP: no agent sees
global state at run time.

**Algorithm — MAPPO/IPPO under CTDE.** *Centralized training, decentralized execution.* A
centralized critic sees a scene summary during training; each car acts from local obs at
run time. The policy is **shared-parameter** across homogeneous agents, so it scales to
many cars cheaply, and is trained with PPO + GAE across many parallel JAX worlds. CTDE is
the standard answer to the core multi-agent problem — **non-stationarity**, where the
environment shifts under each agent as the others learn.

**Reward vs. cost — a Constrained MDP, not a weighted sum.** The reward is *efficiency
only*: `progress + arrival − time`. Nothing about safety lives in it. All safety flows
through a separate **cost channel** computed by a **deterministic verifier** — a pure
function over the logged trajectory that scores the rules of the road:

| channel | terms | constraint target |
|---|---|---|
| hard | car–car collisions, pedestrian hits (weighted heavier) | **0** |
| soft | collision-risk hinge, pedestrian-yield hinge, lane-keeping, wrong-way, over-speed | small budget |

A **dual-Lagrangian PPO** objective (`reward − λ_hard·cost_hard − λ_soft·cost_soft`)
drives both costs to their targets — the λ multipliers rise automatically via dual
ascent, so safety is *enforced*, never traded against throughput by hand-tuned weights.
(An earlier multi-objective reward did exactly that trade — the policy farmed progress
while crashing — which is why the CMDP reframe exists.)

**Safety filters — evaluated, then retired to a backstop.** Classical runtime filters
(brake shield, ORCA, CBF-QP — see `rl/cbf.py` / `rl/safety.py`) were tried: the brake
shield was net-negative (sudden stops cascade into pile-ups) and the CBF-QP is limited
by non-holonomic cars that can't execute a lateral dodge at low speed. The policy itself
learns to avoid conflict states; the write-up is in
[`docs/RESEARCH_SAFETY.md`](docs/RESEARCH_SAFETY.md).

**Network.** Permutation-invariant **Deep Sets** encoders over the variable neighbor and
pedestrian sets (an attention pool was tested and didn't beat it), so the policy is
agnostic to how many cars are nearby — the property that lets one policy generalize
across maps and densities.

The reasoning behind each choice lives in [`components/`](components/) (one JSON per
decision) and [`docs/`](docs/).

---

## Run it

```bash
pip install -e '.[rl,data,dev]'          # JAX/Flax/Optax + OSM tooling + pytest
python -m smoothride.rl.train_local      # train the coordination policy locally
python scripts/eval_policy.py            # held-out evaluation
pytest                                   # ~180 tests (env, rl, data, demo)
```

**Run the 3D viewer locally** (see the next section for what you're looking at):
```bash
python3 scripts/serve_demo.py
# open http://127.0.0.1:8141/cesium/index.html   (landing page: /landing.html)
```
The Cesium ion token is embedded in `cesium/app.js`, so 3D terrain + buildings render
with zero setup. Named regions live in `smoothride/data/map_loader.py::SF_REGIONS`
(downtown, mission, nopa, chinatown_fidi). Deployed (Vercel, static): the site root
(`/`) is the 3D viewer.

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
  (96 cars / 10 real peds, leave-out) — the headline demo — and the dropdown
  scrubs the downtown training progression (iter 0 → 299). See
  [`smoothride/demo/cesium/SCENES.md`](smoothride/demo/cesium/SCENES.md).
- **`public/trajectories.json`** — a synthetic lane-following ambient-traffic
  demo (not an RL rollout), kept as the explicitly-labeled last dropdown entry.

Useful URL params: `?lite=1` (meeting mode: lighter caches + plain buildings),
`?cars=N` (cap the fleet), `?scene=champion` (initial scene). Full list in
[`smoothride/demo/cesium/README.md`](smoothride/demo/cesium/README.md).

---

## Repo map

- `smoothride/env/` — JAX kinematic env, routing, spatial hash, pedestrian paths.
- `smoothride/rl/` — `ppo.py` (MAPPO + dual-Lagrangian), `verifier.py` (the rules /
  cost channel), `networks.py` (Deep Sets + attention), `modal_train.py`,
  `cbf.py`/`safety.py` (retired classical-filter experiments).
- `smoothride/data/` — OSM map loader + SF travel-demand model.
- `smoothride/demo/` — the Cesium 3D viewer, landing page, and scene exporters.
- `smoothride/worldsim/` — experimental rigid-body physics path.
- `components/` — the design decisions, one JSON per component.
- `docs/` — [`DEVLOG.md`](docs/DEVLOG.md) (full build log), `REALISM.md`,
  `RESEARCH_SAFETY.md`, `STACK.md`, and `internal/` handoff notes.
- `scripts/`, `tests/` — eval/export utilities and the pytest suite (~180 tests).

---

## What's next

- Push held-out crash below ~1% or support higher density — the lever is
  environment/map scale, not more cost tuning. An overnight sweep at 288 peds
  (3 per car) confirmed it empirically: strict constraint weights, a retrain, and
  2.5× compute all plateau at the same wall (~19% crashes best on held-out Mission).

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
avoidance) is written up in [`docs/RESEARCH_SAFETY.md`](docs/RESEARCH_SAFETY.md).
