# Demo runbook — 20-minute interview walkthrough

## Start (once, before the call)

```bash
cd ~/Developer/driving
python3 scripts/serve_demo.py        # no-cache server on port 8141
```

- Landing page: <http://127.0.0.1:8141/landing.html>
- 3D sim: <http://127.0.0.1:8141/cesium/index.html>
- If the machine feels loaded while screen-sharing, add `?lite=1`
  (coarser buildings, smaller caches) and/or `?cars=48`.

Do a dry run 10 minutes before: open the sim, confirm the *Policy checkpoint*
dropdown shows **CHAMPION v4loo — held-out Mission** and cars are flowing.
Close other heavy tabs; the sim tab settles around 0.5 GB JS heap / 60 fps.

## Suggested 20-minute arc

1. **Landing page (2 min)** — thesis: remove human drivers and the traffic
   protocol (lights, signs) is legacy; the unsolved layer is fleet *behavior*.
   Scroll: problem → solution → simulator → results.
2. **Champion scene (8 min)** — the default scene is the trained v4loo policy on
   **Mission, a district it never trained on** (leave-one-out). Zoom to an
   intersection: cars queue, negotiate, slow near crossing pedestrians (amber
   dots → 3D figures up close). Color language: blue en-route, green trip done
   (ghosts out — no longer an obstacle), red crash (removed after 3 s).
   Telemetry: cumulative trips climbing, crashes staying ~2/96 (~1–2% — vs 12%
   for the v1 baseline on the same held-out map).
   For more pedestrian action, switch to *"Mission, busy sidewalks (24 peds)"* —
   same checkpoint, 2.4× the pedestrians. Crashes rise to 6/96 (~6%): that's the
   **density frontier** finding demonstrated live (the policy trained around
   ~5 peds), a strong honest talking point rather than a weakness.
3. **Watch it learn (5 min)** — dropdown → *iter 0 (baseline)*: barely moves,
   crashes. Step through iter 50 → 299: throughput rises, crashes fall. This is
   the PPO-Lagrangian story: crashes are a *constraint* (dual-ascent λ), not a
   reward term.
4. **Architecture talk-track (5 min)** — real OSM graph → JAX kinematic env
   (thousands of cars/step on CPU) → Deep-Sets obs → PPO + dual Lagrangian on
   Modal GPUs → deterministic verifier as the safety source of truth →
   checkpoints render into this viewer. README's pipeline diagram mirrors this.

## Q&A ammo

- **"Is this the real policy?"** Yes — every `scene_*.json` is a rollout of a
  saved checkpoint (`scripts/export_snapshots.py`). The last dropdown entry
  ("ambient traffic — synthetic, not RL") is honest set-dressing: procedural
  lane-following used for map/visual work, and it's labeled as such.
- **Density frontier**: near-zero crash needs ≤ ~96 cars + ~5 peds in the
  downtown bbox; ~300 cars saturates the graph (~50% crash). Scaling density is
  an env/map-scale problem, not a tuning fix.
- **Why kinematic, not rigid-body?** `jit`/`vmap`-able bicycle model = hundreds
  of parallel worlds for RL; physics fidelity isn't the bottleneck for the
  behavioral layer.
- **Failed approaches** (good discussion): brake shield was net-negative
  (pile-ups), ORCA/CBF-QP limited by non-holonomic cars → the policy itself has
  to learn conflict avoidance.

## Overnight dense-pedestrian results (already bundled)

A 288-ped (3 peds/car) stress-test scene is bundled in the dropdown:
*"v7 dense — Mission, 288 peds (3/car stress test)"* — 18/96 crashes vs ~50%
for the 5-ped champion at that density. Talking points:

- Trained overnight at 288 peds (config sweep: strict constraint weights
  converged, permissive ones collapsed to 52% crashes — a controlled comparison).
- The env now removes finished pedestrians (they stop being obstacles), same
  rule as remove-on-arrival for cars; standing finished peds used to project a
  permanent 3.5 m keep-out into the road edge.
- A 1000-iter run (`_v9d288long`) tested whether more compute breaks the wall:
  **it doesn't.** Training plateaued at ~0.12 crashes/car from iter 600 on, and
  its Mission renders (30–35 crashed over 3 seeds) never beat the bundled v7
  scene (18). This is the README's density-frontier conclusion, re-confirmed
  empirically overnight: past ~5 peds the limit is env/map-scale, not tuning or
  compute. Say exactly that if asked — it's a strength, not a gap.

The demo stands as bundled: champion (10 peds) headline, busy sidewalks
(24 peds), the dense 288-ped stress test, and the v1 *iter 299* scene
(300 peds downtown, 6 crashes — proof density is learnable in-distribution).
All overnight checkpoints/snapshots remain in the Modal volume
(`smoothride-nav-ckpts`, tags `_v7d288`, `_v7d288b`, `_v8d288`, `_v9d288long`).

## If something breaks

- **Viewer black / tiles missing**: network hiccup on the Cesium CDN — reload;
  worst case add `?lite=1`. Scenes themselves are local files.
- **Laggy while screen-sharing**: `?lite=1&cars=48`, quit other GPU apps.
- **Wrong scene loads**: pick it in the dropdown, or force via `?scene=champion`.
- Landing page and sim are static — restarting `http.server` fixes any stale
  serving.
