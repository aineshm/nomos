# Merge plan — reconciling `worktree-3d-sim-setup` with `origin/main`

**Purpose.** Log, per component, **which branch is the source of truth** and **what must change** to merge the two. Both branches built a 3D Cesium demo independently from an empty common ancestor (`dfc67b9` had no `cesium/`), so most shared files are **add/add conflicts**, not line merges — this doc decides direction up front.

**Branches**
- **OURS** = `worktree-3d-sim-setup` (pushed to `origin/worktree-3d-sim-setup`), based at `dfc67b9`. Contract-first: `scene.json` schema v1, finite-cohort env, state-coloured box cars.
- **MAIN** = `origin/main` (`a270073`). Visual-first: GLB cars + OSM Buildings + facade murals, reads older `trajectories.json`; adds `worldsim/` 3D physics, `legality.py`, respawn fix.

**User decisions captured (2026-06-20):** take MAIN's **GLB models + OSM Buildings**; keep OURS' **data file + envelope (`scene.json`) + orientation**; everything else on OURS is correct. Ion token will be provided (needed for OSM Buildings + World Terrain).

---

## Decision table

| Component | Source of truth | What must change to merge |
|---|---|---|
| **Data contract** (`scene.json` schema v1, `demo/scene.py`) | **OURS** | None. Keep `{schema_version, meta, roads, buildings, worlds{summary,trips_series,cars,peds}}`; cars carry `lng,lat,z,hdg,spd,crash,arr`. This is the superset (MAIN's cars are `lng,lat,hdg,spd,crash` — no `z`, no `arr`). |
| **Exporter** (`demo/export_cesium.py`) | **OURS** | None. Keep ours (bakes `z`, threads `arr`). Drop MAIN's `worldsim/export_cesium.py` as the demo exporter (or retarget it to emit schema v1 — see Physics). |
| **Car geometry** | **MAIN** (GLB sedan/suv/coupe + `palette.json`) | Port GLB model rendering into the viewer **but read `scene.json`** (see Viewer). |
| **Car colour** | **OURS** (red=crashed / green=arrived / blue=en-route) | Apply our **state colour as a tint on the GLB model** (`model.color` + `colorBlendMode`), instead of MAIN's random palette. GLB gives the shape; `crash`/`arr` give the colour. |
| **Buildings** | **MAIN** (Cesium OSM Buildings + facade murals `cesium-murals.js`, `building sides/*.jpg`) | Requires an ion token. Keep our extruded-GeoJSON path as the **no-token fallback**. |
| **Terrain (z-axis)** | **MAIN** approach (Cesium World Terrain, token) + **OURS** baked `z` | With token: World Terrain. Cars sit on baked `z` from `scene.json` either way (clamp model to terrain when token present). |
| **Orientation** | **EITHER** (convergent) | None. Both independently use `HeadingPitchRoll(-hdg,0,0)`. Keep. |
| **Token handling** | pick one | MAIN uses `?ionToken=`/`window.CESIUM_ION_TOKEN`; OURS uses `config.js`/`window.SMOOTHRIDE_CONFIG`. Standardize on `config.js` (gitignored) and also accept `?ionToken=` for convenience. |
| **Env: respawn / cohort** (`env/kinematic.py`) | **OURS** | Keep finite-cohort **remove-on-arrival** + **non-overlapping spawn placement** (root-cause). MAIN's *continuous-respawn* + collision-partner-mask + merge-speed is a **different model**, superseded by the finite cohort. Do NOT take MAIN's respawn block. (MAIN's collision-partner masking idea is already present in ours as the `done`/`immune` exclusion.) |
| **Env: traffic-law shaping** (`legality.py`, `w_offlane`/`w_wrongway`) | **MAIN** | Bring `env/legality.py` over (off-lane / wrong-way geometry). It's the off-lane/wrong-way **predicate the verifier needs** — see Verifier. Adding the reward weights is optional (CMDP reframe routes these through *cost*, not reward). |
| **3D physics** (`worldsim/`) | **MAIN** | Adopt as the physics backend. It already emits the **same `State`/trace** as the kinematic env, so it feeds the SAME exporter once it also bakes `z` + emits `arr`. See Physics. |
| **Verifier** | **`rl-verifier` worktree** (being built) | `legality.py` ≈ the verifier's off-lane/wrong-way predicate. Reconcile: the verifier reads the trace and produces `RunVerdict`; legality's geometry becomes its off-lane/wrong-way check. Don't duplicate. |
| **Perf** (`a270073`: JAX persistent compile cache, cached OSM footprint bounds, tuned OSM Buildings LOD) | **MAIN** | Take wholesale — pure speedups, no conflict with our contract. `jaxcfg.py` + the building-bounds cache are additive. |
| **Misc viewer files** (`index.html`, `style.css`, `config.example.js`) | **merge by hand** | add/add; combine MAIN's GLB/mural markup with our HUD (cars / arrived / crashed live counters). |

---

## Work items (the actual changes)

### A. Viewer bridge — port MAIN's visuals onto OURS' contract (one-directional)
MAIN's car loop already reads `c.hdg/c.lng/c.lat/c.spd/c.crash`; our `scene.json` cars are a **superset**. So:
1. Point the viewer at `public/scene.json` (not `../web/public/trajectories.json`); read `scene.worlds[WORLD].cars`.
2. Render each car as a **GLB model** (random body from `palette.json`) **tinted by `carColor`** (red/green/blue from `crash`/`arr`), positioned at baked `z`, oriented `-hdg`.
3. Buildings: **OSM Buildings + murals** when a token is present; **extruded GeoJSON** (ours) as the no-token fallback.
4. Standardize token via `config.js`, also honor `?ionToken=`.
5. Merge HUDs: keep our live cars / arrived / crashed counters.

### B. Env — keep OURS as-is
- Finite-cohort remove-on-arrival, non-overlapping spawns, state colours. No change. **Do not** import MAIN's respawn block.
- **Add** `env/legality.py` from MAIN (off-lane/wrong-way geometry) for the verifier to consume.

### C. Physics — make `worldsim` emit the contract
- `worldsim/physics_state.py` already maps Newton/MuJoCo → kinematic-style `tr` (pos/heading/speed/crashed/goals/ped).
- To be a drop-in for `scene.json`: it must also (a) bake `z` (our `export_cesium` already does this from `net.node_z` — reuse it) and (b) carry `arrived` (finite cohort). Cleanest: run the **kinematic env for arrival/cohort bookkeeping** and worldsim for **pose**, or add arrival+freeze to the worldsim rollout. **Open question — needs a GPU box to verify Newton anyway.**

### D. Verifier ↔ legality
- Verifier (in `rl-verifier`, off `dfc67b9`) is the deterministic grader over the trace (`RunVerdict`: arrived/travel_time/crash/off_road/rule).
- `legality.py` provides the off-lane / wrong-way **geometry**; fold it in as the verifier's predicate rather than re-deriving. The user's framing "legality is basically the verifier" holds for the *off-lane/wrong-way* slice; the verifier is broader (arrival, travel-time, crash, validity).

---

## Open questions / dependencies
1. **Ion token** — required for OSM Buildings + World Terrain (the chosen visual). User providing. Until then, no-token GeoJSON fallback renders (dark buildings).
2. **Newton runtime** — `worldsim` physics is MuJoCo-validated but Newton/Warp is unverified; needs a GPU box. Physics merge (item C) is gated on that.
3. **Verifier interface freeze** — once `rl-verifier` lands its `Trace`/`verify()` shape, confirm `legality.py` slots in as the off-lane/wrong-way predicate.
4. **Merge mechanics** — every shared `cesium/` filename is add/add. Plan is to take MAIN's files as the base for the *viewer* and re-apply items A1–A5, rather than git-merging line by line.

---

## One-line summary
Keep **OURS** for the data contract + env (finite cohort, spawns, colours, exporter); take **MAIN** for the visuals (GLB cars, OSM Buildings, murals), the perf cache, `legality.py`, and the `worldsim` physics backend — bridging them by pointing MAIN's GLB/building rendering at OUR `scene.json` and tinting GLB cars by our `crash`/`arr` state.
