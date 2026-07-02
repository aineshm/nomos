# Nomos — Cesium 3D viewer

Replays pre-rendered RL rollouts (`public/scene_*.json`, schema v1) on a real 3D
San Francisco: Cesium World Terrain + OSM Buildings, GLB car models, 3D
pedestrians, and a live telemetry dashboard.

## Quick start
```bash
# serve from smoothride/demo so the legacy data path (../web/public/) also resolves
python3 -m http.server 8141 --directory smoothride/demo
# open http://127.0.0.1:8141/cesium/index.html
```
A default Cesium ion token is embedded in `app.js`, so 3D terrain + buildings
render with zero setup (override: `?ionToken=` or a git-ignored `config.js` from
`config.example.js`).

The viewer loads `public/manifest.json` and defaults to the **champion held-out
Mission** scene; the *Policy checkpoint* dropdown switches between the training
progression (iter 0 → 299), the champion run, and a synthetic ambient-traffic
demo (explicitly labeled "not RL").

## Reading the viewer
- **Blue** car = en-route (brighter = faster). **Green** = trip complete (ghosts
  out after a few seconds — no longer an obstacle). **Red** = crashed; the wreck
  lingers 3 s, then is removed (mirroring the env).
- **Amber dots** = pedestrians (resolve into small 3D figures up close).
- Cars render as GLB models near the camera and cross-fade to dots at distance.
- The dashboard recomputes trips / moving / crashes / speeds per frame.

## URL parameters
| Param | Effect |
|---|---|
| `?scene=<substring>` | pick the initial scene (matches file or label, e.g. `?scene=champion`) |
| `?cars=N` | cap the rendered fleet (HUD follows) |
| `?lite=1` | meeting mode: coarser buildings, smaller caches, no facade skin |
| `?sse=N` | building detail (lower = sharper, slower; default 20) |
| `?skin=tile\|single\|off` | facade image mode |
| `?roads=1` | overlay the road graph the env drives on |
| `?track=N` | chase car N |
| `?lon=&lat=&alt=&pitch=&heading=` | camera framing (capture harness) |
| `?t=<frame>&pause=1` | jump to a frame / freeze time |

## Performance notes
Car/ped heights are **baked once per scene load** (batched
`sampleTerrainMostDetailed`, lerped per frame) instead of per-frame
`CLAMP_TO_GROUND` — per-frame clamping cost a scene pick per entity per frame
(~7 fps at 170 cars; ~60 fps baked). The OSM Buildings tile cache is capped at
256 MB (96 MB in `?lite=1`).

## Render a new scene from a checkpoint
```bash
python -m smoothride.demo.export_cesium \
  --trained runs/trained.msgpack --untrained runs/untrained.msgpack \
  --elevation 3dep --buildings --out smoothride/demo/cesium/public/scene_my.json
# then add an entry to public/manifest.json
```
See `SCENES.md` for the bundled scenes and the full re-render recipe.

## The contract
Scene files are schema v1 (`smoothride/demo/scene.py`). Any sim backend that
emits this format (kinematic now, Isaac/PhysX later) replays in this viewer
unchanged. See `docs/internal/HANDOFF-sim-contract.md`.
