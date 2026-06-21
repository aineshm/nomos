# Nomos — Cesium 3D viewer

Replays a `public/scene.json` (schema v1) of meshed cars on 3D San Francisco.

## Quick start (offline, no token)
```bash
python scripts/smoke_3d.py --out smoothride/demo/cesium/public/scene.json
cd smoothride/demo/cesium && python -m http.server 8000   # open http://localhost:8000
```
Flat terrain + GeoJSON buildings, animated cars on the real SF street grid.

## Real terrain + OSM buildings
1. Get a free token at https://ion.cesium.com/
2. `cp config.example.js config.js` and paste the token.
3. Reload — Cesium World Terrain (SF hills) + 3D OSM Buildings.

## From a trained policy
```bash
python -m smoothride.demo.export_cesium \
  --trained runs/trained.msgpack --untrained runs/untrained.msgpack \
  --elevation 3dep --buildings --out smoothride/demo/cesium/public/scene.json
```

## The contract
`scene.json` is schema v1 (`smoothride/demo/scene.py`). Any sim backend that
emits this format (kinematic now, Isaac/PhysX later) replays in this viewer
unchanged. See `docs/internal/HANDOFF-sim-contract.md`.

## Demo scenes & status
Pre-rendered demo scenes (downtown training progression + champion held-out Mission) are bundled in `public/` and listed in **`SCENES.md`** — start the server and pick from the dropdown. Project status & results: top-level `README.md` and `docs/internal/HANDOFF-overnight.md`.
