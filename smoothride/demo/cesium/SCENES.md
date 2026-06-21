# Cesium viewer — bundled demo scenes

These `public/*.json` files are **pre-rendered, self-contained scenes** (each embeds the SF terrain reference, roads, buildings, and the full per-step car/pedestrian trajectory). They are committed so teammates can view the demos **without re-rendering** — just start the viewer and pick a scene from the dropdown.

## View them
```bash
cd smoothride/demo/cesium
python3 -m http.server 8137
# open http://127.0.0.1:8137  → use the "iter" dropdown (top-left HUD) to switch scenes
```
Needs a Cesium ion token in the git-ignored `config.js` (copy `config.example.js` and paste a free token). Without a token the viewer still works (flat ellipsoid + GeoJSON buildings instead of photoreal terrain).

**Reading the viewer:** cars are cones — **red = crashed, green = arrived, blue = en-route** (brighter = faster). Pedestrians are amber cylinders. Press play on the Cesium timeline; zoom into an intersection to watch cars slow for crossing pedestrians and queue without colliding. The HUD shows live cars / trips / crashed counts.

## The dropdown (`manifest.json`)
`manifest.json` lists the scenes and their labels; the viewer builds the dropdown from it and loads the last entry by default. Edit/extend it to add scenes (`{"iter", "file", "label"}`).

## What each scene is

| Dropdown label | File | Model / checkpoint | Region | Cars / Peds | Shows |
|---|---|---|---|---|---|
| iter 0 (baseline) | `scene_it00000.json` | `trained_peds` @ iter 0 (untrained) | downtown | 96 / 300 | Starting point — cars barely move, many crashes |
| iter 50 … 250 | `scene_it000{50,100,150,200,250}.json` | `trained_peds` @ that iteration | downtown | 96 / 300 | **Training progression** — scrub to watch the policy learn |
| iter 299 | `scene_it00299.json` | `trained_peds` (final) | downtown | 96 / 300 | Fully-trained v1 (dense 300-ped downtown) |
| **CHAMPION v4loo — held-out Mission** | `scene_champion_mission.json` | `trained_v4loo` (v2 generalization champion, trained on downtown+nopa+chinatown — **never saw Mission**) | **mission** | 96 / 10 | **Cross-region generalization** — ~1–2% crashes on an unseen neighborhood |

Notes:
- The `iter 0…299` series is the **v1** model (single-cost, dense pedestrians) — kept because it's the clearest "watch it learn" progression for the dropdown.
- The **champion Mission** scene is the **v2** leave-one-out model — the headline generalization result (see top-level `README.md` and `docs/HANDOFF-overnight.md`).
- Scene files are large (~3.6 MB each) and `public/` is otherwise git-ignored; these specific demo scenes were force-added. New ad-hoc renders won't be auto-committed.

## Re-render / add scenes
Checkpoints live in the Modal volume `smoothride-nav-ckpts` (and `runs/` locally). To render a checkpoint to a scene for a region:
```bash
# example: champion v4loo on Mission
modal volume get smoothride-nav-ckpts trained_v4loo.msgpack runs/trained_v4loo.msgpack
mkdir -p runs_demo && cp runs/trained_v4loo.msgpack runs_demo/trained_demo_it00000.msgpack
python3 scripts/export_snapshots.py --region mission --tag _demo \
  --agents 96 --n-peds 10 --steps 250 --elevation synthetic \
  --ckpt-dir runs_demo --out-dir smoothride/demo/cesium/public
# then add an entry to public/manifest.json pointing at the new scene_*.json
```
`scripts/export_snapshots.py` renders a whole series of versioned checkpoints + writes a `manifest.json`; `smoothride/demo/export_cesium.py` renders a single scene. Available regions: `downtown, mission, nopa, chinatown_fidi` (`smoothride/data/map_loader.py::SF_REGIONS`).
