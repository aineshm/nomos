"""Render a series of versioned training checkpoints into viewer scene files.

Discovers ``trained{tag}_it####.msgpack`` files locally, builds the map and env
ONCE, then renders each checkpoint as a self-contained scene JSON that the Cesium
viewer can load.  Also writes a ``manifest.json`` the viewer uses to navigate
between snapshots.

Typical workflow
----------------
1. Pull checkpoints from the Modal volume::

     modal volume get smoothride-nav-ckpts 'trained_pedtest_it*.msgpack' runs/

2. Render all snapshots::

     python scripts/export_snapshots.py \\
         --tag _pedtest --region downtown \\
         --agents 40 --n-peds 100 --steps 150 --elevation synthetic

3. Open the viewer — it reads ``manifest.json`` next to the scene files.

Pure render — no Modal calls are made from this script.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Pure, unit-testable helpers
# ---------------------------------------------------------------------------

_ITER_RE = re.compile(r"trained.*_it(\d+)\.msgpack$")


def parse_iter(filename: str) -> int | None:
    """Extract the iteration number from a versioned checkpoint filename.

    Matches ``trained{tag}_it{NNNNN}.msgpack`` (tag may be empty or any string
    that doesn't itself contain ``_it``).  Returns the integer iter, or None if
    the filename does not match the versioned pattern.

    Examples::

        parse_iter("trained_pedtest_it00050.msgpack")  # -> 50
        parse_iter("trained_it00000.msgpack")          # -> 0
        parse_iter("trained.msgpack")                  # -> None
        parse_iter("untrained.msgpack")                # -> None
    """
    m = _ITER_RE.search(filename)
    if m is None:
        return None
    return int(m.group(1))


def build_manifest(iters: list[int]) -> dict:
    """Build the viewer manifest from a list of iteration numbers.

    Returns a dict conforming to the manifest contract::

        {
          "scenes": [
            {"iter": 0,  "file": "scene_it00000.json", "label": "iter 0 (baseline)"},
            {"iter": 50, "file": "scene_it00050.json", "label": "iter 50"},
            ...
          ]
        }

    The list is sorted by iter ascending.  Iter 0 receives the label
    ``"iter 0 (baseline)"``; all others use ``"iter {n}"``.
    """
    def _label(i: int) -> str:
        return "iter 0 (baseline)" if i == 0 else f"iter {i}"

    scenes = [
        {"iter": i, "file": f"scene_it{i:05d}.json", "label": _label(i)}
        for i in sorted(iters)
    ]
    return {"scenes": scenes}


# ---------------------------------------------------------------------------
# Integration shell
# ---------------------------------------------------------------------------

def _discover_checkpoints(ckpt_dir: str, tag: str) -> list[tuple[int, str]]:
    """Return sorted (iter, path) pairs for versioned checkpoints in *ckpt_dir*."""
    pattern = os.path.join(ckpt_dir, f"trained{tag}_it*.msgpack")
    paths = glob.glob(pattern)
    pairs: list[tuple[int, str]] = []
    for p in paths:
        it = parse_iter(os.path.basename(p))
        if it is not None:
            pairs.append((it, p))
    return sorted(pairs)


def main() -> None:
    from smoothride.data.map_loader import SF_REGIONS, attach_elevation, load_road_network
    from smoothride.demo.export_cesium import (
        DEFAULT_OUT,
        build_from_rollouts,
        _roads_3d,
    )
    from smoothride.demo.export_web import _lonlat_transformer, _to_lonlat
    from smoothride.demo import scene as S
    from smoothride.demo.render import load_params, rollout
    from smoothride.env import kinematic as K
    from smoothride.env.routing import build_route_pool

    import jax
    import numpy as np

    default_out_dir = os.path.dirname(DEFAULT_OUT)

    ap = argparse.ArgumentParser(
        description="Render a series of versioned training checkpoints into viewer scenes."
    )
    ap.add_argument("--tag", default="", help="Checkpoint tag, e.g. '_pedtest'")
    ap.add_argument("--region", default="downtown",
                    choices=list(SF_REGIONS.keys()),
                    help="Named SF region for map/env construction")
    ap.add_argument("--agents", type=int, default=24, help="Number of agents")
    ap.add_argument("--n-peds", type=int, default=12, help="Number of pedestrians")
    ap.add_argument("--steps", type=int, default=300, help="Episode length")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride for scene output")
    ap.add_argument("--elevation", default="synthetic", choices=["3dep", "synthetic"],
                    help="Elevation source (synthetic = fully offline)")
    ap.add_argument("--ckpt-dir", default="runs/",
                    help="Local directory containing checkpoint files")
    ap.add_argument("--out-dir", default=default_out_dir,
                    help="Output directory for scene JSONs and manifest")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for rollout")
    args = ap.parse_args()

    # --- Discover checkpoints ---
    pairs = _discover_checkpoints(args.ckpt_dir, args.tag)
    if not pairs:
        print(
            f"No versioned checkpoints found in '{args.ckpt_dir}' for tag='{args.tag}'.\n"
            f"Pull from the Modal volume first, e.g.:\n"
            f"  modal volume get smoothride-nav-ckpts "
            f"'trained{args.tag}_it*.msgpack' {args.ckpt_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(pairs)} checkpoint(s) in '{args.ckpt_dir}':", flush=True)
    for it, p in pairs:
        print(f"  iter {it:>6d}  {p}", flush=True)

    # --- Build map / env ONCE (expensive; reuse across all checkpoints) ---
    print(f"\nBuilding map for region='{args.region}' elevation={args.elevation}...",
          flush=True)
    net = attach_elevation(load_road_network(bbox=SF_REGIONS[args.region]),
                           source=args.elevation)
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1),
                     n_agents=args.agents, n_peds=args.n_peds, max_steps=args.steps)
    tf = _lonlat_transformer(net)

    corners = np.array([[x0, y0], [x1, y1]], np.float32)
    clon, clat = _to_lonlat(net, tf, corners)
    meta = {
        "dt": float(env.dt) * args.stride,
        "n_steps": len(range(0, args.steps, args.stride)),
        "vmax": float(env.v_max),
        "center": [round(float(clon.mean()), 6), round(float(clat.mean()), 6)],
        "bounds": [[round(float(clon[0]), 6), round(float(clat[0]), 6)],
                   [round(float(clon[1]), 6), round(float(clat[1]), 6)]],
        "zoom": 15.5,
    }
    roads_3d = _roads_3d(net, tf)
    buildings: dict = {"type": "FeatureCollection", "features": []}

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Render each checkpoint ---
    written: list[str] = []
    for it, ckpt_path in pairs:
        print(f"\nRendering iter {it} from {ckpt_path}...", flush=True)
        params = load_params(env, ckpt_path)
        tr = rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)
        rollouts = {f"iter_{it:05d}": tr}
        worlds = build_from_rollouts(net, env, tf, rollouts, args.stride)

        scene = S.build_scene(meta=meta, roads=roads_3d, buildings=buildings,
                              worlds=worlds)
        out_path = os.path.join(args.out_dir, f"scene_it{it:05d}.json")
        nbytes = S.write_scene(out_path, scene)
        print(f"  -> {out_path} ({nbytes / 1024:.0f} KB)", flush=True)
        written.append(out_path)

    # --- Write manifest ---
    iters = [it for it, _ in pairs]
    manifest = build_manifest(iters)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")

    print(
        f"\nDone: {len(written)} scene(s) written to '{args.out_dir}'.\n"
        f"Open the viewer and select 'Load manifest' to step through iterations."
    )


if __name__ == "__main__":
    main()
