"""Offline end-to-end smoke for the 3D pipeline: real SF roads + SYNTHETIC
elevation + a RANDOM policy rollout -> a schema-v1 scene.json the Cesium viewer
can replay. No network beyond the (cached) OSM graph, no checkpoints, no GPU.

  python scripts/smoke_3d.py --out smoothride/demo/cesium/public/scene.json
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from smoothride.data.map_loader import attach_elevation, load_road_network
from smoothride.demo import scene as S
from smoothride.demo.export_cesium import _roads_3d, build_from_rollouts
from smoothride.demo.export_web import _lonlat_transformer, _to_lonlat
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool


def _random_rollout(env, key, steps):
    """Roll the env with uniform-random actions; collect the arrays the exporter wants."""
    st, _ = K.reset(env, key)
    pos, head, spd, crashed, goals, ped = [], [], [], [], [], []
    for t in range(steps):
        key, ka, ks = jax.random.split(key, 3)
        action = jax.random.uniform(ka, (env.n_agents, env.act_dim), minval=-1.0, maxval=1.0)
        st, _, _, _, info = K.step(env, st, action, ks)
        pos.append(st.pos); head.append(st.heading); spd.append(st.speed)
        crashed.append(info["just_crashed"]); goals.append(st.goals); ped.append(st.ped_pos)
    to = lambda xs: np.asarray(jnp.stack(xs))
    return {"pos": to(pos), "heading": to(head), "speed": to(spd),
            "crashed": to(crashed), "goals": to(goals), "ped": to(ped)}


def run(out: str, agents: int = 12, peds: int = 6, steps: int = 60, seed: int = 0) -> str:
    net = attach_elevation(load_road_network(), source="synthetic")
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=256, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=agents, n_peds=peds, max_steps=steps)
    tf = _lonlat_transformer(net)

    rollouts = {"trained": _random_rollout(env, jax.random.PRNGKey(seed), steps)}
    worlds = build_from_rollouts(net, env, tf, rollouts, stride=1)

    corners = np.array([[x0, y0], [x1, y1]], np.float32)
    clon, clat = _to_lonlat(net, tf, corners)
    meta = {"dt": float(env.dt), "n_steps": steps, "vmax": float(env.v_max),
            "center": [round(float(clon.mean()), 6), round(float(clat.mean()), 6)],
            "bounds": [[round(float(clon[0]), 6), round(float(clat[0]), 6)],
                       [round(float(clon[1]), 6), round(float(clat[1]), 6)]],
            "zoom": 15.5}
    scene = S.build_scene(meta=meta, roads=_roads_3d(net, tf),
                          buildings={"type": "FeatureCollection", "features": []},
                          worlds=worlds)
    nbytes = S.write_scene(out, scene)
    print(f"smoke ok: {out} ({nbytes/1024:.0f} KB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoothride/demo/cesium/public/scene.json")
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--peds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    run(args.out, agents=args.agents, peds=args.peds, steps=args.steps)


if __name__ == "__main__":
    main()
