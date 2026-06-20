"""Export the SETPOINT stream that the frozen low-level WheeledLab controller
tracks in Isaac. Runs on any machine (no GPU) — it only replays the kinematic
rollout and saves the per-car target waypoint / velocity / heading at each step.

This is the clean handoff boundary: the coordination policy was trained on the
cheap kinematic env; everything downstream (rigid-body physics, RTX render) just
*tracks these setpoints* in Isaac, so there is no retraining and no body-transfer
problem.

Output: an .npz with
    pos      (T, N, 2)  target waypoint, meters in the local metric frame
    heading  (T, N)     target heading, radians (CCW from +x / east)
    speed    (T, N)     target speed, m/s
    crashed  (T, N)     bool, persistent
plus meta: dt, origin (UTM shift), crs (for georef), v_max, car asset name.

Usage:
  python -m smoothride.demo.isaac.export_setpoints --ckpt runs/trained.msgpack \
      --agents 24 --steps 300 --out runs/isaac/trained_setpoints.npz
"""
from __future__ import annotations

import argparse
import os

import jax
import numpy as np

from ...data.map_loader import load_road_network
from ...env import kinematic as K
from ...env.routing import build_route_pool
from ..render import load_params, rollout

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       "..", "..", "..", "runs", "isaac"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/trained.msgpack")
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--peds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--car-asset", default="wheeledlab/mushr",
                    help="WheeledLab car asset id the Isaac runner will spawn")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "setpoints.npz"))
    args = ap.parse_args()

    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024, seed=0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps)
    params = load_params(env, args.ckpt)
    tr = rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)

    crashed = np.cumsum(tr["crashed"].astype(np.int32), axis=0) > 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        pos=tr["pos"].astype(np.float32),
        heading=tr["heading"].astype(np.float32),
        speed=tr["speed"].astype(np.float32),
        crashed=crashed,
        dt=np.float32(env.dt),
        v_max=np.float32(env.v_max),
        origin=np.asarray(net.origin, np.float64),
        crs=str(net.G.graph["crs"]),
        car_asset=args.car_asset,
    )
    T, N, _ = tr["pos"].shape
    kb = os.path.getsize(args.out) / 1024
    print(f"setpoints: {N} cars x {T} steps @ dt={float(env.dt)}s  -> {args.out} ({kb:.0f} KB)")
    print(f"car asset: {args.car_asset}   (Isaac runner spawns N of these)")


if __name__ == "__main__":
    main()
