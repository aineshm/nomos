"""Scale demo: render a trained (or untrained) policy on the HUGE SF map with
highways and hundreds of cars + pedestrians. Same map/speed as scale training.
"""
import argparse
import os

import jax

from smoothride.data.map_loader import load_sf_graph, to_road_network
from smoothride.demo.render import OUT, load_params, render, rollout
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool

# same huge eastern/central SF chunk used for scale training (incl. US-101/I-280)
HUGE_BBOX = (-122.4300, 37.7250, -122.3800, 37.8050)

ap = argparse.ArgumentParser()
ap.add_argument("--agents", type=int, default=300)
ap.add_argument("--peds", type=int, default=60)
ap.add_argument("--steps", type=int, default=350)
ap.add_argument("--vmax", type=float, default=30.0)
ap.add_argument("--ckpt", default=os.path.join(OUT, "trained_scale.msgpack"))
ap.add_argument("--name", default="scale_trained")
ap.add_argument("--title", default="SCALE — 300 cars on SF highways")
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

net = to_road_network(load_sf_graph(bbox=HUGE_BBOX, cache_name="sf_huge_drive.graphml"))
x0, y0, x1, y1 = net.bounds()
pool = build_route_pool(net, n_routes=2048, max_length_m=2500.0)
env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                 n_peds=args.peds, max_steps=args.steps, v_max=args.vmax)
params = load_params(env, args.ckpt)
tr = rollout(env, params, jax.random.PRNGKey(args.seed), sample=True)
out = os.path.join(OUT, "artifacts", args.name)
gif = render(net, tr["pos"], tr["crashed"], tr["goals"], tr["ped"], out, args.title)
print(f"cars={args.agents} crashed_end={int(tr['crashed'][-1].sum())} "
      f"trips_done={int(tr['goals'][-1].sum())}")
print(f"saved: {gif}")
