"""Host-side routing: precompute a pool of shortest-path routes, carrying the
per-waypoint road attributes the env needs (node id, junction flag, lane count,
speed limit). Graph search stays in networkx; the env just gathers by index.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..data.map_loader import RoadNetwork, _impute_lanes


@dataclass
class RoutePool:
    xy: np.ndarray      # (P, W, 2) waypoint coords, meters (tail-padded)
    n: np.ndarray       # (P,) valid waypoint count
    node: np.ndarray    # (P, W) node index into net.node_xy (pad = last)
    junc: np.ndarray    # (P, W) bool: waypoint node is a real intersection
    lanes: np.ndarray   # (P, W) int: lane count of the edge LEAVING waypoint w
    speed: np.ndarray   # (P, W) float: speed limit (m/s) of that edge

    @property
    def n_routes(self) -> int:
        return self.xy.shape[0]

    @property
    def max_wp(self) -> int:
        return self.xy.shape[1]


def build_route_pool(
    net: RoadNetwork,
    n_routes: int = 1024,
    max_waypoints: int = 32,
    min_waypoints: int = 4,
    max_length_m: float | None = 700.0,
    seed: int = 0,
) -> RoutePool:
    rng = np.random.default_rng(seed)
    nodes = list(net.G.nodes())
    idx_of = {nid: i for i, nid in enumerate(net.node_ids)}

    def street_count(nid):
        return net.G.nodes[nid].get("street_count", net.G.degree(nid))

    P, W = n_routes, max_waypoints
    xy = np.zeros((P, W, 2), np.float32)
    nn = np.zeros((P,), np.int32)
    node = np.zeros((P, W), np.int32)
    junc = np.zeros((P, W), bool)
    lanes = np.ones((P, W), np.int32)
    speed = np.full((P, W), 30.0 / 3.6, np.float32)

    filled, attempts = 0, 0
    while filled < P and attempts < P * 50:
        attempts += 1
        s, t = rng.choice(len(nodes), size=2, replace=False)
        u, v = nodes[s], nodes[t]
        try:
            path = nx.shortest_path(net.G, u, v, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) < min_waypoints:
            continue
        path = path[:max_waypoints]
        wp = np.array([net.node_xy[idx_of[p]] for p in path], np.float32)
        if max_length_m is not None and \
                np.linalg.norm(np.diff(wp, axis=0), axis=-1).sum() > max_length_m:
            continue

        L = len(path)
        xy[filled, :L] = wp
        xy[filled, L:] = wp[-1]
        nn[filled] = L
        node[filled, :L] = [idx_of[p] for p in path]
        node[filled, L:] = idx_of[path[-1]]
        junc[filled, :L] = [street_count(p) >= 3 for p in path]
        for w in range(L - 1):
            data = net.G[path[w]][path[w + 1]][0]
            lanes[filled, w] = _impute_lanes(data.get("lanes"))
            sp = data.get("speed_kph", 30.0)
            sp = float(np.mean(sp)) if isinstance(sp, list) else float(sp)
            speed[filled, w] = sp / 3.6  # km/h -> m/s
        lanes[filled, L - 1:] = lanes[filled, max(L - 2, 0)]
        speed[filled, L - 1:] = speed[filled, max(L - 2, 0)]
        filled += 1

    if filled == 0:
        raise RuntimeError("Could not build any routes — check the graph.")
    if filled < P:
        reps = (P + filled - 1) // filled
        sl = slice(0, filled)
        xy = np.tile(xy[sl], (reps, 1, 1))[:P]
        nn = np.tile(nn[sl], reps)[:P]
        node = np.tile(node[sl], (reps, 1))[:P]
        junc = np.tile(junc[sl], (reps, 1))[:P]
        lanes = np.tile(lanes[sl], (reps, 1))[:P]
        speed = np.tile(speed[sl], (reps, 1))[:P]

    return RoutePool(xy=xy, n=nn, node=node, junc=junc, lanes=lanes, speed=speed)


if __name__ == "__main__":
    from ..data.map_loader import load_road_network

    net = load_road_network()
    rp = build_route_pool(net, n_routes=256)
    print(f"routes={rp.n_routes}  max_wp={rp.max_wp}")
    print(f"waypoints/route: min={rp.n.min()} max={rp.n.max()} mean={rp.n.mean():.1f}")
    print(f"junction waypoints: {rp.junc.mean()*100:.0f}% of valid")
    print(f"lanes: min={rp.lanes.min()} max={rp.lanes.max()} mean={rp.lanes.mean():.2f}")
    print(f"speed m/s: min={rp.speed.min():.1f} max={rp.speed.max():.1f} "
          f"mean={rp.speed.mean():.1f}")
