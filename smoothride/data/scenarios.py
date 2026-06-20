"""Mine specific road-topology EDGE CASES from the SF OpenStreetMap graph, so a
scenario-curriculum trainer can train a policy on each one.

Each scenario is a junction (or bridge) location + a local window the env can be
built around. Control type (signalized / stop / uncontrolled-yield) comes from OSM
node tags; topology (3-way / 4-way) from street_count; highways/ramps/bridges from
edge tags.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

from .map_loader import CACHE_DIR, load_sf_graph, to_road_network

BIG_BBOX = (-122.4300, 37.7250, -122.3800, 37.8050)


@dataclass
class Scenario:
    kind: str            # four_way / three_way / ramp_merge / bridge / uturn ...
    control: str         # signalized / stop / uncontrolled
    node_id: int         # original OSM node id (0 for bridge midpoints)
    x: float             # projected meters (local frame)
    y: float
    street_count: int


def _node_control(d) -> str:
    h = d.get("highway")
    if h == "traffic_signals":
        return "signalized"
    if h == "stop":
        return "stop"
    return "uncontrolled"  # SF "yield-only" / priority junctions


def mine(G, net) -> dict[str, list[Scenario]]:
    idx_of = {nid: i for i, nid in enumerate(net.node_ids)}

    def xy(nid):
        i = idx_of[nid]
        return float(net.node_xy[i, 0]), float(net.node_xy[i, 1])

    # nodes incident to a highway ramp / motorway (merge scenarios)
    ramp_nodes, motorway_nodes, bridge_nodes = set(), set(), set()
    for u, v, d in G.edges(data=True):
        h = d.get("highway")
        h = h if isinstance(h, str) else (h[0] if isinstance(h, list) else h)
        if h in ("motorway_link", "trunk_link", "primary_link"):
            ramp_nodes.update((u, v))
        if h in ("motorway", "trunk"):
            motorway_nodes.update((u, v))
        if d.get("bridge") not in (None, "no"):
            bridge_nodes.update((u, v))

    out: dict[str, list[Scenario]] = {
        "four_way": [], "three_way": [], "ramp_merge": [],
        "highway_uturn": [], "bridge": [], "complex": [],
    }
    for nid, d in G.nodes(data=True):
        if nid not in idx_of:
            continue
        sc = d.get("street_count", G.degree(nid))
        try:
            sc = int(sc)
        except (TypeError, ValueError):
            sc = G.degree(nid)
        ctrl = _node_control(d)
        x, y = xy(nid)
        s = Scenario("", ctrl, int(nid), x, y, sc)
        if d.get("highway") == "turning_circle" or (
                nid in motorway_nodes and sc >= 3):
            out["highway_uturn"].append(Scenario("highway_uturn", ctrl, int(nid), x, y, sc))
        if nid in ramp_nodes:
            out["ramp_merge"].append(Scenario("ramp_merge", ctrl, int(nid), x, y, sc))
        if nid in bridge_nodes:
            out["bridge"].append(Scenario("bridge", ctrl, int(nid), x, y, sc))
        if sc == 4:
            out["four_way"].append(Scenario("four_way", ctrl, int(nid), x, y, sc))
        elif sc == 3:
            out["three_way"].append(Scenario("three_way", ctrl, int(nid), x, y, sc))
        elif sc >= 5:
            out["complex"].append(Scenario("complex", ctrl, int(nid), x, y, sc))
    return out


def window_net(net, cx: float, cy: float, half: float = 160.0):
    """Induce a fixed-size square sub-RoadNetwork around (cx,cy) in box-local
    coords. Fixed size => every scenario window shares grid dims, so windows can
    be stacked and trained in parallel (vmap)."""
    from .map_loader import RoadNetwork
    net_idx = {nid: i for i, nid in enumerate(net.node_ids)}
    sel = [net.node_ids[i] for i in range(len(net.node_ids))
           if abs(net.node_xy[i, 0] - cx) < half and abs(net.node_xy[i, 1] - cy) < half]
    H = net.G.subgraph(sel).copy()
    if H.number_of_nodes() == 0:
        return None
    comp = max(nx.weakly_connected_components(H), key=len)
    H = H.subgraph(comp).copy()
    if H.number_of_nodes() < 4:
        return None
    ids = np.array(list(H.nodes()))
    local = {n: i for i, n in enumerate(ids)}
    origin = (cx - half, cy - half)
    xy = (net.node_xy[[net_idx[n] for n in ids]] - np.array(origin)).astype(np.float32)
    edges, length, lanes, speed = [], [], [], []
    from .map_loader import _impute_lanes
    for u, v, d in H.edges(data=True):
        edges.append((local[u], local[v]))
        length.append(float(d.get("length", 0.0)))
        lanes.append(_impute_lanes(d.get("lanes")))
        sp = d.get("speed_kph", 40.0)
        speed.append(float(np.mean(sp)) if isinstance(sp, list) else float(sp))
    return RoadNetwork(
        node_ids=ids, node_xy=xy, edges=np.array(edges, np.int32),
        edge_length=np.array(length, float), edge_lanes=np.array(lanes, np.int32),
        edge_speed_kph=np.array(speed, float), G=H, origin=origin), (2 * half, 2 * half)


def catalog(scn: dict[str, list[Scenario]]) -> dict:
    cat = {}
    for kind, lst in scn.items():
        by_ctrl = {}
        for s in lst:
            by_ctrl[s.control] = by_ctrl.get(s.control, 0) + 1
        cat[kind] = {"total": len(lst), "by_control": by_ctrl}
    return cat


def pick_representatives(scn, per_kind=8, seed=0):
    """A spread of representative locations per scenario for training."""
    rng = np.random.default_rng(seed)
    reps = {}
    for kind, lst in scn.items():
        if not lst:
            continue
        # prefer uncontrolled (hardest: no signal telling cars what to do)
        unc = [s for s in lst if s.control == "uncontrolled"] or lst
        idx = rng.choice(len(unc), size=min(per_kind, len(unc)), replace=False)
        reps[kind] = [asdict(unc[i]) for i in idx]
    return reps


if __name__ == "__main__":
    G = load_sf_graph(bbox=BIG_BBOX, cache_name="sf_huge_drive.graphml")
    net = to_road_network(G)
    scn = mine(G, net)
    cat = catalog(scn)
    print("=== SF edge-case scenario catalog ===")
    for kind, c in cat.items():
        print(f"{kind:16s} total={c['total']:5d}  by_control={c['by_control']}")
    reps = pick_representatives(scn)
    path = os.path.abspath(os.path.join(CACHE_DIR, "scenarios.json"))
    with open(path, "w") as f:
        json.dump({"catalog": cat, "representatives": reps}, f, indent=2)
    print(f"\nsaved catalog + representatives -> {path}")
