"""Load the San Francisco drivable road network from OpenStreetMap (OSMnx 2.x).

The graph is the single source of truth for the whole project: the kinematic env
drives on it, routing/rewards use it, and the demo geometry is built from it.

We project node coordinates into a local metric frame (meters) so the kinematic
bicycle model and collision footprints can work in real-world units.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox

from . import elevation as _elev

# A small, dense downtown-SF box keeps iteration fast. Order is OSMnx 2.x:
# bbox = (west, south, east, north) = (left, bottom, right, top).
DOWNTOWN_SF_BBOX = (-122.4180, 37.7820, -122.4000, 37.7950)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data_cache")


@dataclass
class RoadNetwork:
    """A road network in a local metric (meter) frame, ready for simulation."""

    node_ids: np.ndarray            # (N,) original OSM node ids
    node_xy: np.ndarray             # (N, 2) projected coords, meters, origin-shifted
    edges: np.ndarray               # (E, 2) directed (u_idx, v_idx) into node_xy
    edge_length: np.ndarray         # (E,) meters
    edge_lanes: np.ndarray          # (E,) lane count (imputed where missing)
    edge_speed_kph: np.ndarray      # (E,) speed limit (imputed where missing)
    G: nx.MultiDiGraph              # the projected graph (for routing / Dijkstra)
    origin: tuple[float, float]     # (x0, y0) subtracted to keep coords small
    node_z: np.ndarray | None = None        # (N,) elevation (m), None until attached
    edge_grade: np.ndarray | None = None    # (E,) rise/run, None until attached

    @property
    def n_nodes(self) -> int:
        return len(self.node_xy)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def bounds(self) -> tuple[float, float, float, float]:
        x, y = self.node_xy[:, 0], self.node_xy[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.abspath(os.path.join(CACHE_DIR, name))


def load_sf_graph(
    bbox: tuple[float, float, float, float] = DOWNTOWN_SF_BBOX,
    cache_name: str = "sf_downtown_drive.graphml",
    refresh: bool = False,
) -> nx.MultiDiGraph:
    """Return a drivable SF graph, cached to graphml on first pull.

    Speeds and travel times are imputed (OSM lanes/maxspeed are often sparse).
    """
    path = _cache_path(cache_name)
    if os.path.exists(path) and not refresh:
        G = ox.load_graphml(path)
    else:
        # network_type="drive" => public drivable roads only.
        G = ox.graph_from_bbox(bbox, network_type="drive")
        G = ox.add_edge_speeds(G)        # -> edge attr 'speed_kph'
        G = ox.add_edge_travel_times(G)  # -> edge attr 'travel_time'
        ox.save_graphml(G, path)
    return G


def _impute_lanes(raw) -> int:
    """OSM 'lanes' is a string, a list, or missing. Collapse to a sane int."""
    if raw is None:
        return 1
    if isinstance(raw, list):
        vals = [int(v) for v in raw if str(v).isdigit()]
        return max(vals) if vals else 1
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return 1


def to_road_network(G: nx.MultiDiGraph) -> RoadNetwork:
    """Project to a metric CRS and flatten into simulation-ready arrays."""
    Gp = ox.project_graph(G)  # auto-picks a UTM zone -> coords in meters

    node_ids = np.array(list(Gp.nodes()))
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    node_xy = np.array([[Gp.nodes[n]["x"], Gp.nodes[n]["y"]] for n in node_ids], float)

    # Shift origin to the box min so coordinates stay small and positive.
    origin = (float(node_xy[:, 0].min()), float(node_xy[:, 1].min()))
    node_xy = node_xy - np.array(origin)

    edges, length, lanes, speed = [], [], [], []
    for u, v, data in Gp.edges(data=True):
        edges.append((idx_of[u], idx_of[v]))
        length.append(float(data.get("length", 0.0)))
        lanes.append(_impute_lanes(data.get("lanes")))
        speed.append(float(data.get("speed_kph", 30.0)))

    return RoadNetwork(
        node_ids=node_ids,
        node_xy=node_xy,
        edges=np.array(edges, dtype=np.int32),
        edge_length=np.array(length, float),
        edge_lanes=np.array(lanes, dtype=np.int32),
        edge_speed_kph=np.array(speed, float),
        G=Gp,
        origin=origin,
    )


def attach_elevation(net: RoadNetwork, source: str = "3dep") -> RoadNetwork:
    """Return a NEW RoadNetwork with node_z + edge_grade populated.

    source="synthetic" uses a deterministic offline hill field (tests / no net).
    source="3dep" samples real USGS 3DEP at each node (needs network + net.G crs).
    """
    if source == "synthetic":
        node_z = _elev.synthetic_elevation(net.node_xy)
    elif source == "3dep":
        from pyproj import Transformer
        tf = Transformer.from_crs(net.G.graph["crs"], "EPSG:4326", always_xy=True)
        east = net.node_xy[:, 0] + net.origin[0]
        north = net.node_xy[:, 1] + net.origin[1]
        lon, lat = tf.transform(east, north)
        node_z = _elev.fetch_node_elevation(np.column_stack([lon, lat]))
    else:
        raise ValueError(f"unknown elevation source: {source!r}")
    grade = _elev.edge_grades(node_z, net.edges, net.edge_length)
    return dataclasses.replace(net, node_z=node_z, edge_grade=grade)


def load_road_network(**kwargs) -> RoadNetwork:
    """Convenience: pull (or load cached) SF graph and return a RoadNetwork."""
    return to_road_network(load_sf_graph(**kwargs))


if __name__ == "__main__":
    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    print(f"nodes={net.n_nodes}  edges={net.n_edges}")
    print(f"extent: {x1 - x0:.0f} m x {y1 - y0:.0f} m")
    print(f"avg edge length: {net.edge_length.mean():.1f} m")
    print(f"lanes: min={net.edge_lanes.min()} max={net.edge_lanes.max()} "
          f"mean={net.edge_lanes.mean():.2f}")
    print(f"speed_kph: mean={net.edge_speed_kph.mean():.1f}")
