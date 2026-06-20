"""Elevation for the SF road network.

Two sources, same output (per-node z in meters):
  * fetch_node_elevation(): real USGS 3DEP via py3dep (network).
  * synthetic_elevation():  a smooth deterministic hill field (offline / tests).

Edge grade = rise / run, the scalar the kinematic env and the demo both consume.
"""
from __future__ import annotations

import numpy as np


def synthetic_elevation(node_xy: np.ndarray, amp: float = 40.0,
                        wavelength: float = 600.0) -> np.ndarray:
    """Deterministic smooth elevation (m) over local metric coords.

    A sum of sinusoids gives SF-like rolling hills without any network call,
    so geometry tests and offline smoke runs are fully reproducible.
    """
    x = node_xy[:, 0] / wavelength
    y = node_xy[:, 1] / wavelength
    z = amp * (0.5 * np.sin(2 * np.pi * x) + 0.5 * np.cos(2 * np.pi * y))
    return (z - z.min()).astype(float)        # shift so min elevation is 0


def edge_grades(node_z: np.ndarray, edges: np.ndarray,
                edge_length: np.ndarray) -> np.ndarray:
    """Per-edge grade = (z_v - z_u) / horizontal_length. Safe on length 0."""
    z_u = node_z[edges[:, 0]]
    z_v = node_z[edges[:, 1]]
    run = np.where(edge_length > 1e-6, edge_length, 1e-6)
    return ((z_v - z_u) / run).astype(float)


def fetch_node_elevation(node_lonlat: np.ndarray) -> np.ndarray:
    """Real elevation (m) for (lon, lat) node coords via USGS 3DEP.

    NETWORK CALL. Kept isolated so unit tests never depend on it.
    """
    import py3dep
    coords = [(float(lon), float(lat)) for lon, lat in node_lonlat]
    elev = py3dep.elevation_bycoords(coords, crs="EPSG:4326")
    return np.asarray(elev, dtype=float)
