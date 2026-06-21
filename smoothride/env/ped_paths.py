"""Deterministic pedestrian paths: a sidewalk run + one perpendicular crossing.

Built once on the host (NumPy); the env interpolates position along the polyline
as a pure function of time (no per-step RNG), so peds are reproducible and
JAX/vmap-friendly. Each path has 4 points: start sidewalk -> walk -> cross to the
far sidewalk -> walk. The crossing leg (point 1 -> point 2) is the moment the ped
is in the roadway, which cars must negotiate.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PedPaths:
    paths: np.ndarray      # (M, 4, 2) polyline points, meters
    cum: np.ndarray        # (M, 4) cumulative arc length per point
    starts: np.ndarray     # (M,) int32 start step (staggered)
    cross_lo: np.ndarray   # (M,) f32 arc length where the crossing leg begins
    cross_hi: np.ndarray   # (M,) f32 arc length where the crossing leg ends


def build_ped_paths(
    routes_xy: np.ndarray,
    routes_n: np.ndarray,
    routes_lanes: np.ndarray,
    lane_width: float,
    n_peds: int,
    seed: int,
    *,
    sidewalk_offset: float = 1.5,
    run_len: float = 12.0,
    max_start: int = 60,
) -> PedPaths:
    """Build M pedestrian polylines deterministically from a fixed RNG seed.

    Args:
        routes_xy: (R, W, 2) route waypoints in meters.
        routes_n: (R,) number of valid waypoints per route.
        routes_lanes: (R, W) lane count at each waypoint.
        lane_width: width of one lane in meters.
        n_peds: number of pedestrians M to generate.
        seed: RNG seed; same seed → identical output.
        sidewalk_offset: meters from road edge to sidewalk centre.
        run_len: meters walked along the sidewalk before/after crossing.
        max_start: exclusive upper bound for randomised start step.

    Returns:
        PedPaths dataclass (immutable).
    """
    rng = np.random.default_rng(seed)
    R = routes_xy.shape[0]
    paths = np.zeros((n_peds, 4, 2), np.float32)
    for m in range(n_peds):
        r = int(rng.integers(0, R))
        nwp = max(int(routes_n[r]), 2)
        w = int(rng.integers(0, nwp - 1))           # segment [w, w+1]
        a, b = routes_xy[r, w], routes_xy[r, w + 1]
        u = b - a
        u = u / (np.linalg.norm(u) + 1e-6)          # along-segment unit
        nrm = np.array([u[1], -u[0]], np.float32)   # right-normal
        lanes = max(int(routes_lanes[r, w]), 1)
        half = lanes * lane_width / 2.0
        s = half + sidewalk_offset                  # sidewalk distance from centreline
        side = 1.0 if rng.random() < 0.5 else -1.0
        mid = (a + b) / 2.0
        p0 = mid + nrm * (s * side)                 # near sidewalk
        p1 = p0 + u * run_len                       # walk along sidewalk
        p2 = p1 - nrm * (2.0 * s * side)            # CROSS to far sidewalk (leg 1->2)
        p3 = p2 + u * run_len                       # walk along far sidewalk
        paths[m] = np.stack([p0, p1, p2, p3]).astype(np.float32)
    seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1)          # (M, 3)
    cum = np.concatenate(
        [np.zeros((n_peds, 1), np.float32), np.cumsum(seg, axis=1)],
        axis=1,
    ).astype(np.float32)
    starts = rng.integers(0, max_start, size=n_peds).astype(np.int32)
    cross_lo = cum[:, 1].copy()
    cross_hi = cum[:, 2].copy()
    return PedPaths(
        paths=paths,
        cum=cum,
        starts=starts,
        cross_lo=cross_lo,
        cross_hi=cross_hi,
    )


def arc_interp(
    paths: jnp.ndarray,
    cum: jnp.ndarray,
    walked: jnp.ndarray,
) -> jnp.ndarray:
    """Batched position along each polyline at arc length ``walked``. Clamped to ends.

    Args:
        paths: (M, 4, 2) polyline points.
        cum:   (M, 4) cumulative arc lengths (cum[:, 0] == 0).
        walked: (M,) arc lengths to interpolate at.

    Returns:
        (M, 2) positions; ``walked`` values outside [0, total] are clamped.
    """
    total = cum[:, -1]
    s = jnp.clip(walked, 0.0, total)
    # segment index: number of cumulative breakpoints strictly <= s after index 0,
    # clamped to [0, n_seg - 1] so we never index out of bounds.
    seg = jnp.clip(
        jnp.sum(cum[:, 1:] <= s[:, None], axis=1),
        0,
        paths.shape[1] - 2,
    )
    lo = jnp.take_along_axis(cum, seg[:, None], axis=1)[:, 0]       # (M,)
    hi = jnp.take_along_axis(cum, (seg + 1)[:, None], axis=1)[:, 0]
    frac = jnp.clip((s - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    a = jnp.take_along_axis(
        paths, seg[:, None, None].repeat(2, axis=2), axis=1
    )[:, 0, :]
    b = jnp.take_along_axis(
        paths, (seg + 1)[:, None, None].repeat(2, axis=2), axis=1
    )[:, 0, :]
    return a + frac[:, None] * (b - a)
