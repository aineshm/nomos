"""Deterministic pedestrian paths: a sidewalk run + one perpendicular crossing.

Built once on the host (NumPy); the env interpolates position along the polyline
as a pure function of time (no per-step RNG), so peds are reproducible and
JAX/vmap-friendly. Each path has 4 points: start sidewalk -> walk -> cross to the
far sidewalk -> walk. The crossing leg (point 1 -> point 2) is the moment the ped
is in the roadway, which cars must negotiate.

v2: Peds now cross at INTERSECTION nodes (junction waypoints) rather than
mid-block. If a route has no junction waypoints, falls back to legacy mid-block
crossing behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Number of polyline points per pedestrian path (start + 3 legs).
N_PED_POINTS: int = 4


@dataclass(frozen=True)
class PedPaths:
    paths: np.ndarray      # (M, 4, 2) polyline points, meters
    cum: np.ndarray        # (M, 4) cumulative arc length per point
    starts: np.ndarray     # (M,) int32 start step (staggered)
    cross_lo: np.ndarray   # (M,) f32 arc length where the crossing leg begins
    cross_hi: np.ndarray   # (M,) f32 arc length where the crossing leg ends


def _junction_waypoints(routes_junc: np.ndarray, r: int, nwp: int) -> list[int]:
    """Return interior waypoint indices that are junctions and have a valid segment."""
    result = []
    for j in range(nwp):
        if not routes_junc[r, j]:
            continue
        has_prev = j > 0
        has_next = j < nwp - 1
        if has_prev or has_next:
            result.append(j)
    return result


def _road_dir_at_junction(
    routes_xy: np.ndarray,
    r: int,
    j: int,
    nwp: int,
) -> np.ndarray:
    """Unit vector along the road at junction waypoint j (float32)."""
    if j < nwp - 1:
        a = routes_xy[r, j].astype(np.float32)
        b = routes_xy[r, j + 1].astype(np.float32)
    else:
        a = routes_xy[r, j - 1].astype(np.float32)
        b = routes_xy[r, j].astype(np.float32)
    delta = b - a
    return delta / (np.linalg.norm(delta).astype(np.float32) + np.float32(1e-6))


def _crossing_at_junction(
    routes_xy: np.ndarray,
    routes_junc: np.ndarray,
    routes_lanes: np.ndarray,
    r: int,
    nwp: int,
    rng: np.random.Generator,
    lane_width: float,
    sidewalk_offset: float,
    run_len: float,
) -> np.ndarray:
    """Build a 4-point crossing polyline anchored at a junction node.

    Returns shape (4, 2) float32, or None if no junction waypoints exist.
    """
    junc_wps = _junction_waypoints(routes_junc, r, nwp)
    if not junc_wps:
        return None  # caller will fall back to mid-block

    j = int(rng.choice(junc_wps))
    node = routes_xy[r, j].astype(np.float32)
    u = _road_dir_at_junction(routes_xy, r, j, nwp)
    nrm = np.array([u[1], -u[0]], np.float32)  # right-normal (perpendicular)

    lanes = max(int(routes_lanes[r, j]), 1)
    half = np.float32(lanes * lane_width / 2.0)
    s = half + np.float32(sidewalk_offset)
    side = np.float32(1.0 if rng.random() < 0.5 else -1.0)

    # Geometry: approach junction along sidewalk, cross road AT the junction node,
    # then continue on far sidewalk.  Points 1->2 are the road-crossing leg.
    p1 = node + nrm * (s * side)                       # near sidewalk AT junction
    p0 = p1 - u * np.float32(run_len)                  # approach: walk BACK along sidewalk
    p2 = node - nrm * (s * side)                       # far sidewalk AT junction
    p3 = p2 + u * np.float32(run_len)                  # walk on far sidewalk

    return np.stack([p0, p1, p2, p3])


def _crossing_midblock(
    routes_xy: np.ndarray,
    routes_lanes: np.ndarray,
    r: int,
    nwp: int,
    rng: np.random.Generator,
    lane_width: float,
    sidewalk_offset: float,
    run_len: float,
) -> np.ndarray:
    """Legacy mid-block crossing: random segment midpoint.  Returns (4, 2) float32."""
    w = int(rng.integers(0, nwp - 1))       # segment [w, w+1]
    a = routes_xy[r, w].astype(np.float32)
    b = routes_xy[r, w + 1].astype(np.float32)
    u = b - a
    u = u / (np.linalg.norm(u).astype(np.float32) + np.float32(1e-6))
    nrm = np.array([u[1], -u[0]], np.float32)
    lanes = max(int(routes_lanes[r, w]), 1)
    half = np.float32(lanes * lane_width / 2.0)
    s = half + np.float32(sidewalk_offset)
    side = np.float32(1.0 if rng.random() < 0.5 else -1.0)
    mid = (a + b) * np.float32(0.5)
    p0 = mid + nrm * (s * side)
    p1 = p0 + u * np.float32(run_len)
    p2 = p1 - nrm * (np.float32(2.0) * s * side)
    p3 = p2 + u * np.float32(run_len)
    return np.stack([p0, p1, p2, p3])


def build_ped_paths(
    routes_xy: np.ndarray,
    routes_n: np.ndarray,
    routes_lanes: np.ndarray,
    routes_junc: np.ndarray,
    lane_width: float,
    n_peds: int,
    seed: int,
    *,
    sidewalk_offset: float = 1.5,
    run_len: float = 12.0,
    max_start: int = 60,
) -> PedPaths:
    """Build M pedestrian polylines deterministically from a fixed RNG seed.

    Peds cross at junction (intersection) nodes when available; otherwise
    falls back to random segment midpoint (legacy mid-block behaviour).

    Args:
        routes_xy: (R, W, 2) route waypoints in meters.
        routes_n: (R,) number of valid waypoints per route.
        routes_lanes: (R, W) lane count at each waypoint.
        routes_junc: (R, W) bool, True where a waypoint is a junction node.
        lane_width: width of one lane in meters.
        n_peds: number of pedestrians M to generate.
        seed: RNG seed; same seed -> identical output.
        sidewalk_offset: meters from road edge to sidewalk centre.
        run_len: meters walked along the sidewalk before/after crossing.
        max_start: exclusive upper bound for randomised start step.

    Returns:
        PedPaths dataclass (immutable).
    """
    rng = np.random.default_rng(seed)
    R = routes_xy.shape[0]
    paths = np.zeros((n_peds, N_PED_POINTS, 2), np.float32)
    for m in range(n_peds):
        r = int(rng.integers(0, R))
        nwp = max(int(routes_n[r]), 2)
        polyline = _crossing_at_junction(
            routes_xy, routes_junc, routes_lanes,
            r, nwp, rng, lane_width, sidewalk_offset, run_len,
        )
        if polyline is None:
            polyline = _crossing_midblock(
                routes_xy, routes_lanes,
                r, nwp, rng, lane_width, sidewalk_offset, run_len,
            )
        paths[m] = polyline

    seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1).astype(np.float32)
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
        paths: (M, N_PED_POINTS, 2) polyline points.
        cum:   (M, N_PED_POINTS) cumulative arc lengths (cum[:, 0] == 0).
        walked: (M,) arc lengths to interpolate at.

    Returns:
        (M, 2) positions; ``walked`` values outside [0, total] are clamped.
    """
    total = cum[:, -1]
    s = jnp.clip(walked, 0.0, total)
    # Count breakpoints (columns 1..) where cum < s (strict less-than).
    # Using strict `<` means a zero-length segment (cum[k] == cum[k+1]) is never
    # counted unless s has genuinely advanced past it, so coincident waypoints
    # cannot push the index onto a later segment when walked==0.
    seg = jnp.clip(
        jnp.sum(cum[:, 1:] < s[:, None], axis=1),
        0,
        N_PED_POINTS - 2,
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
