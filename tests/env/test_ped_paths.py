import jax.numpy as jnp
import numpy as np
import pytest

from smoothride.env.ped_paths import PedPaths, arc_interp, build_ped_paths


@pytest.fixture
def simple_net():
    # one straight 2-lane route: 3 waypoints along +x at y=0
    routes_xy = np.array([[[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]], np.float32)
    routes_n = np.array([3], np.int32)
    routes_lanes = np.array([[2, 2, 2]], np.int32)
    routes_junc = np.array([[False, False, False]], bool)
    return routes_xy, routes_n, routes_lanes, routes_junc


@pytest.fixture
def junction_net():
    """Route with a junction waypoint at index 1 (the middle point)."""
    routes_xy = np.array([[[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]], np.float32)
    routes_n = np.array([3], np.int32)
    routes_lanes = np.array([[2, 2, 2]], np.int32)
    # waypoint 1 is a junction
    routes_junc = np.array([[False, True, False]], bool)
    return routes_xy, routes_n, routes_lanes, routes_junc


@pytest.fixture
def no_junction_net():
    """Route with NO junction waypoints — fallback to mid-block crossing."""
    routes_xy = np.array([[[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]], np.float32)
    routes_n = np.array([3], np.int32)
    routes_lanes = np.array([[2, 2, 2]], np.int32)
    routes_junc = np.array([[False, False, False]], bool)
    return routes_xy, routes_n, routes_lanes, routes_junc


def test_build_shapes_and_determinism(simple_net):
    xy, n, lanes, junc = simple_net
    a = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=5, seed=0)
    b = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=5, seed=0)
    assert isinstance(a, PedPaths)
    assert a.paths.shape == (5, 4, 2)
    assert a.cum.shape == (5, 4)
    assert a.starts.shape == (5,)
    assert a.cross_lo.shape == (5,) and a.cross_hi.shape == (5,)
    # deterministic for a fixed seed
    np.testing.assert_array_equal(a.paths, b.paths)
    np.testing.assert_array_equal(a.starts, b.starts)
    # cumulative arc length is monotonic non-decreasing, starts at 0
    assert np.all(a.cum[:, 0] == 0.0)
    assert np.all(np.diff(a.cum, axis=1) >= -1e-4)
    # crossing leg is a real interval inside the path
    assert np.all(a.cross_hi > a.cross_lo)
    assert np.all(a.cross_hi <= a.cum[:, -1] + 1e-4)


def test_starts_staggered_and_bounded(simple_net):
    xy, n, lanes, junc = simple_net
    p = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=50, seed=1, max_start=60)
    assert p.starts.min() >= 0 and p.starts.max() < 60
    assert len(np.unique(p.starts)) > 1  # actually staggered


def test_arc_interp_endpoints_and_midpoint(simple_net):
    xy, n, lanes, junc = simple_net
    p = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=3, seed=2)
    paths, cum = jnp.asarray(p.paths), jnp.asarray(p.cum)
    # walked=0 -> path start; walked>=total -> path end (clamped)
    at_start = arc_interp(paths, cum, jnp.zeros(3))
    at_end = arc_interp(paths, cum, cum[:, -1] + 100.0)
    np.testing.assert_allclose(np.asarray(at_start), p.paths[:, 0, :], atol=1e-4)
    np.testing.assert_allclose(np.asarray(at_end), p.paths[:, -1, :], atol=1e-4)
    # halfway along total arc length: assert geometrically correct interpolation.
    half_walked = cum[:, -1] * 0.5
    mid = arc_interp(paths, cum, half_walked)
    mid_np = np.asarray(mid)
    assert np.all(np.isfinite(mid_np))

    # Compute expected midpoint analytically for each ped.
    # Find segment index and fractional position from the numpy arrays directly.
    p_np = np.asarray(paths)
    c_np = np.asarray(cum)
    hw_np = np.asarray(half_walked)
    expected = np.zeros((p_np.shape[0], 2), np.float32)
    for i in range(p_np.shape[0]):
        s_val = hw_np[i]
        # segment: last index where cum < s (strict), clamped to [0, n_seg-1]
        seg_i = int(np.clip(np.sum(c_np[i, 1:] < s_val), 0, p_np.shape[1] - 2))
        lo_i, hi_i = c_np[i, seg_i], c_np[i, seg_i + 1]
        frac_i = float(np.clip((s_val - lo_i) / (hi_i - lo_i + 1e-6), 0.0, 1.0))
        expected[i] = p_np[i, seg_i] + frac_i * (p_np[i, seg_i + 1] - p_np[i, seg_i])

    np.testing.assert_allclose(mid_np, expected, atol=1e-4,
                               err_msg="arc_interp midpoint does not match expected interpolated coordinate")


def test_junction_crossing_anchored_at_junction_node(junction_net):
    """Peds must cross at the junction node (waypoint 1 at x=50).

    The crossing-leg midpoint (paths[m,1]+paths[m,2])/2 should have x ≈ 50
    for all peds, since the only route has its junction at waypoint 1 (x=50,y=0).
    """
    xy, n, lanes, junc = junction_net
    p = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=10, seed=42)
    # The crossing leg goes from p1 (near sidewalk, at junction) to p2 (far sidewalk).
    # The midpoint of that leg should be at the junction node's x coord (50).
    crossing_mid = (p.paths[:, 1, :] + p.paths[:, 2, :]) / 2.0
    # The junction is at (50, 0). The road is along x-axis, so crossing is perpendicular:
    # crossing mid should have x ≈ 50 (along-road position) and y near 0 (at the node).
    np.testing.assert_allclose(crossing_mid[:, 0], 50.0, atol=1.0,
                               err_msg="Crossing midpoint x should be at junction node x=50")
    np.testing.assert_allclose(crossing_mid[:, 1], 0.0, atol=1.0,
                               err_msg="Crossing midpoint y should be at junction node y=0")


def test_fallback_no_junctions_produces_valid_paths(no_junction_net):
    """When a route has NO junction waypoints, fall back to mid-block crossing.

    This must not crash and must produce paths with correct shape and valid intervals.
    """
    xy, n, lanes, junc = no_junction_net
    p = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=8, seed=7)
    assert p.paths.shape == (8, 4, 2)
    assert p.cum.shape == (8, 4)
    assert p.starts.shape == (8,)
    assert p.cross_lo.shape == (8,) and p.cross_hi.shape == (8,)
    assert np.all(p.cross_hi > p.cross_lo)
    assert np.all(p.cum[:, 0] == 0.0)
    assert np.all(np.diff(p.cum, axis=1) >= -1e-4)


def test_determinism_with_junction(junction_net):
    """Same seed → identical paths, with junction routing active."""
    xy, n, lanes, junc = junction_net
    a = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=8, seed=99)
    b = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=8, seed=99)
    np.testing.assert_array_equal(a.paths, b.paths)
    np.testing.assert_array_equal(a.starts, b.starts)


def test_cross_lo_hi_are_cum_columns(simple_net):
    """cross_lo == cum[:,1], cross_hi == cum[:,2] (contract)."""
    xy, n, lanes, junc = simple_net
    p = build_ped_paths(xy, n, lanes, junc, lane_width=3.5, n_peds=6, seed=3)
    np.testing.assert_array_equal(p.cross_lo, p.cum[:, 1])
    np.testing.assert_array_equal(p.cross_hi, p.cum[:, 2])
