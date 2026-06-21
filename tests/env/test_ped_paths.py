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
    return routes_xy, routes_n, routes_lanes


def test_build_shapes_and_determinism(simple_net):
    xy, n, lanes = simple_net
    a = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=5, seed=0)
    b = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=5, seed=0)
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
    xy, n, lanes = simple_net
    p = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=50, seed=1, max_start=60)
    assert p.starts.min() >= 0 and p.starts.max() < 60
    assert len(np.unique(p.starts)) > 1  # actually staggered


def test_arc_interp_endpoints_and_midpoint(simple_net):
    xy, n, lanes = simple_net
    p = build_ped_paths(xy, n, lanes, lane_width=3.5, n_peds=3, seed=2)
    paths, cum = jnp.asarray(p.paths), jnp.asarray(p.cum)
    # walked=0 -> path start; walked>=total -> path end (clamped)
    at_start = arc_interp(paths, cum, jnp.zeros(3))
    at_end = arc_interp(paths, cum, cum[:, -1] + 100.0)
    np.testing.assert_allclose(np.asarray(at_start), p.paths[:, 0, :], atol=1e-4)
    np.testing.assert_allclose(np.asarray(at_end), p.paths[:, -1, :], atol=1e-4)
    # halfway along total arc length lies on the polyline (finite, within bbox)
    mid = arc_interp(paths, cum, cum[:, -1] * 0.5)
    assert np.all(np.isfinite(np.asarray(mid)))
