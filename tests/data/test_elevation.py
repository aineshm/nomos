import numpy as np
from smoothride.data import elevation as E


def test_synthetic_elevation_is_deterministic_and_smooth():
    xy = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    z1 = E.synthetic_elevation(xy)
    z2 = E.synthetic_elevation(xy)
    assert z1.shape == (3,)
    np.testing.assert_array_equal(z1, z2)          # deterministic
    assert np.all(np.isfinite(z1))


def test_edge_grades_flat_is_zero():
    node_z = np.array([10.0, 10.0, 10.0])
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    length = np.array([100.0, 50.0])
    g = E.edge_grades(node_z, edges, length)
    np.testing.assert_allclose(g, [0.0, 0.0])


def test_edge_grades_rise_over_run():
    node_z = np.array([0.0, 15.0])     # +15 m over 100 m run
    edges = np.array([[0, 1]], dtype=np.int32)
    length = np.array([100.0])
    g = E.edge_grades(node_z, edges, length)
    np.testing.assert_allclose(g, [0.15])


def test_edge_grades_zero_length_is_safe():
    node_z = np.array([0.0, 5.0])
    edges = np.array([[0, 1]], dtype=np.int32)
    length = np.array([0.0])
    g = E.edge_grades(node_z, edges, length)
    assert np.all(np.isfinite(g))      # no divide-by-zero
