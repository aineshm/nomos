import numpy as np
from smoothride.data.map_loader import RoadNetwork, attach_elevation


def _net():
    return RoadNetwork(
        node_ids=np.array([1, 2, 3, 4]),
        node_xy=np.array([[0., 0.], [100., 0.], [100., 100.], [0., 100.]]),
        edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
        edge_length=np.array([100.0, 100.0]),
        edge_lanes=np.array([1, 1], dtype=np.int32),
        edge_speed_kph=np.array([30.0, 30.0]),
        G=None,
        origin=(0.0, 0.0),
        node_z=None,
        edge_grade=None,
    )


def test_attach_elevation_synthetic_populates_z_and_grade():
    net = attach_elevation(_net(), source="synthetic")
    assert net.node_z.shape == (4,)
    assert net.edge_grade.shape == (2,)
    assert np.all(np.isfinite(net.node_z))
    assert np.all(np.isfinite(net.edge_grade))


def test_attach_elevation_is_immutable():
    net = _net()
    out = attach_elevation(net, source="synthetic")
    assert net.node_z is None        # original untouched
    assert out is not net            # new object
