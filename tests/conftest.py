"""Shared fixtures. Pure/offline — no network, no GPU."""
import numpy as np
import pytest


@pytest.fixture
def square_nodes():
    """4 nodes on a 100 m square, 4 directed edges around it."""
    node_xy = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    edge_length = np.array([100.0, 100.0, 100.0, 100.0])
    return node_xy, edges, edge_length
