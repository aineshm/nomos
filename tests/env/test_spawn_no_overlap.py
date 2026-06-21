"""Spawns must not overlap: no two cars start within a collision of each other,
and no pedestrian starts on top of a car. (Buildings are not in the env — cars
spawn on the drivable road network by construction, so they never start inside a
footprint; see docs/internal/HANDOFF-sim-contract.md §0.)"""
import jax
import numpy as np

from smoothride.env import kinematic as K
from smoothride.env.routing import RoutePool


def _long_route_pool():
    # one long straight road, waypoints every 50 m -> many distinct spawn slots
    xs = np.arange(0.0, 1001.0, 50.0, dtype=np.float32)        # 21 waypoints
    W = xs.shape[0]
    xy = np.stack([xs, np.zeros_like(xs)], axis=-1)[None]      # (1, W, 2)
    return RoutePool(
        xy=xy,
        n=np.array([W], np.int32),
        node=np.arange(W, dtype=np.int32)[None],
        junc=np.zeros((1, W), bool),
        lanes=np.ones((1, W), np.int32),
        speed=np.full((1, W), 16.0, np.float32),
    )


def _env(n_agents, n_peds):
    return K.make_env(_long_route_pool(), world_min=[-20.0, -40.0],
                      world_max=[1020.0, 40.0], n_agents=n_agents, n_peds=n_peds,
                      max_steps=50)


def _pairwise_min(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min()


def test_no_two_cars_spawn_overlapping():
    env = _env(n_agents=10, n_peds=4)
    for seed in range(6):
        st, _ = K.reset(env, jax.random.PRNGKey(seed))
        assert _pairwise_min(np.array(st.pos)) > env.collision_radius


def test_cars_spawn_with_full_separation_when_feasible():
    # 10 cars over 20 distinct 50 m slots -> reject-sampling should reach full sep
    env = _env(n_agents=10, n_peds=4)
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    assert _pairwise_min(np.array(st.pos)) >= env.spawn_sep


def test_no_pedestrian_spawns_on_a_car():
    env = _env(n_agents=8, n_peds=6)
    for seed in range(6):
        st, _ = K.reset(env, jax.random.PRNGKey(seed))
        d = np.linalg.norm(np.array(st.ped_pos)[:, None, :] - np.array(st.pos)[None, :, :], axis=-1)
        assert d.min() > env.ped_radius
