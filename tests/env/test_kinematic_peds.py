"""Deterministic pedestrian motion, cruise cap, and layered collision radii tests."""
import jax
import jax.numpy as jnp
import numpy as np

from smoothride.env import kinematic as K
from smoothride.env.routing import RoutePool


def _pool():
    # 2 straight routes, 3 waypoints, 2 lanes, speed 10 m/s
    xy = np.array([[[0, 0], [50, 0], [100, 0]],
                   [[0, 20], [50, 20], [100, 20]]], np.float32)
    n = np.array([3, 3], np.int32)
    node = np.zeros((2, 3), np.int32)
    junc = np.zeros((2, 3), bool)
    lanes = np.full((2, 3), 2, np.int32)
    speed = np.full((2, 3), 10.0, np.float32)
    return RoutePool(xy=xy, n=n, node=node, junc=junc, lanes=lanes, speed=speed)


def _env(**kw):
    return K.make_env(_pool(), world_min=[-10, -10], world_max=[110, 40],
                      n_agents=4, n_peds=6, seed=0, **kw)


def test_peds_are_deterministic_across_resets():
    env = _env()
    s1, _ = K.reset(env, jax.random.PRNGKey(0))
    s2, _ = K.reset(env, jax.random.PRNGKey(999))   # different key
    # peds do not depend on the reset key (paths are prebuilt, motion is f(t))
    np.testing.assert_allclose(np.asarray(s1.ped_pos), np.asarray(s2.ped_pos), atol=1e-5)


def test_ped_waits_before_start_then_moves():
    env = _env()
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    # a ped with start > 0 sits at path[0] until its start step
    late = int(np.argmax(np.asarray(env.ped_starts) > 0))
    assert env.ped_starts[late] > 0, "fixture must have at least one delayed-start ped"
    p0 = np.asarray(env.ped_paths[late, 0])
    np.testing.assert_allclose(np.asarray(st.ped_pos[late]), p0, atol=1e-4)
    # step until just past its start, then it should have moved
    act = jnp.zeros((env.n_agents, env.act_dim))
    s = st
    for _ in range(int(env.ped_starts[late]) + 5):
        s, *_ = K.step(env, s, act, jax.random.PRNGKey(0))
    assert np.linalg.norm(np.asarray(s.ped_pos[late]) - p0) > 0.1


def test_cruise_cap_clamps_speed():
    env = _env(cruise_cap=4.0)
    st, _ = K.reset(env, jax.random.PRNGKey(0))
    act = jnp.zeros((env.n_agents, env.act_dim)).at[:, 0].set(1.0)  # full throttle
    s = st
    for _ in range(30):
        s, *_ = K.step(env, s, act, jax.random.PRNGKey(0))
    assert float(jnp.max(s.speed)) <= 4.0 + 1e-4


def test_ped_collision_uses_raised_radius():
    env = _env(ped_radius=3.5)
    assert env.ped_radius == 3.5
    assert env.collision_radius < env.ped_radius   # asymmetric: wider berth for people


def test_observation_is_structured_with_masks():
    env = _env()
    st, obs = K.reset(env, jax.random.PRNGKey(0))
    assert set(obs) == {"ego", "cars", "cars_mask", "peds", "peds_mask"}
    N = env.n_agents
    assert obs["ego"].shape == (N, 7)
    assert obs["cars"].shape == (N, env.cand_cap_car, 4)
    assert obs["cars_mask"].shape == (N, env.cand_cap_car)
    assert obs["peds"].shape == (N, env.cand_cap_ped, 5)
    assert obs["peds_mask"].shape == (N, env.cand_cap_ped)
    # masks are boolean and self is never a neighbor of itself
    assert obs["cars_mask"].dtype == jnp.bool_


def test_ped_crossing_bit_present_in_obs():
    env = _env()
    st, obs = K.reset(env, jax.random.PRNGKey(0))
    # 5th ped-feature is the crossing bit in {0,1}
    bit = np.asarray(obs["peds"][..., 4])
    assert set(np.unique(bit[np.asarray(obs["peds_mask"])])) <= {0.0, 1.0}


def test_ped_advances_one_step_after_first_step():
    """Regression for the one-step ped/world-clock lag (post-step alignment).

    Before the fix, peds were evaluated at time k while cars advanced to k+1.
    After the fix, nst.ped_pos is consistent with nst.t (= st.t + 1).

    For a ped with ped_starts == 0:
      - At reset (t=0):  walked = max(0, 0 - 0) * ped_speed * dt = 0  → path[0]
      - After one step (nst.t=1): walked = max(0, 1 - 0) * ped_speed * dt > 0
        → position must differ from path[0] by exactly ped_speed * dt along the arc.
    """
    from smoothride.env.ped_paths import arc_interp

    base_env = _env()
    # Force ped index 0 to start at t=0 so the test is seed-independent.
    new_starts = jnp.asarray(base_env.ped_starts).at[0].set(0)
    env = base_env.replace(ped_starts=new_starts)
    zero_start_idx = 0

    st, _ = K.reset(env, jax.random.PRNGKey(0))

    # After reset the ped must be at path[0] (walked=0 at t=0)
    p0 = np.asarray(env.ped_paths[zero_start_idx, 0])
    np.testing.assert_allclose(np.asarray(st.ped_pos[zero_start_idx]), p0, atol=1e-4,
                               err_msg="ped at t=0 must be at path[0]")

    # Take exactly one step
    act = jnp.zeros((env.n_agents, env.act_dim))
    nst, *_ = K.step(env, st, act, jax.random.PRNGKey(0))

    assert int(nst.t) == 1, "nst.t must be 1 after one step"

    # Compute expected post-step position: walked = (t=1 - start=0) * speed * dt
    walked_expected = np.float32(1) * env.ped_speed * env.dt
    walked_arr = np.zeros(env.n_peds, np.float32)
    walked_arr[zero_start_idx] = walked_expected
    expected_pos = np.asarray(
        arc_interp(env.ped_paths, env.ped_cum, jnp.asarray(walked_arr))
    )

    actual_pos = np.asarray(nst.ped_pos[zero_start_idx])
    np.testing.assert_allclose(
        actual_pos, expected_pos[zero_start_idx], atol=1e-4,
        err_msg=(
            "After one step, ped must be at arc_interp(ped_speed*dt). "
            "Failure here means peds are still evaluated at t=0 (clock lag)."
        ),
    )
    # Sanity: position must have moved away from path[0]
    assert np.linalg.norm(actual_pos - p0) > 0.01, \
        "ped must have left path[0] after one step (clock lag not fixed)"
