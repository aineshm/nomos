"""Tests for the continuous ped-yield cost term in verifier.py (Task 3).

TDD: these tests are written BEFORE the implementation.
"""
import numpy as np
import pytest

from smoothride.rl.verifier import cost_signal, ped_yield_cost, step_cost


def _one(
    pos: list[float],
    speed: float,
    ped: list[float],
    crossing: bool,
    r_ped: float = 3.5,
    r_yield: float = 9.0,
    cap: float = 7.0,
) -> float:
    """Helper: (1,1,2),(1,1),(1,1,2),(1,1) -> scalar."""
    return float(
        ped_yield_cost(
            np.array([[pos]], np.float32),
            np.array([[speed]], np.float32),
            np.array([[ped]], np.float32),
            np.array([[crossing]], bool),
            r_ped,
            r_yield,
            cap,
        )[0, 0]
    )


def test_zero_when_far() -> None:
    assert _one([0, 0], 7.0, [100, 0], True) == 0.0


def test_zero_when_stopped_even_if_adjacent() -> None:
    assert _one([0, 0], 0.0, [4.0, 0], True) == 0.0


def test_zero_when_ped_not_crossing() -> None:
    assert _one([0, 0], 7.0, [4.0, 0], False) == 0.0


def test_ramps_with_proximity() -> None:
    near = _one([0, 0], 7.0, [4.0, 0], True)  # just outside hard radius
    far = _one([0, 0], 7.0, [8.0, 0], True)   # near the outer edge
    assert near > far > 0.0


def test_ramps_with_speed() -> None:
    fast = _one([0, 0], 7.0, [5.0, 0], True)
    slow = _one([0, 0], 2.0, [5.0, 0], True)
    assert fast > slow > 0.0


def test_bounded_and_graded_not_binary() -> None:
    c = _one([0, 0], 7.0, [3.5, 0], True)      # at hard radius, full speed
    assert 0.9 <= c <= 1.0
    mid = _one([0, 0], 7.0, [6.25, 0], True)   # midpoint of [3.5, 9]
    assert 0.2 < mid < 0.8                       # continuous, not 0/1


def test_multi_ped_max_aggregation() -> None:
    """Aggregation over crossing peds is max, not mean or sum.

    One car at origin, full speed (7 m/s). Two CROSSING peds:
      - near ped at (4, 0) — d ≈ 4 m
      - far ped at (8.5, 0) — d ≈ 8.5 m

    Expected cost == single-ped cost for the NEAR ped only.
    This test FAILS if .max is changed to .mean or .sum.
    """
    r_ped, r_yield, cap = 3.5, 9.0, 7.0
    speed = 7.0

    # Two-ped scenario: shape (T=1, N=1, M=2)
    pos = np.array([[[0.0, 0.0]]], np.float32)       # (1, 1, 2)
    spd = np.array([[speed]], np.float32)              # (1, 1)
    ped_pos = np.array([[[4.0, 0.0], [8.5, 0.0]]], np.float32)  # (1, 2, 2)
    ped_crossing = np.array([[True, True]], bool)      # (1, 2)

    result = float(ped_yield_cost(pos, spd, ped_pos, ped_crossing, r_ped, r_yield, cap)[0, 0])

    # Expected: near-ped single cost
    expected = _one([0.0, 0.0], speed, [4.0, 0.0], True, r_ped, r_yield, cap)

    assert abs(result - expected) < 1e-5, (
        f"Expected max-aggregation cost {expected:.6f}, got {result:.6f}. "
        "Aggregation must be max, not mean/sum."
    )
    # Sanity: far-ped-only cost is strictly less
    far_only = _one([0.0, 0.0], speed, [8.5, 0.0], True, r_ped, r_yield, cap)
    assert result > far_only, "max should dominate the far ped"


def test_empty_ped_array_returns_zeros() -> None:
    """Zero pedestrians → all-zeros output of shape (T, N)."""
    T, N = 3, 4
    pos = np.zeros((T, N, 2), np.float32)
    spd = np.ones((T, N), np.float32) * 5.0
    ped_pos = np.zeros((T, 0, 2), np.float32)         # M=0
    ped_crossing = np.zeros((T, 0), bool)

    result = ped_yield_cost(pos, spd, ped_pos, ped_crossing)

    assert result.shape == (T, N)
    assert np.all(result == 0.0)


def test_cost_signal_includes_ped_yield(make_trace):
    # a car at origin moving at cruise speed, a crossing ped 4 m away -> cost > lane terms
    # prox = (r_yield - dist) / (r_yield - r_ped) = (9 - 4) / (9 - 3.5) = 5/5.5 ≈ 0.909
    # speed_factor = speed / cruise_cap = 7.0 / 7.0 = 1.0 → ped_yield_cost ≈ 0.909
    trace = make_trace(n_steps=1, n_agents=1, n_peds=1,
                       pos=[[[0.0, 0.0]]], speed=[[7.0]],
                       ped_pos=[[[4.0, 0.0]]], ped_crossing=[[True]])
    c = cost_signal(trace)
    assert c.shape == (1, 1)
    assert 0.85 < c[0, 0] < 0.95   # ped-yield term: prox≈0.909, speed_factor=1.0


def test_step_cost_raises_when_ped_pos_without_ped_crossing() -> None:
    """step_cost must fail fast when ped_pos is given but ped_crossing is None."""
    T, N = 2, 1
    pos = np.zeros((T, N, 2), np.float32)
    seg_start = np.zeros((T, N, 2), np.float32)
    seg_end = np.ones((T, N, 2), np.float32)
    lane_count = np.ones((T, N), np.float32)
    lane_width = 3.5
    heading = np.zeros((T, N), np.float32)
    speed = np.ones((T, N), np.float32)
    spawn_grace = np.zeros((T, N), np.int32)
    crashed = np.zeros((T, N), np.float32)
    ped_pos = np.zeros((T, 1, 2), np.float32)

    with pytest.raises(ValueError, match="ped_crossing"):
        step_cost(
            pos, seg_start, seg_end, lane_count, lane_width,
            heading, speed, spawn_grace, crashed,
            ped_pos=ped_pos, ped_crossing=None,
        )


# ---------------------------------------------------------------------------
# Task 5: Integration test — collect logs peds and verifier_cost uses them
# ---------------------------------------------------------------------------

import jax  # noqa: E402  (import after test helpers to keep ordering explicit)

from smoothride.rl import ppo  # noqa: E402
from tests.env.test_kinematic_peds import _env  # noqa: E402


def test_collect_logs_peds_and_verifier_cost_runs() -> None:
    """collect() must log ped_pos (B,T,M,2) and ped_crossing (B,T,M) in the batch;
    verifier_cost() must accept those keys and return a non-negative (B,T,N) cost."""
    env = _env(cruise_cap=4.0)
    ts = ppo.make_train_state(env, ppo.PPOConfig(n_worlds=2), jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), 2)

    assert batch["ped_pos"].shape == (2, env.max_steps, env.n_peds, 2), (
        f"Expected ped_pos shape (2, {env.max_steps}, {env.n_peds}, 2), "
        f"got {batch['ped_pos'].shape}"
    )
    assert batch["ped_crossing"].shape == (2, env.max_steps, env.n_peds), (
        f"Expected ped_crossing shape (2, {env.max_steps}, {env.n_peds}), "
        f"got {batch['ped_crossing'].shape}"
    )

    import jax.numpy as jnp  # noqa: PLC0415

    cost = ppo.verifier_cost(env, batch)
    assert cost.shape == (2, env.max_steps, env.n_agents), (
        f"Expected cost shape (2, {env.max_steps}, {env.n_agents}), got {cost.shape}"
    )
    assert float(jnp.asarray(cost).max()) >= 0.0, "cost must be non-negative"
