"""Tests for the continuous ped-yield cost term in verifier.py (Task 3).

TDD: these tests are written BEFORE the implementation.
"""
import numpy as np
import pytest

from smoothride.rl.verifier import ped_yield_cost, step_cost


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
