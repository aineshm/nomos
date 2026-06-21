"""Tests for the continuous ped-yield cost term in verifier.py (Task 3).

TDD: these tests are written BEFORE the implementation.
"""
import numpy as np

from smoothride.rl.verifier import ped_yield_cost


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
