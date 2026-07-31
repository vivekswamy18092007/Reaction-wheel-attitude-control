"""Hypervolume indicator and plateau detection.

These are the tests that matter for trusting the stopping rule: HV has to be
correct on cases you can compute by hand, and it has to be monotone with
Pareto dominance -- that monotonicity is the entire reason it replaced
per-objective bests as the convergence signal.
"""

import numpy as np
import pytest

from deimos.tuning.hypervolume import (
    StallDetector, hypervolume, hypervolume_fraction,
)


# --- correctness on hand-computable cases ---------------------------------

def test_single_point_2d_is_a_rectangle():
    # minimization: the point (1,2) against reference (4,5) dominates a
    # 3 x 3 box.
    assert hypervolume([[1.0, 2.0]], [4.0, 5.0]) == pytest.approx(9.0)


def test_single_point_3d_is_a_box():
    assert hypervolume([[1.0, 1.0, 1.0]], [3.0, 4.0, 5.0]) == pytest.approx(2 * 3 * 4)


def test_two_disjoint_points_2d():
    # (1,3) covers 3x2=6, (3,1) covers 1x4=4, overlap is the box dominated by
    # (3,3) = 1x2 = 2. Inclusion-exclusion: 6 + 4 - 2 = 8.
    assert hypervolume([[1.0, 3.0], [3.0, 1.0]], [4.0, 5.0]) == pytest.approx(8.0)


def test_dominated_point_adds_nothing():
    a = hypervolume([[1.0, 1.0]], [4.0, 4.0])
    b = hypervolume([[1.0, 1.0], [2.0, 2.0]], [4.0, 4.0])
    assert a == pytest.approx(b)


def test_point_worse_than_reference_contributes_nothing():
    assert hypervolume([[5.0, 5.0]], [4.0, 4.0]) == pytest.approx(0.0)
    # ...and must not subtract from a genuine contribution either
    assert (hypervolume([[1.0, 1.0], [5.0, 5.0]], [4.0, 4.0])
            == pytest.approx(hypervolume([[1.0, 1.0]], [4.0, 4.0])))


def test_non_finite_rows_are_dropped_not_propagated():
    ref = [4.0, 4.0]
    with_nan = [[1.0, 1.0], [np.nan, 2.0], [2.0, np.inf]]
    assert hypervolume(with_nan, ref) == pytest.approx(hypervolume([[1.0, 1.0]], ref))


def test_empty_front_is_zero():
    assert hypervolume(np.empty((0, 3)), [1.0, 1.0, 1.0]) == 0.0


# --- the property the stopping rule depends on ----------------------------

def test_hypervolume_is_monotone_with_dominance():
    ref = [10.0, 10.0, 10.0]
    worse = np.array([[5.0, 5.0, 5.0], [6.0, 4.0, 5.0]])
    better = worse - 1.0          # strictly dominates every point
    assert hypervolume(better, ref) > hypervolume(worse, ref)


def test_adding_a_non_dominated_point_increases_hypervolume():
    # This is what per-objective bests CANNOT see: the endpoints are
    # unchanged, only the interior of the front gained a point.
    ref = [10.0, 10.0]
    ends = [[1.0, 9.0], [9.0, 1.0]]
    filled = ends + [[4.0, 4.0]]
    best_before = np.min(np.array(ends), axis=0)
    best_after = np.min(np.array(filled), axis=0)
    assert np.allclose(best_before, best_after)      # endpoints identical
    assert hypervolume(filled, ref) > hypervolume(ends, ref)


def test_fraction_is_between_zero_and_one():
    ref = [10.0, 10.0, 10.0]
    pts = np.random.default_rng(0).random((25, 3)) * 9.0
    frac = hypervolume_fraction(pts, ref)
    assert 0.0 <= frac <= 1.0
    assert frac == pytest.approx(hypervolume(pts, ref) / 1000.0)


def test_fraction_of_the_utopia_point_is_one():
    assert hypervolume_fraction([[0.0, 0.0, 0.0]], [2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_mismatched_reference_dimension_raises():
    with pytest.raises(ValueError):
        hypervolume([[1.0, 2.0, 3.0]], [4.0, 5.0])


# --- plateau detection ------------------------------------------------------

def test_stall_detector_waits_for_min_generations():
    d = StallDetector(patience=3, tol=1e-3, min_generations=10)
    for _ in range(8):
        d.update(0.5)              # perfectly flat, but too early to act
    assert not d.stalled()


def test_stall_detector_fires_on_a_plateau():
    d = StallDetector(patience=3, tol=1e-3, min_generations=5)
    for _ in range(12):
        d.update(0.5)
    assert d.stalled()
    assert "plateau" in d.reason()


def test_stall_detector_does_not_fire_while_still_improving():
    d = StallDetector(patience=3, tol=1e-3, min_generations=5)
    for i in range(15):
        d.update(0.1 * (1.0 + i))   # 10% relative gain per generation
    assert not d.stalled()


def test_stall_detector_tolerates_nan_hypervolume():
    # hv_reference=None means HV is NaN; the detector must simply never fire
    # rather than crash the run.
    d = StallDetector(patience=2, tol=1e-3, min_generations=3)
    for _ in range(10):
        d.update(float("nan"))
    assert not d.stalled()


def test_relative_gain_matches_the_definition():
    d = StallDetector(patience=2, tol=1e-9, min_generations=1)
    for v in (1.0, 1.5, 2.0):
        d.update(v)
    assert d.relative_gain == pytest.approx((2.0 - 1.0) / 1.0)
