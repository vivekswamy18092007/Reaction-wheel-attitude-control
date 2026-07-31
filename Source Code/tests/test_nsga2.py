"""
Covers NSGA2's eval_timeout -- added after a real overnight run had one
candidate's simulate() call take ~1800s against a ~3s baseline (no
exception, just very slow) and block an entire generation behind it, since
ProcessPoolExecutor.map() yields results in submission order. See the
NSGA2 docstring in tuning/nsga2.py for the full incident.

_SleepEvaluator is a module-level (picklable) test double standing in for
objectives.Evaluator: it exposes the same `.worst` contract eval_timeout's
fallback relies on, and sleeps on command instead of running a real
simulation, so these tests exercise the real ProcessPoolExecutor path in
milliseconds rather than minutes.
"""

import time

import numpy as np
import pytest

from deimos.tuning.nsga2 import NSGA2


class _SleepEvaluator:
    worst = (999.0, 999.0)

    def __init__(self, slow_seconds=5.0, threshold=0.5):
        self.slow_seconds = slow_seconds
        self.threshold = threshold

    def __call__(self, decoded):
        if decoded[0] > self.threshold:
            time.sleep(self.slow_seconds)
        return (float(decoded[0]), float(decoded[0]))


def _decode(genes):
    return genes


def test_eval_timeout_requires_multiple_workers():
    with pytest.raises(ValueError, match="n_workers"):
        NSGA2(n_genes=1, evaluate=_SleepEvaluator(), decode=_decode,
              n_workers=1, eval_timeout=1.0)


def test_eval_timeout_requires_a_worst_attribute_on_evaluate():
    bare_callable = lambda decoded: (0.0, 0.0)  # noqa: E731 -- fine, never pickled
    with pytest.raises(ValueError, match="worst"):
        NSGA2(n_genes=1, evaluate=bare_callable, decode=_decode,
              n_workers=2, eval_timeout=1.0)


def test_eval_timeout_bounds_a_slow_candidate_and_scores_it_worst_case():
    """One of 4 candidates sleeps 5s; eval_timeout=0.5s. The whole batch
    must finish in a couple of seconds, not five, and the slow candidate's
    objectives must equal .worst rather than its real (never-returned)
    result. n_workers >= pop_size so every candidate gets its own worker
    immediately -- otherwise a fast candidate queued behind the slow one on
    the same worker would time out too, and which candidates that happens
    to would depend on the pool's internal scheduling, not on this test."""
    # eval_timeout=3.0s / slow_seconds=8.0s, not the tighter 0.5s/5.0s this
    # started with: pool startup alone (fresh interpreter + numpy import per
    # worker, especially on Windows) can itself take more than half a second,
    # which made the tight margin trip on the FAST candidates too rather
    # than isolating the genuinely slow one. The gap between timeout and
    # sleep duration just needs to be comfortably wider than process-spawn
    # jitter, not razor-thin.
    ev = _SleepEvaluator(slow_seconds=8.0, threshold=0.5)
    ga = NSGA2(n_genes=1, evaluate=ev, decode=_decode, pop_size=4,
               n_workers=4, eval_timeout=3.0, verbose_eval=False)

    seeds = np.array([[0.1], [0.9], [0.2], [0.3]])  # row 1 (0.9) is the slow one
    t0 = time.perf_counter()
    ga.run(n_generations=0, seed_individuals=seeds)
    wall = time.perf_counter() - t0

    # run() returns only the Pareto-optimal subset, which here collapses to
    # one point (every objective is (val, val), so the smallest val
    # dominates everything) -- the per-individual record in .history is
    # what actually has all 4 results.
    obj = ga.history[0]["population_objectives"]

    assert wall < 6.0, f"batch took {wall:.1f}s -- eval_timeout did not bound the slow candidate"
    assert np.allclose(obj[1], ev.worst)
    assert ga.n_timeouts == 1
    # the three well-behaved candidates still got their REAL result, not a
    # timeout fallback -- a bug that timed out everything would still leave
    # wall time low, so this is the assertion that catches that failure mode
    for i in (0, 2, 3):
        assert np.allclose(obj[i], [seeds[i, 0], seeds[i, 0]])


def test_eval_timeout_none_leaves_the_original_parallel_path_unchanged():
    """No eval_timeout -> the pool.map() branch runs, same as before this
    existed. A candidate that WOULD exceed some timeout must be allowed to
    just finish normally when no timeout is set."""
    ev = _SleepEvaluator(slow_seconds=0.2, threshold=0.5)
    ga = NSGA2(n_genes=1, evaluate=ev, decode=_decode, pop_size=3,
               n_workers=3, eval_timeout=None, verbose_eval=False)

    seeds = np.array([[0.1], [0.9], [0.3]])
    ga.run(n_generations=0, seed_individuals=seeds)
    obj = ga.history[0]["population_objectives"]

    assert ga.n_timeouts == 0
    for i in range(3):
        assert np.allclose(obj[i], [seeds[i, 0], seeds[i, 0]])
