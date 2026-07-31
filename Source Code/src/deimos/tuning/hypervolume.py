"""
Hypervolume indicator and plateau detection for a multi-objective search.

WHY THIS EXISTS
---------------
The obvious convergence signal for NSGA-II is "has the best value of each
objective stopped improving", and that signal is wrong. Per-objective bests
are the *endpoints* of the front. In a 3-objective search the front can be
reshaping substantially in its interior -- the knee moving, the middle
filling in, dominated interior points being replaced -- while all three
endpoints sit perfectly flat. Stopping on flat endpoints stops the run while
it is still doing the work you actually care about.

Hypervolume is the standard single-number quality measure for a whole front
(Zitzler & Thiele 1998). It is the volume of objective space dominated by
the front, measured against a fixed reference point, so it is sensitive to
*all three* things a front can improve at once:

    convergence   -- points move toward the utopia corner, volume grows
    spread        -- the front covers more of the trade surface, volume grows
    cardinality   -- filling a gap in the interior adds volume

and it is strictly monotonic with Pareto dominance: if front A dominates
front B then HV(A) > HV(B), always. That is the property per-objective bests
lack, and it is what makes "HV has plateaued" a defensible stopping rule
rather than a guess.

THE REFERENCE POINT MUST BE FIXED
---------------------------------
HV is only comparable across generations (and across seeds) if the reference
point never moves. A reference derived from the current population -- e.g.
the per-objective max -- makes the number meaningless, because the box being
measured changes size every generation and HV can then fall while the front
strictly improves.

So the reference used here is the *worst-case objective tuple* the evaluator
already defines for infeasible candidates (objectives._worst_case). It is a
constant of the problem, not of the run: it depends only on the controller
type and the evaluation duration. Every seed, every generation and every
restart therefore measures against the same box, and the numbers can be
averaged and plotted against each other honestly.

Volumes are reported as a FRACTION of that box, i.e. HV / prod(reference).
That keeps the number in [0, 1] and readable ("the front dominates 38% of
the worst-case box") instead of a raw volume in units of deg s^2 * N m s *
dimensionless, which nobody can interpret.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# hypervolume
# --------------------------------------------------------------------------

def _hv2(pts: np.ndarray) -> float:
    """2D hypervolume, maximization, reference at the origin. O(n log n).

    Sweep in x descending, maintaining the largest y seen. A point only adds
    area if its y beats everything already processed, and because processing
    is x-descending every earlier point has a larger x -- so the strip
    (y_max, y] is covered out to exactly this point's x and no further.
    """
    if len(pts) == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])
    vol = 0.0
    y_max = 0.0
    for x, y in pts[order]:
        if y > y_max:
            vol += x * (y - y_max)
            y_max = y
    return float(vol)


def _hv_recursive(pts: np.ndarray, m: int) -> float:
    """M-dimensional hypervolume by slicing along the last axis.

    Maximization, reference at the origin, all coordinates >= 0. Slice the
    space at every distinct value of the last coordinate; within a slice the
    dominated region is exactly the (m-1)-dimensional hypervolume of the
    points that reach that deep, so the problem recurses one dimension down.

    O(n^2 log n) for m=3, which at these population sizes (front <= ~100) is
    a few milliseconds per generation -- utterly dominated by the simulate()
    calls it is measuring. No reason to reach for a fancier algorithm.
    """
    if len(pts) == 0:
        return 0.0
    if m == 1:
        return float(pts[:, 0].max())
    if m == 2:
        return _hv2(pts)

    order = np.argsort(-pts[:, m - 1])
    pts = pts[order]
    depths = pts[:, m - 1]

    vol = 0.0
    for i in range(len(pts)):
        lower = depths[i + 1] if i + 1 < len(pts) else 0.0
        height = depths[i] - lower
        if height > 0:
            vol += height * _hv_recursive(pts[: i + 1, : m - 1], m - 1)
    return float(vol)


def hypervolume(objectives, reference) -> float:
    """Volume of objective space dominated by `objectives`, bounded by
    `reference`. Minimization: a point counts only where it is BETTER than
    the reference in every objective.

    objectives: (n, M) array. Need not be the Pareto front -- dominated
        points contribute nothing extra, so passing a whole population gives
        the same answer as passing its front. Rows containing NaN/inf are
        dropped rather than poisoning the result.
    reference: (M,) the fixed worst-case corner. Must be strictly worse than
        the interesting region in every objective, and must never change
        between calls you intend to compare.
    """
    obj = np.atleast_2d(np.asarray(objectives, dtype=float))
    ref = np.asarray(reference, dtype=float).ravel()
    if obj.size == 0:
        return 0.0
    if obj.shape[1] != ref.size:
        raise ValueError(f"objectives have {obj.shape[1]} columns but reference has {ref.size}")

    obj = obj[np.isfinite(obj).all(axis=1)]
    if len(obj) == 0:
        return 0.0

    # Flip to maximization with the reference at the origin. A point worse
    # than the reference in any objective gives a non-positive coordinate and
    # dominates nothing, so those rows are dropped outright.
    v = ref[None, :] - obj
    v = v[(v > 0).all(axis=1)]
    if len(v) == 0:
        return 0.0

    return _hv_recursive(v, v.shape[1])


def hypervolume_fraction(objectives, reference) -> float:
    """hypervolume() as a fraction of the reference box's own volume, i.e. in
    [0, 1]. This is the number worth plotting and quoting: it is unitless and
    comparable across controller types, durations and seeds."""
    ref = np.asarray(reference, dtype=float).ravel()
    box = float(np.prod(ref))
    if not np.isfinite(box) or box <= 0:
        return float("nan")
    return hypervolume(objectives, ref) / box


# --------------------------------------------------------------------------
# plateau detection
# --------------------------------------------------------------------------

class StallDetector:
    """Stop when hypervolume has stopped growing meaningfully.

    The rule: look back `patience` generations. If the front has gained less
    than `tol` in RELATIVE hypervolume over that whole window, the search has
    plateaued and further generations are buying nothing worth the wall clock.

    Relative, not absolute, because HV fractions differ by an order of
    magnitude between controller types and scenarios -- an absolute threshold
    tuned on one is meaningless on another.

    `patience` is a real knob and worth stating in the report: too small and
    you stop during a normal flat stretch between improvements (NSGA-II
    progresses in bursts, because a single lucky mutation can open a new
    region of the front); too large and you pay for generations you already
    knew were dead. 20-30 is a reasonable default at these population sizes.

    min_generations exists so a warm-started run -- which can look flat for
    the first few generations precisely because it started from a good
    design -- is never stopped before the search has had a chance to move.
    """

    def __init__(self, patience: int = 25, tol: float = 1e-4,
                 min_generations: int = 40):
        self.patience = int(patience)
        self.tol = float(tol)
        self.min_generations = int(min_generations)
        self.history: list[float] = []

    def update(self, hv: float) -> None:
        self.history.append(float(hv))

    @property
    def relative_gain(self) -> float:
        """Relative HV gain over the last `patience` generations, or NaN if
        there is not enough history yet."""
        if len(self.history) <= self.patience:
            return float("nan")
        past = self.history[-1 - self.patience]
        now = self.history[-1]
        if not np.isfinite(past) or not np.isfinite(now) or past <= 0:
            return float("nan")
        return (now - past) / past

    def stalled(self) -> bool:
        if len(self.history) < max(self.min_generations, self.patience + 1):
            return False
        gain = self.relative_gain
        return bool(np.isfinite(gain) and gain < self.tol)

    def reason(self) -> str:
        return (f"hypervolume plateau: relative gain {self.relative_gain:.2e} "
                f"< {self.tol:.0e} over the last {self.patience} generations")
