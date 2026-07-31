"""
Multi-objective gain tuning via NSGA-II, for the PD, PID and Wie-eigenaxis
control laws.

Chromosome (genes live in [0,1], log-uniform decoded to physical gains):

    PD  -- 6 genes:  Kp = [Kp_x, Kp_y, Kp_z],  Kd = [Kd_x, Kd_y, Kd_z]
    PID -- 9 genes:  the same, plus Ki = [Ki_x, Ki_y, Ki_z]
    WIE -- 3 genes:  k, d, ki  (scalars), giving K = k*J, D = d*J, mu = 1
                     and an integral gain ki*diag(J)

Objectives (all minimized):

    PD  -- (ITAE, control effort, saturation fraction)
    PID -- (ITAE, steady-state error, control effort)
    WIE -- (ITAE, steady-state error, control effort)
           PID and WIE additionally demote saturation to a hard constraint
           (see below), and share an objective set so their fronts can be
           overlaid directly.


WHY THE EIGENAXIS SEARCH IS THREE GENES AND NOT NINE
----------------------------------------------------
Wie/Weiss/Arapostathis Case 1 is the eigenaxis law, and the eigenaxis
property is a consequence of a specific structure, not of the gain values:
with mu = 1 the controller cancels the plant's gyroscopic term, and with
K = k*J and D = d*J the closed loop collapses to

    omega_dot = -d*omega - s*k*q_ev

in which J does not appear. All three components of q_ev then obey the same
scalar second-order equation, so omega stays parallel to the initial error
eigenaxis and the spacecraft turns about ONE fixed axis -- the shortest
angular path.

Handing that search six independent per-axis gains would not be a more
general version of this. It would be a different controller: the moment
K stops being a scalar multiple of J the cancellation is gone, the axes
converge at different rates, and the path bows off the eigenaxis. A 9-gene
"WIE" search would therefore spend its budget rediscovering that a
near-eigenaxis solution is a good compromise, having already thrown away the
guarantee that made the law worth implementing.

So three genes is the honest parameterization of the question "what are the
best gains for an eigenaxis maneuver". The trade surface is small but it is
real: k and d still trade settling against effort and saturation, and ki
still trades steady-state accuracy against overshoot.

The search space is also entirely stable by construction. Sec. III gives
global asymptotic stability for mu = 1 and any K = K^T > 0, D = D^T > 0, and
every k, d > 0 satisfies that against a positive-definite J -- so unlike the
PD/PID searches, no candidate the GA can propose is unstable, and no
stability constraint has to be added to the objectives to keep it away from
one. (The ki term is a heuristic augmentation outside that proof, which is
why it is bounded to a small ratio of k and given an anti-windup cap.)


WHY ITAE AND NOT SETTLING TIME
------------------------------
The original first objective was `settling_time(1 deg)`, capped at the run
duration when a candidate never settled. That cap is a sentinel, and a
sentinel is poison in a selection-based search: EVERY non-settling individual
scores exactly `duration`, so they all tie, and selection among them becomes
random. Early in a run -- exactly when the search most needs a gradient to
climb -- most of the population is in that tie. Worse, argmin() over that
column returns index 0 rather than anything meaningful, so the "fastest" pick
was not necessarily fast.

A soft penalty (duration + k * final_error) papers over the cliff but
introduces a magic constant k that has to be tuned and then defended.

ITAE removes the branch entirely:

    ITAE = integral of t * |theta_err(t)| dt        [deg s^2]

It is finite and strictly ordered for every candidate, so there is no tie and
no cliff. The `t` weight is what makes it a *settling* objective rather than
just an error objective: error early in a slew is unavoidable and barely
penalized, while error late in the window is multiplied by a large t and
punished hard. A run that never settles accumulates its worst penalty exactly
where a settling run has stopped accumulating at all, so non-settling
candidates score badly BY CONSTRUCTION rather than by a hand-set constant.

ITAE also distinguishes "missed by 2 deg" from "missed by 12 deg", which a
thresholded settling time cannot, and it penalizes a candidate that settles
fast and then drifts back out -- which a first-crossing settling time also
cannot.

Settling time is still computed and reported in the comparison table and in
the run report. It is just not what the search is steered by.


WHY PID GETS A DIFFERENT OBJECTIVE SET
--------------------------------------
Integral action exists to kill *steady-state* error against a persistent
bias. On a disturbance-free slew there is no such bias: the PD terms drive
the error to zero on their own, so every nonzero Ki can only add windup and
overshoot, and a GA scoring only (ITAE, effort, saturation) will correctly
but uselessly drive Ki toward its lower bound -- you would be paying for a
9-gene search to rediscover a PD.

So tuning a PID is only a meaningful question against a disturbance, and the
objective that makes it meaningful is the steady-state error the integral
term is there to remove. `tune()` will refuse to quietly ignore this: it
warns when asked to tune PID on a disturbance-free config, and both the CLI
and the overnight driver enable a gravity-gradient bias by default for PID.

Saturation is a *constraint* for PID rather than a fourth objective. Adding a
fourth objective dilutes Pareto dominance badly at these population sizes
(most of the population becomes mutually non-dominated and selection pressure
collapses), and saturation is not a quantity anyone wants to trade *for* --
it is a limit to respect. Candidates exceeding `saturation_limit` are
returned at worst-case, i.e. dominated by any feasible solution.


MULTI-SCENARIO EVALUATION
-------------------------
`make_evaluator` accepts either one config or a list. With a list, a
candidate is simulated on every scenario and the objectives are aggregated
elementwise WORST-CASE (max), so a gain set only scores well if it scores
well everywhere. This is the direct answer to "are these gains overfit to one
slew?" -- with worst-case aggregation they provably are not, because a gain
set that wins on scenario A by losing on scenario B is scored on its loss.

Mean aggregation is deliberately not offered: it lets a candidate buy a good
average by being excellent on the easy scenario and unacceptable on the hard
one, which is the exact failure mode the multi-scenario run exists to rule
out.

Fields confirmed against sim/results.py's SimResults dataclass:
  results.attitude_error_deg   -- (T,) error vs the config's OWN target
  results.settling_time(thr)   -- first time the error stays under thr, or None
  results.control_effort()     -- integral of |u| dt
  results.saturation_fraction()-- fraction of steps with any wheel at max_torque
"""

from __future__ import annotations

import copy
import warnings

import numpy as np

from deimos.tuning.nsga2 import NSGA2
from deimos.sim.runner import simulate


# ---------------- search space ----------------

KP_BOUNDS = (1e-4, 1e-1)   # log-uniform; contains the near_eigenaxis reference
KD_BOUNDS = (1e-4, 1e-1)   # gains computed earlier for this J

# Ki is searched as a RATIO of that axis's Kp, not as an absolute gain:
# control/pid.py's design rule is "keep Ki small relative to Kp/Kd", which an
# absolute bound cannot express when Kp itself spans three decades in the same
# search -- an absolute Ki of 1e-3 is negligible against Kp=1e-1 and
# catastrophic against Kp=1e-4. The upper end (0.1*Kp) sits just above the
# ratio implied by the hand-picked pair in configs/controllers/pid_example.yaml
# (Ki=1e-4 against Kp=6e-3, i.e. ~0.017), which that file documents as already
# near the windup limit -- so the range brackets the known-good design instead
# of spending half its span in the limit-cycling region.
KI_RATIO_BOUNDS = (1e-4, 1e-1)

# --- Wie eigenaxis regulator (controller_type="WIE") --------------------
#
# The chromosome is TWO scalars, not six per-axis gains, and that is the
# whole point rather than a simplification. Wie Case 1 is
#
#     u = (omega x J omega) - D omega - s K q_ev,   K = k*J,  D = d*J,  mu = 1
#
# and with mu=1 the gyroscopic term cancels exactly, leaving
#
#     omega_dot = -d*omega - s*k*q_ev
#
# with J gone. All three components of q_ev then obey the SAME scalar
# second-order equation, so omega stays parallel to the initial error
# eigenaxis and the body turns about one fixed axis -- the shortest angular
# path between the two attitudes. Searching per-axis K breaks the K = k*J tie
# that produces that cancellation: the axes then converge at different rates
# and the path bows off the eigenaxis. So the eigenaxis property is what
# COSTS the search four genes, and a 2-gene search is the honest
# parameterization of "tune an eigenaxis maneuver", not a shortcut.
#
# Bounds bracket the sizing-rule designs this spacecraft actually uses:
# design(zeta, t_s) gives k = 2*omega_n^2, d = 2*zeta*omega_n with
# omega_n = 8/(zeta*t_s), i.e. k ~ 0.14-2.6 and d ~ 0.53-2.0 across
# t_s = 8-30 s and zeta = 0.7-1.0. Three decades centered near there leaves
# room on both sides: the low end for the minimum-effort Pareto endpoint,
# the high end for the torque-saturated fast endpoint.
#
# Every point in this box is globally asymptotically stable: Sec. III
# guarantees it for mu=1 and any K = K^T > 0, D = D^T > 0, and k, d > 0 with
# J positive definite satisfies that. The search therefore needs no
# stability constraint -- it cannot propose an unstable candidate.
K_SCALE_BOUNDS = (1e-2, 1e1)   # k, where K = k*J
D_SCALE_BOUNDS = (1e-2, 1e1)   # d, where D = d*J

# ki is searched as a ratio of k for the same reason PID's Ki is searched as
# a ratio of Kp: "small relative to the proportional gain" is the actual
# design rule, and k spans three decades in this very search.
KI_WIE_RATIO_BOUNDS = (1e-4, 1e-1)

# (name, bounds, n_genes_in_block). A block of size 3 is per-axis; a block of
# size 1 is a single scalar shared by all three axes.
_SPEC = {
    "PD": (("Kp", KP_BOUNDS, 3), ("Kd", KD_BOUNDS, 3)),
    "PID": (("Kp", KP_BOUNDS, 3), ("Kd", KD_BOUNDS, 3), ("Ki/Kp", KI_RATIO_BOUNDS, 3)),
    "WIE": (("k", K_SCALE_BOUNDS, 1), ("d", D_SCALE_BOUNDS, 1),
            ("ki/k", KI_WIE_RATIO_BOUNDS, 1)),
}

# The physical gain each block decodes to, for CSV/report column names.
# Distinct from the search labels above because the last block is searched as
# a ratio but reported as an absolute gain.
_ABSOLUTE_NAMES = {"PD": ("Kp", "Kd"), "PID": ("Kp", "Kd", "Ki"),
                   "WIE": ("k", "d", "ki")}

OBJECTIVE_LABELS = {
    "PD": ("ITAE (deg s^2)", "control effort (N m s)", "saturation fraction"),
    "PID": ("ITAE (deg s^2)", "steady-state error (deg)", "control effort (N m s)"),
    # Deliberately the SAME three objectives as PID, so the two fronts are
    # directly comparable -- "what does the eigenaxis law buy or cost against
    # a tuned PID" is answerable by overlaying them, which it would not be if
    # each law were scored on its own objective set.
    #
    # Eigenaxis deviation is NOT an objective. Under Case 1 it is zero by
    # construction whenever the wheels are unsaturated and J is exact, so as
    # an objective it would be a near-constant column that dilutes Pareto
    # dominance while trading nothing. What it actually measures here is how
    # far torque saturation and the gravity-gradient bias push the real
    # maneuver off the ideal path -- which is a DIAGNOSTIC of the solution,
    # recorded for every individual (see DIAGNOSTIC_LABELS), not a knob.
    "WIE": ("ITAE (deg s^2)", "steady-state error (deg)", "control effort (N m s)"),
}

# Worst case for the two non-ITAE objectives, per type. The ITAE entry is
# computed from the duration (see worst_case) because it scales as duration^2.
_WORST_TAIL = {"PD": (1e3, 1.0), "PID": (180.0, 1e3), "WIE": (180.0, 1e3)}

# Maximum possible attitude error [deg]: a quaternion error of 180 deg.
_MAX_ATTITUDE_ERROR_DEG = 180.0


def worst_case(controller_type: str = "PD", duration: float = 120.0) -> tuple:
    """The objective tuple returned for an infeasible or numerically broken
    candidate -- and, equally importantly, the FIXED hypervolume reference
    point (see tuning/hypervolume.py).

    The ITAE entry is the ITAE of a run that sat at the maximum possible
    attitude error for the entire window:

        integral of t * 180 dt over [0, T]  =  90 * T^2

    That is a genuine upper bound on the objective, not a magic number, which
    is what makes it a legitimate reference: no achievable front point can
    ever fall outside the box, so the hypervolume fraction is always in [0,1]
    and is comparable across seeds and across restarts.
    """
    ctype = _normalize_type(controller_type)
    itae_worst = 0.5 * _MAX_ATTITUDE_ERROR_DEG * float(duration) ** 2
    return (itae_worst,) + _WORST_TAIL[ctype]


def _normalize_type(controller_type: str) -> str:
    t = str(controller_type).strip().upper()
    if t not in _SPEC:
        raise ValueError(f"controller_type must be one of {sorted(_SPEC)}, got '{controller_type}'")
    return t


def n_genes(controller_type: str = "PD") -> int:
    return sum(size for _, _, size in _SPEC[_normalize_type(controller_type)])


def gene_labels(controller_type: str = "PD") -> list[str]:
    """A size-3 block is per-axis and gets x/y/z suffixes; a size-1 block is a
    single scalar and keeps its bare name (`k`, not `k_x`), because suffixing
    it would imply a per-axis freedom the eigenaxis law does not have."""
    out = []
    for name, _, size in _SPEC[_normalize_type(controller_type)]:
        out += [f"{name}_{axis}" for axis in "xyz"] if size == 3 else [name]
    return out


def absolute_gain_labels(controller_type: str = "PD") -> list[str]:
    """Column names for the DECODED physical gains, matching the order of
    `np.concatenate(decode(genes))`. Differs from gene_labels() in that the
    trailing ratio block is named for the absolute gain it becomes."""
    ctype = _normalize_type(controller_type)
    out = []
    for name, (_, _, size) in zip(_ABSOLUTE_NAMES[ctype], _SPEC[ctype]):
        out += [f"{name}_{axis}" for axis in "xyz"] if size == 3 else [name]
    return out


def gene_bounds(controller_type: str = "PD") -> list[tuple[float, float]]:
    return [bounds
            for _, bounds, size in _SPEC[_normalize_type(controller_type)]
            for _ in range(size)]


def _log_uniform(g, lo, hi):
    return 10 ** (np.log10(lo) + g * (np.log10(hi) - np.log10(lo)))


class Decoder:
    """genes in [0,1]^N -> a tuple of per-axis gain vectors.

    A class rather than a closure purely so it survives pickling. Windows
    spawns worker processes rather than forking them, so anything crossing
    the process boundary for a parallel evaluation must be picklable, and a
    closure is not. Callable, so it is a drop-in for the old closure.
    """

    def __init__(self, controller_type: str = "PD"):
        self.controller_type = _normalize_type(controller_type)

    def __call__(self, genes):
        genes = np.asarray(genes, dtype=float)
        spec = _SPEC[self.controller_type]
        blocks, i = [], 0
        for _, bounds, size in spec:
            blocks.append(_log_uniform(genes[i:i + size], *bounds))
            i += size
        if self.controller_type in ("PID", "WIE"):
            # ratio -> absolute (Ki for PID, ki for WIE)
            blocks[2] = blocks[2] * blocks[0]
        return tuple(blocks)


def make_decode(controller_type: str = "PD"):
    """PD  -> (kp, kd);  PID -> (kp, kd, ki);  WIE -> (k, d, ki), where the
    last gene block is decoded as a RATIO of the first and multiplied through
    by it, so the integral gain is always small relative to the proportional
    gain by construction rather than by luck.

    Every block is a numpy array, including WIE's length-1 ones, so
    downstream code (concatenate, decoded_to_flat, the CSV writers) needs no
    per-type branch on scalar-vs-vector."""
    return Decoder(controller_type)


def _log_uniform_inverse(value, lo, hi):
    return (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))


def encode_config_gains(config, controller_type: str = "PD") -> np.ndarray | None:
    """Inverse of make_decode(): a SimConfig's existing Kp/Kd(/Ki) -> genes.

    Used to warm-start the search from the gains already in the preset. Genes
    are clipped into [0,1], so a hand-picked gain outside the search bounds
    seeds the nearest reachable point rather than silently escaping them.
    Returns None if the config has nothing usable to seed from (e.g. a Wie
    preset asked to seed a PD search: its K/D are derived from J rather than
    stored as PD-style gains).
    """
    ctype = _normalize_type(controller_type)

    if ctype == "WIE":
        return _encode_wie_gains(config)

    Kp, Kd = config.controller.Kp, config.controller.Kd
    if Kp is None or Kd is None:
        return None

    kp, kd = np.diag(np.asarray(Kp)), np.diag(np.asarray(Kd))
    if np.any(kp <= 0) or np.any(kd <= 0):
        return None

    genes = [_log_uniform_inverse(kp, *KP_BOUNDS), _log_uniform_inverse(kd, *KD_BOUNDS)]

    if ctype == "PID":
        Ki = config.controller.Ki
        # A PD preset carries no Ki; seed the geometric middle of the ratio
        # range rather than refusing to warm-start, so the Kp/Kd knowledge
        # is still used and only the ratio starts uninformed.
        ratio = (np.full(3, np.sqrt(KI_RATIO_BOUNDS[0] * KI_RATIO_BOUNDS[1]))
                 if Ki is None else np.asarray(Ki, dtype=float) / kp)
        genes.append(_log_uniform_inverse(np.clip(ratio, *KI_RATIO_BOUNDS), *KI_RATIO_BOUNDS))

    return np.clip(np.concatenate(genes), 0.0, 1.0)


def _encode_wie_gains(config) -> np.ndarray | None:
    """Warm start for the WIE search, from whichever form the preset uses.

    This is where the t_s = 8/(zeta*omega_n) sizing rule earns its keep. It is
    a poor CONSTRAINT on the search (see K_SCALE_BOUNDS) because it is a
    linearized, unsaturated, disturbance-free estimate -- but it is an
    excellent STARTING POINT, because it is a physically reasoned guess at
    the right order of magnitude rather than a random draw. NSGA-II is
    elitist, so seeding it guarantees the returned front is never dominated
    by the textbook design: the search can only improve on the sizing rule,
    never regress below it. That is exactly the claim worth being able to
    make about a GA result.

    Returns None if the preset is not a Wie one at all, in which case the
    search starts cold rather than seeding a meaningless point.
    """
    c = config.controller
    if c.k_scale is not None and c.d_scale is not None:
        k, d = float(c.k_scale), float(c.d_scale)
    elif c.zeta is not None and c.settling_time_s is not None:
        # Mirror of WieRegulator.design()'s Case 1 arithmetic.
        omega_n = 8.0 / (float(c.zeta) * float(c.settling_time_s))
        k, d = 2.0 * omega_n ** 2, 2.0 * float(c.zeta) * omega_n
    else:
        return None

    if k <= 0.0 or d <= 0.0:
        return None

    genes = [_log_uniform_inverse(np.array([k]), *K_SCALE_BOUNDS),
             _log_uniform_inverse(np.array([d]), *D_SCALE_BOUNDS)]

    # A preset carrying no Ki seeds the geometric middle of the ratio range
    # rather than refusing to warm-start: the k/d knowledge is still worth
    # using, and only the integral ratio starts uninformed.
    Ki = c.Ki
    if Ki is None:
        ratio = np.sqrt(KI_WIE_RATIO_BOUNDS[0] * KI_WIE_RATIO_BOUNDS[1])
    else:
        # Ki is stored as the length-3 vector ki*diag(J); recover the scalar.
        Jd = np.diag(np.asarray(config.satellite.inertia_tensor, dtype=float))
        ki = float(np.mean(np.asarray(Ki, dtype=float) / Jd))
        ratio = ki / k if ki > 0 else np.sqrt(KI_WIE_RATIO_BOUNDS[0]
                                              * KI_WIE_RATIO_BOUNDS[1])
    genes.append(_log_uniform_inverse(
        np.clip(np.array([ratio]), *KI_WIE_RATIO_BOUNDS), *KI_WIE_RATIO_BOUNDS))

    return np.clip(np.concatenate(genes), 0.0, 1.0)


def decoded_to_flat(decoded, controller_type: str = "PD") -> np.ndarray:
    """(kp, kd[, ki]) -> a flat length-6/9 vector matching gene_labels().

    For PID the Ki block is converted back to the Ki/Kp ratio the search
    actually operates on, so it lands in the same [bounds] the parallel-
    coordinates plot normalizes against.
    """
    blocks = [np.atleast_1d(np.asarray(b, dtype=float)) for b in decoded]
    if _normalize_type(controller_type) in ("PID", "WIE"):
        blocks[2] = blocks[2] / blocks[0]
    return np.concatenate(blocks)


def normalized_gene_position(decoded, controller_type: str = "PD") -> np.ndarray:
    """Where each gene sits inside its own log-uniform search bounds, in
    [0,1]: 0 = lower bound, 1 = upper bound.

    This is what answers "did the search pin this gain against its bound?",
    which is the question behind the gene-bounded vs actuator-bounded
    diagnostic in tuning/report.py -- and a question evaluators ask.
    """
    flat = decoded_to_flat(decoded, controller_type)
    bounds = gene_bounds(controller_type)
    lo = np.array([np.log10(b[0]) for b in bounds])
    hi = np.array([np.log10(b[1]) for b in bounds])
    return (np.log10(flat) - lo) / (hi - lo)


# ---------------- applying a candidate to a config ----------------

def apply_gains(cfg, decoded, controller_type: str = "PD"):
    """Write a decoded gain set onto a SimConfig, in place."""
    ctype = _normalize_type(controller_type)

    if ctype == "WIE":
        return _apply_wie_gains(cfg, decoded)

    cfg.controller.type = ctype
    cfg.controller.Kp = np.diag(decoded[0])
    cfg.controller.Kd = np.diag(decoded[1])
    # Ki is a length-3 VECTOR, not a matrix -- both PDController and
    # PIDController multiply it elementwise against the accumulator.
    # For PD it must be cleared explicitly: seeding from a PID preset would
    # otherwise leave a stale Ki set and silently tune a PID while reporting
    # PD gains.
    cfg.controller.Ki = np.asarray(decoded[2], dtype=float) if ctype == "PID" else None
    return cfg


def _apply_wie_gains(cfg, decoded):
    """Write a decoded (k, d, ki) onto a SimConfig as a Wie Case 1 preset.

    The integral gain needs care. WieRegulator applies it ELEMENTWISE,
    `u -= Ki * z`, so writing a plain scalar-times-identity Ki would add a
    term that is NOT proportional to J -- and that term is precisely what
    breaks the eigenaxis property the rest of this controller is built to
    preserve. Setting Ki = ki * diag(J) makes the elementwise product equal
    ki*J @ z for this spacecraft's diagonal inertia tensor, so the integral
    term is J-matched like K and D are and the closed loop stays

        omega_dot = -d*omega - s*k*q_ev - ki*z

    with J absent from every term. Off-diagonal inertia would break the
    equivalence, which is safe here (confirmed diagonal-dominant, products of
    inertia ~1e-7) but is the assumption to revisit if the CAD model changes.

    The PD-style Kp/Kd fields are cleared explicitly: a config composed from
    a PD or PID preset would otherwise carry stale gains that the Wie branch
    of the registry ignores but any later inspection of the config would
    report as if they were live.
    """
    J = np.asarray(cfg.satellite.inertia_tensor, dtype=float)
    k = float(np.asarray(decoded[0]).ravel()[0])
    d = float(np.asarray(decoded[1]).ravel()[0])
    ki = float(np.asarray(decoded[2]).ravel()[0])

    cfg.controller.type = "wie"
    cfg.controller.case = "eigenaxis"
    cfg.controller.k_scale = k
    cfg.controller.d_scale = d
    # Forced on for every candidate, not inherited from the preset. Without
    # it the mu=1 decoupling misses the wheels' own momentum and the
    # candidate is not an eigenaxis controller at all (3.65 deg mean
    # deviation vs 0.005 deg -- see control/wie.py). A search that inherited
    # this from a preset could quietly spend a whole night optimizing the
    # wrong control law because one YAML key was absent.
    cfg.controller.decouple_wheel_momentum = True
    # Mutually exclusive with k_scale/d_scale -- see sim/config.py.
    cfg.controller.zeta = None
    cfg.controller.settling_time_s = None
    cfg.controller.Kp = None
    cfg.controller.Kd = None
    cfg.controller.Ki = ki * np.diag(J)
    return cfg


# ---------------- fitness ----------------

def itae(attitude_error_deg, t) -> float:
    """Integral of t * |theta_err| dt  [deg s^2].

    The time weight is the whole point: it makes late error expensive and
    early error nearly free, so this is a settling measure, not just an error
    measure. See the module docstring for why this replaced a thresholded
    settling time as the search objective.
    """
    t = np.asarray(t, dtype=float)
    e = np.abs(np.asarray(attitude_error_deg, dtype=float))
    return float(np.trapezoid(t * e, t))


# Quantities recorded for EVERY individual of EVERY generation alongside the
# objectives. None of these steer the search -- they are the engineering
# numbers you actually report and defend, and they are free because the
# simulation that produced the objectives already computed them.
#
# Settling time is here rather than in the objectives for exactly the reason
# the module docstring gives: as a SEARCH signal it is a sentinel that ties
# every non-settling candidate, but as a REPORTED number it is the thing a
# reader understands immediately. Recording it costs nothing and means the
# question "what settling time did generation 40 achieve" has an answer
# without re-running anything.
DIAGNOSTIC_LABELS = (
    "settling_time_s",         # NaN = never settled within the window
    "steady_state_error_deg",
    "final_error_deg",
    "overshoot_deg",
    "control_effort_Nms",
    "itae_deg_s2",
    "saturation_fraction",
    "peak_wheel_torque_Nm",
    "max_wheel_speed_rpm",
    # Time-average angle between omega and the initial error eigenaxis, over
    # the samples where the body was actually moving. ~0 means the maneuver
    # was a true shortest-path rotation.
    #
    # For a WIE (Case 1) candidate this is the honesty check on the whole
    # exercise: the law is eigenaxis by construction only for exact J,
    # unsaturated wheels and no external torque, and this run has a
    # gravity-gradient bias on and a 4e-3 N m wheel limit. So a nonzero
    # number here is not a bug -- it is the measured cost of running an ideal
    # law on real hardware, and it is the quantity that says whether the
    # fastest gains on the front bought their speed by abandoning the
    # eigenaxis path.
    #
    # For PD/PID it is free (the simulation already computed it) and gives
    # the baseline the eigenaxis law is supposed to beat.
    "mean_eigenaxis_deviation_deg",
)

_N_DIAG = len(DIAGNOSTIC_LABELS)


class Evaluator:
    """Scores one decoded gain set. Callable, and picklable so it can be sent
    to worker processes for parallel evaluation.

    base_configs: one SimConfig or a list of them. With more than one, the
        candidate is simulated on each and the objective vectors are
        aggregated elementwise WORST-CASE. Every config is deep-copied per
        evaluation and only the controller fields are overridden, so J,
        hardware, dt and disturbance stay exactly as configured and every
        candidate is scored on identical plants.

    duration: overrides each config's own sim.duration, so all scenarios are
        scored over the same window and their ITAEs are commensurate.
    steady_state_frac: tail fraction of the run averaged for the PID
        steady-state-error objective (0.2 = last 20%).
    saturation_limit: PID only -- candidates whose torque saturation exceeds
        this are treated as infeasible (worst-case objectives).
    integral_limit_frac: anti-windup cap on the integral term's contribution,
        as a fraction of the per-wheel torque limit. Without it the
        accumulator integrates the entire slew transient and, at the top of
        the Ki/Kp range, demands several times the wheel limit -- every
        candidate then limit-cycles and the search returns gains worse than
        an untuned PD. Set to None to reproduce that unbounded behaviour.

    with_diagnostics: when True, __call__ returns (objectives, diagnostics)
        instead of just objectives, where diagnostics is a vector matching
        DIAGNOSTIC_LABELS. NSGA2 detects this via the `diagnostic_labels`
        attribute and stores the extra columns in its history, so every
        individual of every generation carries its settling time, effort and
        steady-state error -- not just the three numbers the search sorts on.

    Saturation is read from results.saturation_fraction(), which already
    thresholds against the max_torque of the ReactionWheelArray actually
    built for this run -- so there is no separate u_max to pass in and no way
    for it to drift out of sync with the hardware limit the sim enforced.
    """

    def __init__(self, base_configs, controller_type: str = "PD", duration=120.0,
                 settle_threshold_deg=1.0, steady_state_frac=0.2,
                 saturation_limit=0.10, integral_limit_frac=0.25,
                 with_diagnostics=False):
        self.controller_type = _normalize_type(controller_type)
        self.base_configs = (list(base_configs)
                             if isinstance(base_configs, (list, tuple))
                             else [base_configs])
        self.duration = float(duration)
        self.settle_threshold_deg = float(settle_threshold_deg)
        self.steady_state_frac = float(steady_state_frac)
        self.saturation_limit = saturation_limit
        self.integral_limit_frac = integral_limit_frac
        self.with_diagnostics = bool(with_diagnostics)
        self.worst = worst_case(self.controller_type, self.duration)

    @property
    def diagnostic_labels(self):
        """NSGA2 reads this to decide whether __call__ returns diagnostics.
        None means "objectives only" (the plain, original contract)."""
        return DIAGNOSTIC_LABELS if self.with_diagnostics else None

    def _worst_diagnostics(self):
        """A broken candidate has no meaningful engineering numbers. NaN, not
        zero: zero settling time and zero effort would look like a PERFECT
        candidate in any downstream min()/plot, which is exactly backwards."""
        return np.full(_N_DIAG, np.nan)

    # -- one scenario --------------------------------------------------
    def _score_one(self, base_config, decoded):
        """-> (objectives tuple, diagnostics vector)."""
        cfg = apply_gains(copy.deepcopy(base_config), decoded, self.controller_type)
        cfg.sim.duration = self.duration
        if self.integral_limit_frac is not None:
            # Scaled off the hardware limit this run actually enforces, so it
            # cannot drift out of sync with the wheels the sim built.
            cfg.controller.integral_limit = self.integral_limit_frac * cfg.wheels.max_torque

        try:
            results = simulate(cfg)
        except Exception:
            # non-convergent / numerically broken gain set -- penalize to
            # worst-case rather than crashing the GA run
            return self.worst, self._worst_diagnostics()

        if (not np.isfinite(results.q).all() or not np.isfinite(results.u).all()
                or not np.isfinite(results.attitude_error_deg).all()):
            return self.worst, self._worst_diagnostics()

        score_itae = itae(results.attitude_error_deg, results.t)
        effort = results.control_effort()
        saturation = results.saturation_fraction()

        tail = results.attitude_error_deg[
            int((1.0 - self.steady_state_frac) * len(results.t)):]
        steady_state_error = float(np.mean(tail))

        settle = results.settling_time(self.settle_threshold_deg)
        diagnostics = np.array([
            np.nan if settle is None else float(settle),
            steady_state_error,
            results.final_attitude_error_deg(),
            results.overshoot_deg(),
            effort,
            score_itae,
            saturation,
            results.peak_wheel_torque(),
            results.max_wheel_speed() * 60.0 / (2.0 * np.pi),
            results.mean_eigenaxis_deviation_deg(),
        ], dtype=float)

        if self.controller_type == "PD":
            return (score_itae, effort, saturation), diagnostics

        # --- PID and WIE (same objective triple, same feasibility rule) ---
        if (self.saturation_limit is not None
                and saturation > self.saturation_limit):
            # Infeasible: dominated by every feasible candidate. The
            # diagnostics are still real measurements of what this gain set
            # did, so they are kept -- that is how you see *how* infeasible
            # the rejected region was.
            return self.worst, diagnostics

        return (score_itae, steady_state_error, effort), diagnostics

    # -- all scenarios, worst-case aggregated --------------------------
    def _aggregate(self, decoded):
        pairs = [self._score_one(cfg, decoded) for cfg in self.base_configs]
        scores = [p[0] for p in pairs]
        diags = np.asarray([p[1] for p in pairs], dtype=float)

        if len(scores) == 1:
            return scores[0], diags[0]

        # Elementwise max. Note this is a per-objective worst case, so the
        # aggregate is not necessarily attained by any single scenario -- that
        # is intentional and conservative: it is the tightest bound you can
        # state for "this gain set, on any of these maneuvers".
        obj = tuple(np.max(np.asarray(scores, dtype=float), axis=0))

        # Diagnostics aggregate worst-case too, so every reported number is
        # "the worst this gain set did on any scenario". nanmax would hide a
        # scenario that never settled behind one that did, so settling time
        # keeps plain max semantics: NaN in, NaN out.
        with np.errstate(invalid="ignore"):
            diag = np.max(diags, axis=0)
        return obj, diag

    def __call__(self, decoded):
        obj, diag = self._aggregate(decoded)
        return (obj, diag) if self.with_diagnostics else obj

    # -- diagnostics, not used by the search ---------------------------
    def per_scenario(self, decoded) -> list[tuple]:
        """The un-aggregated per-scenario objective vectors. Used by the
        report to show how much spread there is between scenarios for a
        chosen gain set -- a large spread means the worst-case aggregation
        was doing real work."""
        return [self._score_one(cfg, decoded)[0] for cfg in self.base_configs]

    def per_scenario_diagnostics(self, decoded) -> list[np.ndarray]:
        """Per-scenario DIAGNOSTIC_LABELS vectors, un-aggregated."""
        return [self._score_one(cfg, decoded)[1] for cfg in self.base_configs]


def make_evaluator(base_config, controller_type: str = "PD", **kwargs):
    """Backwards-compatible constructor for Evaluator."""
    return Evaluator(base_config, controller_type=controller_type, **kwargs)


# ---------------- Pareto front utilities ----------------

def knee_point(obj):
    """
    Index of the knee point: closest (in normalized objective space) to
    the utopia point (min of each objective). Reasonable default pick
    when you don't have explicit preference weights.
    """
    obj = np.asarray(obj, dtype=float)
    lo, hi = obj.min(axis=0), obj.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    norm = (obj - lo) / span
    dist = np.linalg.norm(norm, axis=1)
    return int(np.argmin(dist))


def pareto_picks(obj, controller_type: str = "PD") -> dict[str, int]:
    """{label: row index into obj/pop} for the handful of solutions worth
    re-simulating: the best in each individual objective, plus the knee.

    The first objective is ITAE for both types, so its winner is called
    "best_tracking" and not "fastest": ITAE rewards converging quickly AND
    staying converged, so the winner is not always the one that first crosses
    the 1 deg threshold.
    """
    ctype = _normalize_type(controller_type)
    obj = np.asarray(obj, dtype=float)
    if ctype == "PD":
        picks = {"best_tracking": 0, "cheapest_effort": 1, "least_saturated": 2}
    else:
        picks = {"best_tracking": 0, "most_accurate": 1, "cheapest_effort": 2}
    out = {label: int(np.argmin(obj[:, i])) for label, i in picks.items()}
    out["knee"] = knee_point(obj)
    return out


def to_runs_dict(pop, obj, decode, prefix="GA", controller_type: str = "PD"):
    """
    Package the selected Pareto points as gain sets labeled the same way
    studies/explore.ipynb's `runs` dict is built, so they can be re-simulated
    and dropped straight into compare()/comparison_table alongside the PD/Wie
    cases already there.
    """
    ctype = _normalize_type(controller_type)
    out = {}
    for label, idx in pareto_picks(obj, ctype).items():
        decoded = decode(pop[idx])
        if ctype == "WIE":
            # k, d, ki are scalars, not 3x3 matrices -- K = k*J is built at
            # controller-construction time from the plant's own J, so storing
            # a matrix here would freeze in whichever J happened to be
            # loaded and quietly break if the run is replayed on another one.
            entry = {"k": float(np.asarray(decoded[0]).ravel()[0]),
                     "d": float(np.asarray(decoded[1]).ravel()[0]),
                     "ki": float(np.asarray(decoded[2]).ravel()[0])}
        else:
            entry = {"Kp": np.diag(decoded[0]), "Kd": np.diag(decoded[1]),
                     "Ki": np.asarray(decoded[2]) if ctype == "PID" else None}
        out[f"{prefix}_{label}"] = {
            "index": idx, "type": ctype,
            **entry,
            "objectives": tuple(np.asarray(obj)[idx]),
        }
    return out


def history_gains(history, controller_type: str = "PD", generation=-1,
                    front_only=False):
    """
    Decode a whole generation's genes (stored by NSGA2 when
    store_full_population=True, the default) into physical Kp/Kd(/Ki)
    arrays, without re-running the GA.

    history: ga.history, or the "history" key loaded back from a
        checkpoint_path/save_path pickle.
    generation: index into history (default -1, the last recorded
        generation -- NOT necessarily the returned Pareto front if the run
        was cut short, but for a completed run these coincide).
    front_only: True -> only this generation's Pareto front (always
        available, even if store_full_population was off). False -> the
        full population for that generation (requires
        store_full_population=True at GA construction time; raises if that
        data wasn't kept).

    Returns a dict with "Kp", "Kd", "Ki" (each an (n, 3) array of per-axis
    gains, Ki all-None for PD) and "objectives" ((n, M) array), n either
    front_size or pop_size depending on front_only.
    """
    ctype = _normalize_type(controller_type)
    decode = make_decode(ctype)
    entry = history[generation]

    key_genes = "front_genes" if front_only else "population_genes"
    key_obj = "front_objectives" if front_only else "population_objectives"

    if key_genes not in entry:
        raise KeyError(
            f"'{key_genes}' not in this history entry -- if front_only=False, "
            "this run's NSGA2 was constructed with store_full_population=False "
            "(or this is an old checkpoint from before that field existed), "
            "so only the Pareto front (front_only=True) is recoverable.")

    genes, obj = entry[key_genes], entry[key_obj]
    decoded = [decode(g) for g in genes]

    if ctype == "WIE":
        return {
            "k": np.array([d[0] for d in decoded]).ravel(),
            "d": np.array([d[1] for d in decoded]).ravel(),
            "ki": np.array([d[2] for d in decoded]).ravel(),
            "objectives": np.asarray(obj),
        }

    return {
        "Kp": np.array([d[0] for d in decoded]),
        "Kd": np.array([d[1] for d in decoded]),
        "Ki": (np.array([d[2] for d in decoded]) if ctype == "PID" else None),
        "objectives": np.asarray(obj),
    }


def merge_fronts(fronts_obj, fronts_genes):
    """Pool several runs' Pareto fronts and return the non-dominated subset
    of the pool, plus which run each survivor came from.

    This is what makes a multi-seed run more than three separate answers: the
    merged front is the best trade surface the whole overnight budget found,
    and how much of it each seed contributed is a direct, honest measure of
    run-to-run variance (one seed supplying the entire merged front means the
    others converged to a strictly worse surface).

    Returns (obj, genes, source_index).
    """
    from deimos.tuning.nsga2 import fast_non_dominated_sort

    obj = np.vstack([np.atleast_2d(o) for o in fronts_obj])
    genes = np.vstack([np.atleast_2d(g) for g in fronts_genes])
    source = np.concatenate([np.full(len(np.atleast_2d(o)), i)
                             for i, o in enumerate(fronts_obj)])

    keep = fast_non_dominated_sort(obj)[0]
    keep = np.asarray(sorted(keep))
    return obj[keep], genes[keep], source[keep]


# ---------------- entry point ----------------

def tune(base_config, controller_type: str = "PD", pop_size=40, n_generations=80,
          seed=0, verbose=True, warm_start=True, checkpoint_path=None,
          save_path=None, n_workers=1, status_path=None, deadline=None,
          stall_patience=None, stall_tol=1e-4, stall_min_generations=40,
          eval_timeout=None, **evaluator_kwargs):
    """
    Returns (pop, obj, decode, ga) -- `ga` carries `.history`, the
    per-generation record viz.tuning.plot_convergence() draws.

    base_config: one SimConfig, or a list of them for worst-case
        multi-scenario evaluation (see Evaluator).

    warm_start: seed the initial population with base_config's existing gains
        (see encode_config_gains). Because NSGA-II is elitist this guarantees
        the returned front is never dominated by the gains you started with --
        the search can only improve on the preset, never regress from it.
        With a list of configs the FIRST one supplies the seed.

    n_workers: processes used to evaluate a generation. Evaluations within a
        generation are completely independent, so this is close to a linear
        speedup. 1 keeps the original in-process behaviour.

    checkpoint_path: forwarded to NSGA2.run() -- (pop, obj, history) pickled
        to this path after EVERY generation, so a killed kernel loses at most
        one generation's worth of simulate() calls instead of the whole run.

    status_path: small JSON written every generation (generation, hypervolume,
        elapsed, ETA). Cheap to poll from another process or a phone.

    deadline: absolute time.time() after which the run stops cleanly at the
        next generation boundary. This is what makes an overnight budget a
        budget rather than a hope.

    stall_patience: stop when the hypervolume of the front has gained less
        than `stall_tol` relative over this many generations. None disables
        plateau stopping (fixed generation count only).

    save_path: if given, pickle the FINAL (pop, obj, ga.history, runs_dict)
        here once the run completes.

    eval_timeout: [s] forwarded to NSGA2 -- see its docstring. None (default)
        trusts every candidate to evaluate in bounded time, same as before
        this existed. Set it (n_workers > 1 required) to stop one pathological
        candidate from blocking an entire generation, and with it, an entire
        wall-clock deadline.
    """
    ctype = _normalize_type(controller_type)
    configs = (list(base_config) if isinstance(base_config, (list, tuple))
               else [base_config])

    if ctype in ("PID", "WIE") and not any(getattr(c.disturbance, "enabled", False)
                                           for c in configs):
        warnings.warn(
            f"Tuning {ctype} on a disturbance-free config: with no steady bias to "
            "reject, integral action has nothing to do and the search will "
            "drive Ki toward its lower bound (i.e. rediscover a PD). Enable "
            "a disturbance -- e.g. dynamics.disturbances.gravity_gradient_torque(J), "
            "or the `deimos tune` CLI's default --gravity-gradient -- for a "
            "meaningful PID result.",
            RuntimeWarning,
            stacklevel=2,
        )

    decode = make_decode(ctype)
    # with_diagnostics on by default: settling time, effort and steady-state
    # error for every individual of every generation cost nothing extra (the
    # simulation already computed them) and are what the report is written
    # from. Pass with_diagnostics=False explicitly to opt out.
    evaluator_kwargs.setdefault("with_diagnostics", True)
    evaluate = Evaluator(configs, controller_type=ctype, **evaluator_kwargs)

    ga = NSGA2(n_genes=n_genes(ctype), evaluate=evaluate, decode=decode,
               pop_size=pop_size, seed=seed, verbose_eval=verbose,
               n_workers=n_workers,
               hv_reference=evaluate.worst,
               stall_patience=stall_patience, stall_tol=stall_tol,
               stall_min_generations=stall_min_generations,
               eval_timeout=eval_timeout)

    seeds = encode_config_gains(configs[0], ctype) if warm_start else None
    if verbose and warm_start:
        print("warm start : "
              + ("seeded with the preset's existing gains"
                 if seeds is not None
                 else "preset has no PD-style Kp/Kd to seed from, starting cold"))

    pop, obj = ga.run(n_generations, verbose=verbose, seed_individuals=seeds,
                       checkpoint_path=checkpoint_path, status_path=status_path,
                       deadline=deadline)

    if save_path is not None:
        import pickle
        runs = to_runs_dict(pop, obj, decode, controller_type=ctype)
        with open(save_path, "wb") as f:
            pickle.dump({
                "pop": pop, "obj": obj, "history": ga.history, "runs": runs,
                "diag": getattr(ga, "final_diagnostics", None),
                "diagnostic_labels": ga.diagnostic_labels,
                "controller_type": ctype, "seed": seed,
                "pop_size": pop_size, "n_generations": n_generations,
                "hv_reference": np.asarray(evaluate.worst),
                "eval_duration_s": evaluate.duration,
                "n_scenarios": len(configs),
                "scenario_names": [c.name for c in configs],
                "stop_reason": ga.stop_reason,
            }, f)
        if verbose:
            print(f"saved final GA result (pop/obj/history/runs) to {save_path}", flush=True)

    return pop, obj, decode, ga
