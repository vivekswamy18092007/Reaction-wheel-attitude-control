"""
Tests for the Wie Case 1 eigenaxis regulator and its NSGA-II search
(controller_type="WIE").

The central test here is `test_wheel_momentum_decoupling_restores_eigenaxis`.
Everything else in this file is ordinary plumbing coverage; that one guards a
failure mode that is genuinely dangerous because it is SILENT: without the
wheel-momentum term the controller still converges, still settles in about the
same time, and still looks correct in every plot anyone normally looks at --
while the one property that makes it "the eigenaxis case" is gone. It is the
kind of bug an overnight gain search would faithfully optimize around.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from deimos.control.wie import WieRegulator, build_wie_from_config
from deimos.sim.config import compose_config
from deimos.sim.runner import simulate
from deimos.tuning import objectives as O

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
J = np.diag([1.4010e-2, 1.4130e-2, 2.9380e-3])


def _eigenaxis_config(scenario="slew_40_30_25.yaml", duration=60.0):
    cfg = compose_config(CONFIGS / "scenarios" / scenario,
                         CONFIGS / "controllers" / "wie_eigenaxis.yaml")
    cfg.sim.duration = duration
    return cfg


# --------------------------------------------------------------------------
# gains
# --------------------------------------------------------------------------

def test_eigenaxis_gains_are_scalar_multiples_of_J():
    K, D, mu = WieRegulator.eigenaxis_gains(J, k=0.32, d=0.8)
    assert np.allclose(K, 0.32 * J)
    assert np.allclose(D, 0.8 * J)
    assert mu == 1.0


def test_eigenaxis_gains_agree_with_the_sizing_rule():
    """The direct form must reach exactly the same (K, D) the (zeta, t_s)
    design rule does, or the warm start would seed a different controller
    than the preset it claims to come from."""
    zeta, t_s = 1.0, 20.0
    omega_n = 8.0 / (zeta * t_s)
    K1, D1, mu1 = WieRegulator.design(J, "eigenaxis", zeta=zeta, settling_time=t_s)
    K2, D2, mu2 = WieRegulator.eigenaxis_gains(J, k=2 * omega_n ** 2,
                                               d=2 * zeta * omega_n)
    assert np.allclose(K1, K2)
    assert np.allclose(D1, D2)
    assert mu1 == mu2 == 1.0


@pytest.mark.parametrize("k,d", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_eigenaxis_gains_reject_nonpositive_scales(k, d):
    """K and D must stay positive definite for the Sec. III guarantee. A
    non-positive scale is not a slightly-worse controller, it is outside the
    proof, so it raises rather than being clipped."""
    with pytest.raises(ValueError):
        WieRegulator.eigenaxis_gains(J, k=k, d=d)


# --------------------------------------------------------------------------
# the wheel-momentum decoupling term
# --------------------------------------------------------------------------

def test_wheel_momentum_decoupling_restores_eigenaxis():
    """THE test in this file.

    DEIMoS's plant is J*omega_dot = u - omega x (J*omega + h_w). The paper's
    mu=1 term cancels only omega x J*omega, leaving -omega x h_w -- a torque
    that is not along the error eigenaxis and that grows through the slew,
    because h_w is exactly what the wheels accumulate to perform it.

    Note what this test deliberately also asserts: settling time barely
    changes. That is the whole reason the bug is worth a dedicated test --
    the metric everyone checks is insensitive to it, so nothing else in the
    suite would catch a regression here.
    """
    cfg = _eigenaxis_config()

    on = simulate(cfg)
    off = copy.deepcopy(cfg)
    off.controller.decouple_wheel_momentum = False
    off = simulate(off)

    assert on.mean_eigenaxis_deviation_deg() < 0.05, (
        "with wheel-momentum decoupling on, Case 1 must hold the eigenaxis "
        "to well under a tenth of a degree")
    assert off.mean_eigenaxis_deviation_deg() > 1.0, (
        "without it the residual -omega x h_w should visibly bow the path; "
        "if this no longer holds the plant or the wheels changed")
    assert on.mean_eigenaxis_deviation_deg() < 0.1 * off.mean_eigenaxis_deviation_deg()

    # ... while the number a reader would check stays essentially the same.
    assert abs(on.settling_time(1.0) - off.settling_time(1.0)) < 1.0


def test_decoupling_flag_is_inert_for_mu_zero_cases():
    """Cases 2-4 use mu=0, so there is no decoupling term for h_w to enter.
    The controller must not then demand h_w from the runner."""
    K, D, mu = WieRegulator.design(J, "near_eigenaxis", zeta=1.0, settling_time=8.0)
    w = WieRegulator(J, K, D, mu, decouple_wheel_momentum=True)
    assert mu == 0.0
    assert w.needs_wheel_momentum is False
    # and it computes without h_w rather than raising
    w.compute_torque(np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))


def test_missing_h_w_raises_rather_than_silently_degrading():
    """Falling back to body-only decoupling would produce a plausible-looking
    but non-eigenaxis run. Loud failure is the only safe behaviour."""
    K, D, mu = WieRegulator.eigenaxis_gains(J, k=0.32, d=0.8)
    w = WieRegulator(J, K, D, mu, decouple_wheel_momentum=True)
    q = np.array([0.9, 0.2, 0.2, 0.2]); q /= np.linalg.norm(q)
    with pytest.raises(ValueError):
        w.compute_torque(q, np.array([0.01, 0.0, 0.0]))


def test_wie_case3_preset_is_unchanged():
    """Regression guard: the mu=0 preset's behaviour on the reference slew
    must stay pinned. Golden value re-recorded 2026-07 when the plant moved
    to the 3U CAD inertia tensor (was 14.08 s on the old placeholder
    tensor); K/D derive from J at build time, so the tensor swap legitimately
    changed the number once, and it must not drift again."""
    cfg = compose_config(CONFIGS / "scenarios" / "slew_40_30_25.yaml",
                         CONFIGS / "controllers" / "wie_case3.yaml")
    r = simulate(cfg)
    assert r.settling_time(1.0) == pytest.approx(14.58, abs=0.2)


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------

def test_direct_and_sizing_rule_forms_are_mutually_exclusive():
    with pytest.raises(ValueError):
        build_wie_from_config(
            {"case": "eigenaxis", "k_scale": 0.3, "d_scale": 0.8,
             "zeta": 1.0, "settling_time_s": 20.0}, J)


def test_k_scale_rejected_for_non_eigenaxis_cases():
    """K = k*J IS the eigenaxis case; cases 2-4 build K from J in ways with no
    single scale factor, so accepting k_scale there would silently substitute
    a different control law."""
    with pytest.raises(ValueError):
        build_wie_from_config({"case": "robust", "k_scale": 0.3, "d_scale": 0.8}, J)


def test_anti_windup_caps_the_integral_contribution():
    K, D, mu = WieRegulator.eigenaxis_gains(J, k=0.32, d=0.8)
    Ki = np.full(3, 1e-3)
    w = WieRegulator(J, K, D, mu, Ki=Ki, integral_limit=1e-5,
                     decouple_wheel_momentum=False)
    q = np.array([0.7071, 0.7071, 0.0, 0.0])
    for _ in range(5000):                       # long enough to wind up hard
        w.compute_torque(q, np.zeros(3), dt=0.01)
    assert np.all(np.abs(Ki * w._z) <= 1e-5 + 1e-12)


# --------------------------------------------------------------------------
# the GA search space
# --------------------------------------------------------------------------

def test_wie_gene_layout():
    assert O.n_genes("WIE") == 3
    assert O.gene_labels("WIE") == ["k", "d", "ki/k"]
    assert O.absolute_gain_labels("WIE") == ["k", "d", "ki"]
    assert len(O.gene_bounds("WIE")) == 3


def test_warm_start_round_trips_the_preset_design():
    """encode -> decode must return the preset's own k and d, or the elitism
    argument ('the front can never be worse than the design you started
    from') does not actually hold."""
    cfg = _eigenaxis_config()
    genes = O.encode_config_gains(cfg, "WIE")
    assert genes is not None and genes.shape == (3,)
    assert np.all((genes >= 0.0) & (genes <= 1.0))

    k, d, _ = (float(np.asarray(b).ravel()[0])
               for b in O.make_decode("WIE")(genes))
    omega_n = 8.0 / (cfg.controller.zeta * cfg.controller.settling_time_s)
    assert k == pytest.approx(2 * omega_n ** 2, rel=1e-9)
    assert d == pytest.approx(2 * cfg.controller.zeta * omega_n, rel=1e-9)


def test_pd_preset_cannot_warm_start_a_wie_search():
    """A PD preset carries no k/d. Returning None (start cold) is correct;
    silently seeding a meaningless point would be worse than not seeding."""
    cfg = compose_config(CONFIGS / "scenarios" / "slew_40_30_25.yaml",
                         CONFIGS / "controllers" / "pd_baseline.yaml")
    assert O.encode_config_gains(cfg, "WIE") is None


def test_apply_gains_writes_a_J_matched_eigenaxis_controller():
    cfg = _eigenaxis_config()
    decoded = (np.array([0.5]), np.array([1.2]), np.array([3e-4]))
    out = O.apply_gains(copy.deepcopy(cfg), decoded, "WIE")

    assert out.controller.type == "wie"
    assert out.controller.case == "eigenaxis"
    assert out.controller.k_scale == 0.5 and out.controller.d_scale == 1.2
    # forced on regardless of what the preset said -- see _apply_wie_gains
    assert out.controller.decouple_wheel_momentum is True
    # the sizing-rule fields must be cleared, or config.py rejects the config
    assert out.controller.zeta is None and out.controller.settling_time_s is None
    assert out.controller.Kp is None and out.controller.Kd is None
    # Ki must stay proportional to J, or the integral term breaks eigenaxis
    assert np.allclose(out.controller.Ki, 3e-4 * np.diag(
        np.asarray(cfg.satellite.inertia_tensor)))


def test_every_candidate_in_the_search_space_is_stable_and_simulable():
    """The whole (k, d) box satisfies Sec. III, so no candidate should blow
    up. Sampling the corners and centre is a cheap guard against a decode or
    bounds change that lets a non-positive gain through."""
    cfg = _eigenaxis_config(duration=30.0)
    decode = O.make_decode("WIE")
    for g in [np.zeros(3), np.ones(3), np.full(3, 0.5),
              np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.5])]:
        r = simulate(O.apply_gains(copy.deepcopy(cfg), decode(g), "WIE"))
        assert np.isfinite(r.q).all()
        assert np.isfinite(r.attitude_error_deg).all()


def test_evaluator_records_eigenaxis_deviation():
    cfg = _eigenaxis_config(duration=30.0)
    ev = O.Evaluator([cfg], controller_type="WIE", duration=30.0,
                     with_diagnostics=True)
    obj, diag = ev(O.make_decode("WIE")(O.encode_config_gains(cfg, "WIE")))

    assert len(obj) == 3 and all(np.isfinite(obj))
    i = list(O.DIAGNOSTIC_LABELS).index("mean_eigenaxis_deviation_deg")
    assert np.isfinite(diag[i]) and diag[i] < 0.05
