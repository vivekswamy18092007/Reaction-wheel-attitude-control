"""
Magnetorquer desaturation: the cross-product law's geometric properties, the
hardware dipole clip, the engagement hysteresis, and one short end-to-end
run proving momentum actually leaves the wheels.
"""

from pathlib import Path

import numpy as np
import pytest

from deimos.actuators.magnetorquers import MagnetorquerArray
from deimos.dynamics.environment import magnetic_field_inertial, magnetic_field_body
from deimos.sim.config import compose_config
from deimos.sim.runner import simulate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _engaged_mtq(**kwargs):
    """An array with the latch already tripped, so law properties can be
    probed without threading a momentum history through every test."""
    mtq = MagnetorquerArray(threshold=0.0, **kwargs)
    mtq._active = True
    return mtq


# --------------------------------------------------------------------------
# control-law geometry
# --------------------------------------------------------------------------

def test_torque_is_perpendicular_to_B():
    """tau = m x B can never have a component along B -- the fundamental
    magnetorquer limitation the law is built around."""
    mtq = _engaged_mtq()
    rng = np.random.default_rng(0)
    for _ in range(20):
        h = rng.normal(size=3) * 1e-3
        B = rng.normal(size=3) * 3e-5
        tau = mtq.torque(mtq.desat_dipole(h, B), B)
        assert abs(tau @ B) < 1e-18


def test_law_drains_the_perpendicular_momentum_component():
    """Unclipped, tau = -k * h_perp exactly: momentum decays, never grows."""
    k = 1e-3
    mtq = _engaged_mtq(k_desat=k, max_dipole=np.inf)
    rng = np.random.default_rng(1)
    for _ in range(20):
        h = rng.normal(size=3) * 1e-3
        B = rng.normal(size=3) * 3e-5
        tau = mtq.torque(mtq.desat_dipole(h, B), B)
        B_hat = B / np.linalg.norm(B)
        h_perp = h - B_hat * (B_hat @ h)
        assert np.allclose(tau, -k * h_perp)
        assert tau @ h <= 1e-18   # never pumps energy INTO the wheel momentum


def test_dipole_clips_to_hardware_limit():
    mtq = _engaged_mtq(k_desat=1.0, max_dipole=0.2)   # huge gain forces the clip
    m = mtq.desat_dipole(np.array([1.0, -1.0, 0.5]), np.array([0.0, 0.0, 3e-5]))
    assert np.all(np.abs(m) <= 0.2 + 1e-15)
    assert np.max(np.abs(m)) == pytest.approx(0.2)


def test_zero_field_commands_zero_dipole():
    mtq = _engaged_mtq()
    assert np.all(mtq.desat_dipole(np.ones(3), np.zeros(3)) == 0.0)


# --------------------------------------------------------------------------
# engagement hysteresis
# --------------------------------------------------------------------------

def test_latch_engages_above_threshold_and_holds_until_stop():
    mtq = MagnetorquerArray(threshold=1e-3, stop_frac=0.5)
    B = np.array([0.0, 0.0, 3e-5])

    # below threshold, never engaged: no command
    assert np.all(mtq.desat_dipole(np.array([5e-4, 0, 0]), B) == 0.0)
    assert not mtq.active

    # crosses threshold: engages
    assert np.any(mtq.desat_dipole(np.array([2e-3, 0, 0]), B) != 0.0)
    assert mtq.active

    # back inside the band but above stop: STAYS engaged (no chatter)
    assert np.any(mtq.desat_dipole(np.array([8e-4, 0, 0]), B) != 0.0)
    assert mtq.active

    # below stop_frac * threshold: disengages
    assert np.all(mtq.desat_dipole(np.array([4e-4, 0, 0]), B) == 0.0)
    assert not mtq.active


def test_reset_clears_the_latch():
    mtq = MagnetorquerArray(threshold=1e-3)
    mtq.desat_dipole(np.array([2e-3, 0, 0]), np.array([0.0, 0.0, 3e-5]))
    assert mtq.active
    mtq.reset()
    assert not mtq.active


# --------------------------------------------------------------------------
# geomagnetic field model
# --------------------------------------------------------------------------

def test_field_magnitude_in_dipole_range_at_500km():
    """The dipole model at 500 km spans roughly 20-50 uT depending on orbit
    position (1x equator to 2x poles, scaled by (R_E/r)^3)."""
    mags = [np.linalg.norm(magnetic_field_inertial(t)) for t in np.linspace(0, 6000, 100)]
    assert 1.5e-5 < min(mags) and max(mags) < 6e-5


def test_field_rotates_over_the_orbit():
    B0 = magnetic_field_inertial(0.0)
    B1 = magnetic_field_inertial(1000.0)   # ~1/6 orbit later
    cos = (B0 @ B1) / (np.linalg.norm(B0) * np.linalg.norm(B1))
    assert cos < 0.999   # direction genuinely changed


def test_body_frame_transform_preserves_magnitude():
    q = np.array([0.9, 0.2, 0.3, 0.1])
    q = q / np.linalg.norm(q)
    B_i = magnetic_field_inertial(123.0)
    B_b = magnetic_field_body(123.0, q)
    assert np.linalg.norm(B_b) == pytest.approx(np.linalg.norm(B_i))


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_desat_recovery_scenario_dumps_momentum():
    """Momentum leaves the wheels while attitude holds -- the whole point.
    Runs the first 600 s of the desat scenario (enough for a clear trend
    without the full 3000 s)."""
    cfg = compose_config(CONFIGS / "scenarios" / "desat_recovery.yaml",
                         CONFIGS / "controllers" / "wie_eigenaxis.yaml")
    cfg.sim.duration = 600.0
    r = simulate(cfg)

    assert r.desat_used()
    # meaningful fraction of the initial momentum gone already
    assert r.wheel_momentum_norm[-1] < 0.8 * r.wheel_momentum_norm[0]
    # monotone-ish decay while active: final < any early sample
    assert r.wheel_momentum_norm[-1] < r.wheel_momentum_norm[10]
    # and the body stayed pointed the whole time
    assert r.attitude_error_deg.max() < 1.0
