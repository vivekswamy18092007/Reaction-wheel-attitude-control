"""
Wheel speed saturation as PLANT physics: a pinned wheel accepts no further
accelerating torque (allocate() with Omega), and the propagator actually
holds the speed at the limit instead of integrating through it.
"""

import numpy as np
import pytest

from deimos.actuators.reaction_wheels import ReactionWheelArray


def _wheels(**kwargs):
    return ReactionWheelArray(config="orthogonal", max_torque=4e-3,
                              max_speed=100.0, **kwargs)


def test_pinned_wheel_rejects_accelerating_torque():
    w = _wheels(Omega0=[100.0, 0.0, 0.0])
    # torque about -x maps to g = +max on wheel 1 (g = -W^+ tau), pushing it
    # further past its +100 rad/s limit -> must be zeroed
    g, _ = w.allocate(np.array([-4e-3, 0.0, 0.0]), w.Omega)
    assert g[0] == 0.0


def test_pinned_wheel_may_still_brake():
    w = _wheels(Omega0=[100.0, 0.0, 0.0])
    g, _ = w.allocate(np.array([+4e-3, 0.0, 0.0]), w.Omega)   # g_1 negative: braking
    assert g[0] == pytest.approx(-4e-3)


def test_unpinned_wheels_unaffected():
    w = _wheels(Omega0=[100.0, 0.0, 0.0])
    g, _ = w.allocate(np.array([0.0, -4e-3, 0.0]), w.Omega)
    assert g[1] == pytest.approx(4e-3)


def test_omitting_omega_reproduces_torque_clip_only():
    """Standalone/legacy callers that don't track speeds keep the old
    behaviour -- and the propagator is the one required to pass Omega."""
    w = _wheels(Omega0=[100.0, 0.0, 0.0])
    g, _ = w.allocate(np.array([-4e-3, 0.0, 0.0]))
    assert g[0] == pytest.approx(4e-3)


def test_step_holds_speed_at_the_limit():
    """Driving a wheel at its limit for many steps must not integrate past
    it (beyond one step's worth of acceleration)."""
    w = _wheels(Omega0=[99.9, 0.0, 0.0])
    for _ in range(200):
        w.step(np.array([-4e-3, 0.0, 0.0]), dt=0.01)
    one_step_overshoot = (4e-3 / w.Iw[0]) * 0.01
    assert w.Omega[0] <= 100.0 + one_step_overshoot + 1e-9
