import numpy as np
import pytest

from deimos.actuators.reaction_wheels import ReactionWheelArray


def test_cone_axes_are_mutually_orthogonal():
    W = ReactionWheelArray.cone_axes()
    assert np.allclose(W.T @ W, np.eye(3), atol=1e-9)


def test_orthogonal_axes_is_identity():
    assert np.allclose(ReactionWheelArray.orthogonal_axes(), np.eye(3))


def test_bad_config_raises():
    with pytest.raises(ValueError):
        ReactionWheelArray(config="not-a-real-config")


def test_pyramid_config_allows_four_redundant_wheels():
    # pyramid is intentionally redundant/non-orthogonal (4 wheels for 3 axes);
    # requesting it under config="pyramid" must NOT trigger the cone/orthogonal
    # sanity check (it would always fail for 4 columns).
    rw = ReactionWheelArray(config="pyramid", tilt_deg=32.0)
    assert rw.N == 4


def test_allocate_reproduces_commanded_torque_when_unsaturated():
    rw = ReactionWheelArray(config="cone")
    tau_cmd = np.array([1.0e-4, -5.0e-5, 2.0e-5])
    g, saturated = rw.allocate(tau_cmd)
    assert not np.any(saturated)
    assert np.allclose(rw.body_torque(g), tau_cmd, atol=1e-12)


def test_allocate_clips_to_max_torque():
    rw = ReactionWheelArray(config="cone", max_torque=1.0e-3)
    huge_tau = np.array([1.0, 0.0, 0.0])
    g, saturated = rw.allocate(huge_tau)
    assert np.any(saturated)
    assert np.all(np.abs(g) <= rw.max_torque + 1e-15)


def test_momentum_and_kinetic_energy_scale_with_speed():
    rw = ReactionWheelArray(config="cone", wheel_inertia=1e-5, Omega0=[0.0, 0.0, 0.0])
    h0 = rw.momentum()
    ke0 = rw.kinetic_energy()
    assert np.allclose(h0, 0.0)
    assert ke0 == pytest.approx(0.0)

    rw2 = ReactionWheelArray(config="cone", wheel_inertia=1e-5, Omega0=[10.0, 10.0, 10.0])
    assert rw2.kinetic_energy() == pytest.approx(0.5 * 1e-5 * 3 * 10.0**2)
