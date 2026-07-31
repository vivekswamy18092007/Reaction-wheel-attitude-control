import numpy as np
import pytest

from deimos.dynamics.rigid_body import RotationalDynamics


def test_omega_dot_zero_torque_and_symmetric_case_is_zero():
    # omega parallel to a principal axis with h_w=0: omega x J*omega = 0
    # exactly, so omega_dot must be zero with no external torque.
    J = np.diag([1.4010e-2, 1.4130e-2, 2.9380e-3])
    dyn = RotationalDynamics(J)
    omega = np.array([0.0, 0.0, 2.5])  # pure spin about the 3rd principal axis
    omega_dot = dyn.omega_dot(omega, torque=np.zeros(3))
    assert np.allclose(omega_dot, 0.0, atol=1e-12)


def test_omega_dot_matches_euler_equation_by_hand():
    J = np.diag([2.0, 3.0, 4.0])
    dyn = RotationalDynamics(J)
    omega = np.array([1.0, 1.0, 1.0])
    torque = np.array([0.5, -0.2, 0.1])
    h_w = np.array([0.1, 0.0, -0.05])

    H = J @ omega + h_w
    expected = np.linalg.inv(J) @ (torque - np.cross(omega, H))
    assert np.allclose(dyn.omega_dot(omega, torque, h_w), expected)


def test_omega_dot_defaults_h_w_to_zero():
    J = np.diag([2.0, 3.0, 4.0])
    dyn = RotationalDynamics(J)
    omega = np.array([1.0, -1.0, 0.5])
    torque = np.array([0.1, 0.2, 0.3])
    assert np.allclose(dyn.omega_dot(omega, torque), dyn.omega_dot(omega, torque, np.zeros(3)))
