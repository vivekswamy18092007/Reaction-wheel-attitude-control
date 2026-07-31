import numpy as np
import pytest

from deimos.math.kinematics import QuaternionKinematics


def test_quaternion_dot_zero_omega_is_zero():
    k = QuaternionKinematics()
    q_dot = k.quaternion_dot(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert np.allclose(q_dot, 0.0)


def test_quaternion_dot_preserves_unit_norm_to_first_order():
    # d/dt (q.q) = 2 q . q_dot; for a unit quaternion under pure kinematic
    # propagation this must be exactly zero (Omega(omega) is skew-symmetric).
    k = QuaternionKinematics()
    q = np.array([0.8, 0.1, 0.3, -0.2])
    q = q / np.linalg.norm(q)
    omega = np.array([0.5, -0.3, 0.2])
    q_dot = k.quaternion_dot(q, omega)
    assert np.dot(q, q_dot) == pytest.approx(0.0, abs=1e-12)


def test_quaternion_dot_matches_hand_derivation():
    k = QuaternionKinematics()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([1.0, 2.0, 3.0])
    # At the identity quaternion, q_dot = 0.5 * [0, wx, wy, wz]
    expected = 0.5 * np.array([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(k.quaternion_dot(q, omega), expected)
