import numpy as np
import pytest

from deimos.math.quaternion import Quaternion, euler_to_quaternion, quaternion_to_euler, quaternion_error


def test_identity_quaternion_norm_is_one():
    q = Quaternion(1, 0, 0, 0)
    assert q.norm() == pytest.approx(1.0)


def test_normalize_scales_to_unit_norm():
    q = Quaternion(2, 0, 0, 0).normalize()
    assert q.norm() == pytest.approx(1.0)


def test_conjugate_negates_vector_part():
    q = Quaternion(0.5, 0.1, 0.2, 0.3)
    qc = q.conjugate()
    assert qc.q[0] == pytest.approx(q.q[0])
    assert np.allclose(qc.q[1:], -q.q[1:])


def test_quaternion_times_inverse_is_identity():
    q = euler_to_quaternion(np.radians(30), np.radians(-15), np.radians(80)).normalize()
    q_inv = q.inverse()
    result = (q * q_inv).q
    assert np.allclose(result, [1, 0, 0, 0], atol=1e-10)


def test_rotation_matrix_is_orthonormal():
    q = euler_to_quaternion(np.radians(40), np.radians(20), np.radians(-10)).normalize()
    R = q.to_rotation_matrix()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_euler_round_trip():
    roll, pitch, yaw = np.radians(25), np.radians(-40), np.radians(133)
    q = euler_to_quaternion(roll, pitch, yaw)
    r2, p2, y2 = quaternion_to_euler(q)
    assert r2 == pytest.approx(roll, abs=1e-9)
    assert p2 == pytest.approx(pitch, abs=1e-9)
    assert y2 == pytest.approx(yaw, abs=1e-9)


def test_error_quaternion_of_identical_quaternions_is_identity():
    q = euler_to_quaternion(np.radians(12), np.radians(-8), np.radians(4)).normalize()
    q_err = q.error_quaternion(q)
    assert q_err.q[0] == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(q_err.q[1:], 0.0, atol=1e-10)


def test_array_quaternion_error_matches_object_form():
    q0 = euler_to_quaternion(np.radians(55), np.radians(65), np.radians(15)).normalize()
    qt = euler_to_quaternion(np.radians(10), np.radians(-5), np.radians(20)).normalize()
    array_form = quaternion_error(q0.q, qt.q)
    object_form = q0.error_quaternion(qt).q
    assert np.allclose(array_form, object_form, atol=1e-12)
