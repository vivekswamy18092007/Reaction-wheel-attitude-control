import numpy as np
import pytest

from deimos.control.pd import PDController
from deimos.control.pid import PIDController
from deimos.control.wie import WieRegulator
from deimos.control.lqr import LQRController
from deimos.control import registry
from deimos.sim.config import ControllerConfig

J = np.diag([1.4010e-2, 1.4130e-2, 2.9380e-3])


def test_pd_design_matches_wie_eigenaxis_case():
    Kp, Kd = PDController.design(J, zeta=1.0, settling_time=8.0)
    K, D, mu = WieRegulator.design(J, "eigenaxis", zeta=1.0, settling_time=8.0)
    assert np.allclose(Kp, K)
    assert np.allclose(Kd, D)
    assert mu == 1.0


def test_pd_control_torque_opposes_error():
    Kp = np.diag([0.01, 0.01, 0.01])
    Kd = np.diag([0.02, 0.02, 0.02])
    pd = PDController(Kp=Kp, Kd=Kd)
    q = np.array([0.9, 0.1, 0.2, 0.3])
    q = q / np.linalg.norm(q)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    u = pd.compute_torque(q, omega=np.zeros(3), q_target=q_target)
    # error vector part points the same direction as q's vector part when
    # q_target is identity, so the restoring torque must oppose it.
    assert np.dot(u, q[1:]) < 0


def test_pd_reset_clears_integral_state():
    pd = PDController(Kp=np.eye(3), Kd=np.eye(3), Ki=np.array([0.1, 0.1, 0.1]))
    q = np.array([0.9, 0.1, 0.2, 0.3])
    q = q / np.linalg.norm(q)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    pd.compute_torque(q, omega=np.zeros(3), q_target=q_target, dt=0.01)
    assert not np.allclose(pd._z, 0.0)
    pd.reset()
    assert np.allclose(pd._z, 0.0)


def test_pid_requires_ki():
    with pytest.raises(ValueError):
        PIDController(Kp=np.eye(3), Kd=np.eye(3), Ki=None)


def _wind_up(controller, steps=4000, dt=0.01):
    """Hold a large constant attitude error so the accumulator integrates a
    full slew's worth of error, the situation that produced limit cycles."""
    q = np.array([0.9, 0.3, 0.2, 0.1])
    q /= np.linalg.norm(q)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(steps):
        u = controller.compute_torque(q, omega=np.zeros(3), q_target=q_target, dt=dt)
    return u


def test_pid_without_anti_windup_integral_runs_away():
    # Documents the failure the anti-windup option exists to fix: unbounded,
    # the integral term alone dwarfs the 4e-3 N m wheel limit.
    pid = PIDController(Kp=np.diag([0.016] * 3), Kd=np.diag([0.025] * 3),
                         Ki=np.full(3, 1.6e-3))
    u = _wind_up(pid)
    assert np.abs(u).max() > 4e-3


def test_pid_anti_windup_caps_the_integral_contribution():
    limit = 1.0e-3
    pid = PIDController(Kp=np.zeros((3, 3)), Kd=np.zeros((3, 3)),
                         Ki=np.full(3, 1.6e-3), integral_limit=limit)
    u = _wind_up(pid)
    # Kp and Kd are zero here, so u IS the integral term.
    assert np.abs(u).max() <= limit + 1e-12


def test_pid_anti_windup_also_clamps_the_accumulator():
    # Capping only the output would let _z grow without bound and take
    # arbitrarily long to unwind -- the classic windup failure.
    limit, Ki = 1.0e-3, 1.6e-3
    pid = PIDController(Kp=np.eye(3), Kd=np.eye(3), Ki=np.full(3, Ki),
                         integral_limit=limit)
    _wind_up(pid)
    assert np.all(np.abs(pid._z) <= limit / Ki + 1e-12)


def test_pid_anti_windup_defaults_off():
    pid = PIDController(Kp=np.eye(3), Kd=np.eye(3), Ki=np.full(3, 1e-4))
    assert pid.integral_limit is None


def test_pd_anti_windup_applies_to_its_optional_ki():
    limit = 1.0e-3
    pd = PDController(Kp=np.zeros((3, 3)), Kd=np.zeros((3, 3)),
                       Ki=np.full(3, 1.6e-3), integral_limit=limit)
    u = _wind_up(pd)
    assert np.abs(u).max() <= limit + 1e-12


def test_anti_windup_ignores_zero_ki_axes_without_dividing_by_zero():
    pd = PDController(Kp=np.zeros((3, 3)), Kd=np.zeros((3, 3)),
                       Ki=np.array([0.0, 1.6e-3, 0.0]), integral_limit=1e-3)
    u = _wind_up(pd)
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(pd._z))


def test_wie_sign_freezes_at_first_call():
    K, D, mu = WieRegulator.design(J, "eigenaxis", zeta=1.0, settling_time=8.0)
    wie = WieRegulator(J, K, D, mu)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    q_negative_scalar = np.array([-0.9, 0.1, 0.2, 0.3])
    q_negative_scalar /= np.linalg.norm(q_negative_scalar)

    wie.compute_torque(q_negative_scalar, omega=np.zeros(3), q_target=q_target)
    first_sign = wie._sign
    assert first_sign == -1.0

    # Even if q0 later flips positive, the sign stays latched (Remark 5).
    q_positive_scalar = np.array([0.9, 0.1, 0.2, 0.3])
    q_positive_scalar /= np.linalg.norm(q_positive_scalar)
    wie.compute_torque(q_positive_scalar, omega=np.zeros(3), q_target=q_target)
    assert wie._sign == first_sign


def test_wie_u_max_saturates():
    K, D, mu = WieRegulator.design(J, "eigenaxis", zeta=1.0, settling_time=1.0)
    wie = WieRegulator(J, K, D, mu, u_max=1e-6)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    u = wie.compute_torque(q, omega=np.zeros(3), q_target=q_target)
    assert np.all(np.abs(u) <= 1e-6 + 1e-15)


def test_lqr_is_not_implemented():
    with pytest.raises(NotImplementedError):
        LQRController()


def test_registry_dispatches_pd_pid_wie():
    pd_cfg = ControllerConfig(type="PD", Kp=np.eye(3), Kd=np.eye(3))
    assert isinstance(registry.build(pd_cfg, J), PDController)

    pid_cfg = ControllerConfig(type="PID", Kp=np.eye(3), Kd=np.eye(3), Ki=np.array([0.1, 0.1, 0.1]))
    assert isinstance(registry.build(pid_cfg, J), PIDController)

    wie_cfg = ControllerConfig(type="wie", case="eigenaxis", zeta=1.0, settling_time_s=8.0)
    assert isinstance(registry.build(wie_cfg, J), WieRegulator)


def test_registry_unknown_type_raises():
    bad_cfg = ControllerConfig(type="not-a-controller")
    with pytest.raises(ValueError):
        registry.build(bad_cfg, J)
