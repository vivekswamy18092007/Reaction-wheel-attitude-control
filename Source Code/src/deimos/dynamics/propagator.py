"""
RK4 propagator for the coupled body + reaction-wheel (+ magnetorquer) system.

Zero-order holds, and why they are safe:
  * The wheel motor torque g is allocated ONCE per step and held across the
    4 RK4 sub-stages -- the control loop runs at the step rate, not the
    sub-stage rate, mirroring a discrete flight computer.
  * Speed saturation (allocate(..., Omega)) is evaluated at the step-start
    Omega. A wheel can therefore overshoot max_speed by at most one step's
    worth of acceleration (g/Iw * dt ~ 7 rad/s at the torque cap) -- 0.5% of
    the 1256 rad/s limit, and the next step zeroes the accelerating torque.
  * The magnetorquer dipole m and the field B are also held per step: B
    rotates at twice orbit rate (~2e-3 rad/s), so it is constant to ~1e-5
    over even the desaturation scenario's 0.1 s steps.
"""

import numpy as np

from deimos.math.kinematics import QuaternionKinematics
from deimos.dynamics.rigid_body import RotationalDynamics


class AttitudeSimulator:
    """RK4 propagator. See module docstring for the ZOH contract.

    mtq / B_func: optional magnetorquer desaturation. When both are given,
    each step computes B = B_func(t, q), m = mtq.desat_dipole(h_w, B) and
    applies tau_mtq = m x B as an external torque on the body. The attitude
    controller then counters it with the wheels, which is precisely what
    dumps their momentum (see actuators/magnetorquers.py). m and tau_mtq are
    logged every step -- zeros when desaturation is off -- so SimResults has
    a uniform shape either way.
    """

    def __init__(self, inertia_tensor, q0, omega0, wheel_array, dt=0.01,
                 mtq=None, B_func=None):
        self.kinematics = QuaternionKinematics()
        self.dynamics = RotationalDynamics(inertia_tensor)
        self.wheels = wheel_array
        self.dt = dt

        if (mtq is None) != (B_func is None):
            raise ValueError(
                "mtq and B_func come as a pair: the desat law needs the field "
                "and the field is useless without the law. Pass both or neither.")
        self.mtq = mtq
        self.B_func = B_func

        self.q = np.array(q0, dtype=np.float64)
        self.omega = np.array(omega0, dtype=np.float64)
        self.Omega = self.wheels.Omega.copy()

        self.t = 0.0
        self.history = {"t": [], "q": [], "omega": [], "Omega": [], "u": [],
                        "g": [], "m": [], "tau_mtq": []}
        self._log(u=np.zeros(3), g=np.zeros(self.wheels.N),
                  m=np.zeros(3), tau_mtq=np.zeros(3))

    def _state_derivative(self, q, omega, Omega, g, tau_ext):
        q_dot = self.kinematics.quaternion_dot(q, omega)
        h_w = self.wheels.momentum(Omega)
        tau_body = self.wheels.body_torque(g)
        omega_dot = self.dynamics.omega_dot(omega, tau_ext + tau_body, h_w)
        Omega_dot = self.wheels.wheel_accel(g)
        return q_dot, omega_dot, Omega_dot

    def step(self, u_cmd=None, tau_ext=None):
        dt = self.dt
        q, omega, Omega = self.q, self.omega, self.Omega

        if u_cmd is None:
            u_cmd = np.zeros(3)
        if tau_ext is None:
            tau_ext = np.zeros(3)

        # Magnetorquer desaturation: dipole from the step-start state, torque
        # held ZOH like everything else the flight computer commands.
        m = np.zeros(3)
        tau_mtq = np.zeros(3)
        if self.mtq is not None:
            B_body = self.B_func(self.t, q)
            m = self.mtq.desat_dipole(self.wheels.momentum(Omega), B_body)
            tau_mtq = self.mtq.torque(m, B_body)
        tau_ext = np.asarray(tau_ext, dtype=np.float64) + tau_mtq

        # Omega passed so speed saturation is enforced in the plant, not just
        # reported after the fact -- see ReactionWheelArray.allocate().
        g, torque_saturated = self.wheels.allocate(u_cmd, Omega)

        k1_q, k1_w, k1_O = self._state_derivative(q, omega, Omega, g, tau_ext)
        k2_q, k2_w, k2_O = self._state_derivative(
            q + 0.5*dt*k1_q, omega + 0.5*dt*k1_w, Omega + 0.5*dt*k1_O, g, tau_ext)
        k3_q, k3_w, k3_O = self._state_derivative(
            q + 0.5*dt*k2_q, omega + 0.5*dt*k2_w, Omega + 0.5*dt*k2_O, g, tau_ext)
        k4_q, k4_w, k4_O = self._state_derivative(
            q + dt*k3_q, omega + dt*k3_w, Omega + dt*k3_O, g, tau_ext)

        q_new = q + (dt/6.0)*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        omega_new = omega + (dt/6.0)*(k1_w + 2*k2_w + 2*k3_w + k4_w)
        Omega_new = Omega + (dt/6.0)*(k1_O + 2*k2_O + 2*k3_O + k4_O)

        q_new = q_new / np.linalg.norm(q_new)

        self.q = q_new
        self.omega = omega_new
        self.Omega = Omega_new
        self.wheels.Omega = Omega_new
        self.t += dt
        self._log(u=u_cmd, g=g, m=m, tau_mtq=tau_mtq)

    def _log(self, u, g, m, tau_mtq):
        self.history["t"].append(self.t)
        self.history["q"].append(self.q.copy())
        self.history["omega"].append(self.omega.copy())
        self.history["Omega"].append(self.Omega.copy())
        self.history["u"].append(np.asarray(u).copy())
        self.history["g"].append(np.asarray(g).copy())
        self.history["m"].append(np.asarray(m).copy())
        self.history["tau_mtq"].append(np.asarray(tau_mtq).copy())

    def run(self, duration, u_func=None, tau_ext_func=None):
        steps = int(duration / self.dt)
        for _ in range(steps):
            u_cmd = u_func(self.t, self.q, self.omega, self.Omega) if u_func else np.zeros(3)
            tau_ext = tau_ext_func(self.t, self.q, self.omega) if tau_ext_func else np.zeros(3)
            self.step(u_cmd, tau_ext)
