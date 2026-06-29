import numpy as np
from QuaternionKinematics import QuaternionKinematics
from RotationalDynamics import RotationalDynamics


class AttitudeSimulator:
    def __init__(self, inertia_tensor, q0, omega0, wheel_array, dt=0.01):
        self.kinematics = QuaternionKinematics()
        self.dynamics = RotationalDynamics(inertia_tensor)
        self.wheels = wheel_array
        self.dt = dt

        self.q = np.array(q0, dtype=np.float64)
        self.omega = np.array(omega0, dtype=np.float64)
        self.Omega = self.wheels.Omega.copy()

        self.t = 0.0
        self.history = {"t": [], "q": [], "omega": [], "Omega": [], "u": [], "g": []}
        self._log(u=np.zeros(3), g=np.zeros(self.wheels.N))

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

        g, torque_saturated = self.wheels.allocate(u_cmd)

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
        self._log(u=u_cmd, g=g)

    def _log(self, u, g):
        self.history["t"].append(self.t)
        self.history["q"].append(self.q.copy())
        self.history["omega"].append(self.omega.copy())
        self.history["Omega"].append(self.Omega.copy())
        self.history["u"].append(np.asarray(u).copy())
        self.history["g"].append(np.asarray(g).copy())

    def run(self, duration, u_func=None, tau_ext_func=None):
        steps = int(duration / self.dt)
        for _ in range(steps):
            u_cmd = u_func(self.t, self.q, self.omega, self.Omega) if u_func else np.zeros(3)
            tau_ext = tau_ext_func(self.t, self.q, self.omega) if tau_ext_func else np.zeros(3)
            self.step(u_cmd, tau_ext)
