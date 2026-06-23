import numpy as np
from QuaternionKinematics import QuaternionKinematics
from RotationalDynamics import RotationalDynamics



class AttitudeSimulator:
    def __init__(self, inertia_tensor, q0, omega0, dt=0.01):
        self.kinematics = QuaternionKinematics()
        self.dynamics = RotationalDynamics(inertia_tensor)
        self.dt = dt

        self.q = np.array(q0, dtype=np.float64)        
        self.omega = np.array(omega0, dtype=np.float64) 

        self.t = 0.0
        self.history = {"t": [], "q": [], "omega": []}
        self._log()

    def _state_derivative(self, q, omega, torque):
        q_dot = self.kinematics.quaternion_dot(q, omega)
        omega_dot = self.dynamics.omega_dot(omega, torque)
        return q_dot, omega_dot

    def step(self, torque=np.zeros(3)):
        dt = self.dt
        q, omega = self.q, self.omega

        # RK4 implementation
        k1_q, k1_w = self._state_derivative(q, omega, torque)
        k2_q, k2_w = self._state_derivative(q + 0.5*dt*k1_q, omega + 0.5*dt*k1_w, torque)
        k3_q, k3_w = self._state_derivative(q + 0.5*dt*k2_q, omega + 0.5*dt*k2_w, torque)
        k4_q, k4_w = self._state_derivative(q + dt*k3_q, omega + dt*k3_w, torque)

        q_new = q + (dt/6.0)*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        omega_new = omega + (dt/6.0)*(k1_w + 2*k2_w + 2*k3_w + k4_w)

        # re-normalize quaternion every step 
        q_new = q_new / np.linalg.norm(q_new)

        self.q = q_new
        self.omega = omega_new
        self.t += dt
        self._log()

    def _log(self):
        self.history["t"].append(self.t)
        self.history["q"].append(self.q.copy())
        self.history["omega"].append(self.omega.copy())

    def run(self, duration, torque_func=None):
        steps = int(duration / self.dt)
        for _ in range(steps):
            torque = torque_func(self.t, self.q, self.omega) if torque_func else np.zeros(3)
            self.step(torque)


