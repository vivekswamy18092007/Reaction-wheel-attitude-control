import numpy as np
from quaternion import Quaternion


class PIDController:

    def __init__(self, Kp, Kd, Ki=None):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = np.asarray(Ki, dtype=float) if Ki is not None else None
        self._z = np.zeros(3)

    def control_torque(self, q, q_target, omega, dt=None):
        q_error = q.error_quaternion(q_target)
        q_ev = q_error.vector_part()
        u = -self.Kp @ q_ev - self.Kd @ omega

        if self.Ki is not None and dt is not None:
            u = u - self.Ki * self._z
            self._z = self._z + q_ev * dt

        return u

    def reset(self):
        """Call before each new simulation run. the integral accumulator
        persists across calls otherwise, so reusing one instance across
        multiple runs without resetting carries state across runs
        incorrectly)."""
        self._z = np.zeros(3)

   
