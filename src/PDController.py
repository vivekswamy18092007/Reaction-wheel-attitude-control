import numpy as np


class PDController:

    def __init__(self, Kp, Kd):
        self.Kp = Kp
        self.Kd = Kd

    def control_torque(self, q, q_target, omega):
        q_error = q.error_quaternion(q_target)
        u = -self.Kp @ q_error.vector_part() - self.Kd @ omega
        return u
