import numpy as np


class RotationalDynamics:
    def __init__(self, inertia_tensor):
        self.inertia_tensor = np.array(inertia_tensor, dtype=np.float64)
        self.inertia_tensor_inv = np.linalg.inv(self.inertia_tensor)

    def omega_dot(self, omega, torque, h_w=None):
        # Euler's Equation with wheel momentum coupling:
        # J*omega_dot = torque - omega x (J*omega + h_w)
        if h_w is None:
            h_w = np.zeros(3)

        H = self.inertia_tensor @ omega + h_w
        omega_dot = self.inertia_tensor_inv @ (torque - np.cross(omega, H))

        return omega_dot
