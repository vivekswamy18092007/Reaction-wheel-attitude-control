import numpy as np


class RotationalDynamics:
    def __init__(self, inertia_tensor):

        self.inertia_tensor =np.array(inertia_tensor, dtype=np.float64)
        self.inertia_tensor_inv  = np.linalg.inv( self.inertia_tensor)

    def omega_dot( self, omega, torque):

        #Euler's Equation

        #inertia_tensor*omega_dot = torque - np.cross( omega, inertia_tensor*omega)

        H = self.inertia_tensor @ omega
        omega_dot = self.inertia_tensor_inv @ (torque - np.cross( omega, H))

        return omega_dot_value
