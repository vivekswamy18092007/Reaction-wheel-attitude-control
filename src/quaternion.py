import numpy as np


class Quaternion:

    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z], dtype=float)

    def __repr__(self):
        w, x, y, z = self.q

        return (
            f"Quaternion(\n"
            f"  w={w:.6f}\n"
            f"  x={x:.6f}\n"
            f"  y={y:.6f}\n"
            f"  z={z:.6f}\n"
            f")"
        )
    
    # Quaternion magnitude 

    def norm(self):
        return np.linalg.norm(self.q)
    
    # Quaternion normalization
    
    def normalize(self):

        n = self.norm()

        if n == 0:
            raise ValueError("Cannot normalize a zero quaternion")

        self.q = self.q / n

        return self
    
    # Quaternion conjugate
    
    def conjugate(self):

        w, x, y, z = self.q

        return Quaternion(
            w,
            -x,
            -y,
            -z
        )
    
    # Quaternion inversion


    def inverse(self):

        n2 = self.norm()**2

        if n2 == 0:
            raise ValueError("Cannot invert a zero quaternion")

        q_conj = self.conjugate()

        return Quaternion(
            *(q_conj.q / n2)
        )
    
    # Quaternion multiplication

    def __mul__(self, other):

        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q

        w = (
            w1*w2
            - x1*x2
            - y1*y2
            - z1*z2
        )

        x = (
            w1*x2
            + x1*w2
            + y1*z2
            - z1*y2
        )

        y = (
            w1*y2
            - x1*z2
            + y1*w2
            + z1*x2
        )

        z = (
            w1*z2
            + x1*y2
            - y1*x2
            + z1*w2
        )

        return Quaternion(w, x, y, z)
    
# Euler angles to Quaternion conversion


import numpy as np

def euler_to_quaternion(roll, pitch, yaw):

    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)

    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)

    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy

    x = sr * cp * cy - cr * sp * sy

    y = cr * sp * cy + sr * cp * sy

    z = cr * cp * sy - sr * sp * cy

    return Quaternion(w, x, y, z)

# Quaternion to Euler angles conversion

def quaternion_to_euler(q):

    w, x, y, z = q.q

    roll = np.arctan2(
        2*(w*x + y*z),
        1 - 2*(x*x + y*y)
    )

    pitch = np.arcsin(
        2*(w*y - z*x)
    )

    yaw = np.arctan2(
        2*(w*z + x*y),
        1 - 2*(y*y + z*z)
    )

    return roll, pitch, yaw
    
roll = np.radians(30)
pitch = np.radians(20)
yaw = np.radians(10)

q = euler_to_quaternion(
    roll,
    pitch,
    yaw
)

# Quaternion to Rotation matrix conversion

def to_rotation_matrix(self):

    w, x, y, z = self.q

    R = np.array([

        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w)
        ],

        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w)
        ],

        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y)
        ]

    ])

    return R

