import numpy as np


class QuaternionKinematics:

    def quaternion_dot(self, q, omega):

        """
        Quaternion Propogation Equation:

        q_dot = 0.5 * Omega(omega) * q
        where Omega(omega) is the skew-symmetric matrix of angular velocity omega:

        Omega(omega) =

                        [  0   -wx  -wy  -wz ]
                        [ wx     0   wz  -wy ]
                        [ wy   -wz    0   wx ]
                        [ wz    wy  -wx    0 ]


        """

        

        wx = omega[0]
        wy = omega[1]
        wz = omega[2]

        

        Omega = np.array([
            [0,   -wx,  -wy,  -wz],
            [wx,   0,    wz,  -wy],
            [wy,  -wz,   0,    wx],
            [wz,   wy,  -wx,   0 ]
        ])

        

        q_dot = 0.5 * (Omega @ q)

        return q_dot
