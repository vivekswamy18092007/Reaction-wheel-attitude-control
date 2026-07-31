import numpy as np

from deimos.math.quaternion import quaternion_error


class PDController:

    def __init__(self, Kp, Kd, Ki=None, u_max=None, integral_limit=None):
        """integral_limit: [N m] per-axis cap on the integral term's
        contribution when the optional Ki is in use. None = unbounded.
        See control/pid.py for why the cap is on the contribution rather
        than on the raw accumulator."""
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = np.asarray(Ki, dtype=float) if Ki is not None else None
        self.u_max = u_max
        self.integral_limit = integral_limit
        self._z = np.zeros(3)

    @staticmethod
    def design(J, zeta, settling_time):
        """
        Kp = k*J, Kd = d*J from the same large-angle settling-time relation
        WieRegulator.design() uses (omega_n = 8/(zeta*t_s), k = 2*omega_n^2,
        d = 2*zeta*omega_n). This lands on exactly Wie's "eigenaxis" case
        gains: with mu=1 cancelling the gyroscopic term, Wie's law reduces to
        u = -D*omega - K*q_ev, the same structure as PD's u = -Kp*q_ev - Kd*omega.
        """
        J = np.asarray(J, dtype=float)
        omega_n = 8.0 / (zeta * settling_time)
        k = 2.0 * omega_n**2
        d = 2.0 * zeta * omega_n
        return k * J, d * J

    def compute_torque(self, q, omega, q_target, dt=None):
        q_e = quaternion_error(q, q_target)
        q_ev = q_e[1:]
        u = -self.Kp @ q_ev - self.Kd @ omega

        if self.Ki is not None and dt is not None:
            integral = self.Ki * self._z
            if self.integral_limit is not None:
                integral = np.clip(integral, -self.integral_limit, self.integral_limit)
            u = u - integral
            self._z = self._z + q_ev * dt
            if self.integral_limit is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_max = np.where(np.abs(self.Ki) > 0.0,
                                      self.integral_limit / np.abs(self.Ki), np.inf)
                self._z = np.clip(self._z, -z_max, z_max)

        if self.u_max is not None:
            u = np.clip(u, -self.u_max, self.u_max)

        return u

    def reset(self):
        """Call before each new simulation run. the integral accumulator
        persists across calls otherwise, so reusing one instance across
        multiple runs without resetting carries state across runs
        incorrectly)."""
        self._z = np.zeros(3)
