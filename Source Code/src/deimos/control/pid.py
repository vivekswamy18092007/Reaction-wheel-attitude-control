"""
Quaternion PID regulator -- a distinct, selectable controller.type ("PID")
alongside pd.py's "PD" and wie.py's "wie".

    u = -Kp*q_ev - Kd*omega - Ki*integral(q_ev dt)

Unlike pd.py (where Ki is optional and defaults off), this class is meant to
be used WITH integral action -- Ki is required at construction. If you don't
want integral action, use PDController from pd.py ("type: PD") instead of
leaving Ki unset here.

Gains are set directly (Kp, Kd, Ki) -- no design()/zeta/settling_time
helper here by choice. Same quaternion-error convention as pd.py:
q_e = q_target^-1 (x) q, scalar-first.

Anti-windup is OPT-IN via `integral_limit` and defaults to off, preserving
the original unbounded behaviour for existing presets. Leaving it off on a
large slew is a real failure mode, not a theoretical one: the accumulator
integrates the whole transient, and at Ki/Kp = 0.1 the integral term reaches
several times the wheel torque limit and drives a sustained limit cycle.
Set `integral_limit` (see below) if you use meaningful integral action.
"""

import numpy as np

from deimos.math.quaternion import quaternion_error


class PIDController:

    def __init__(self, Kp, Kd, Ki, u_max=None, integral_limit=None):
        """
        integral_limit: [N m] per-axis cap on the *contribution* of the
            integral term, |Ki * z|. None (default) = unbounded.

            Capping the contribution rather than the raw accumulator keeps
            the limit physically meaningful and independent of Ki's
            magnitude -- you set it as a fraction of the actuator's torque
            budget ("integral action may claim at most 25% of the wheel
            limit"), and it means the same thing for every gain set a
            search might try. The accumulator itself is clamped to match,
            because capping only the output would still let z grow without
            bound and then take arbitrarily long to unwind, which is the
            classic windup failure.
        """
        if Ki is None:
            raise ValueError(
                "PIDController requires Ki -- if you don't want integral "
                "action, use PDController from control/pd.py instead "
                "(controller.type: \"PD\")."
            )
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = np.asarray(Ki, dtype=float)
        self.u_max = u_max
        self.integral_limit = integral_limit
        self._z = np.zeros(3)

    def _accumulate(self, q_ev, dt):
        """Advance the integral state, clamped if anti-windup is enabled."""
        self._z = self._z + q_ev * dt
        if self.integral_limit is None:
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            z_max = np.where(np.abs(self.Ki) > 0.0,
                              self.integral_limit / np.abs(self.Ki), np.inf)
        self._z = np.clip(self._z, -z_max, z_max)

    def compute_torque(self, q, omega, q_target, dt=None):
        q_e = quaternion_error(q, q_target)
        q_ev = q_e[1:]
        u = -self.Kp @ q_ev - self.Kd @ omega

        if dt is not None:
            # dt must match the propagator's step, or the integral term
            # silently does nothing -- no error is raised either way.
            integral = self.Ki * self._z
            if self.integral_limit is not None:
                integral = np.clip(integral, -self.integral_limit, self.integral_limit)
            u = u - integral
            self._accumulate(q_ev, dt)

        if self.u_max is not None:
            u = np.clip(u, -self.u_max, self.u_max)

        return u

    def reset(self):
        """Call before each new simulation run. The integral accumulator
        persists across calls otherwise, so reusing one instance across
        multiple runs without resetting carries state across runs
        incorrectly."""
        self._z = np.zeros(3)
