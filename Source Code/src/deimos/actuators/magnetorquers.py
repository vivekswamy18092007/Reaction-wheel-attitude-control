"""
Magnetorquer model + cross-product desaturation law.

Role in this system: MOMENTUM DUMPING ONLY. The reaction wheels do all
attitude control; the magnetorquers exist to bleed the momentum the wheels
accumulate (from disturbance torques or a pre-loaded state) before they pin
at max speed and lose control authority. This is the competition brief's
"recover from wheel saturation using an additional actuator type" bonus.

Physics
-------
A magnetorquer produces a dipole moment m [A m^2]; in the geomagnetic field
B [T] the torque on the spacecraft is

    tau_mtq = m x B                                  (external torque)

Only the component of any desired torque perpendicular to B is reachable --
torque along B is impossible (m x B is always perpendicular to B). The
cross-product desaturation law embraces that limit:

    m = (k / |B|^2) * (h_w x B)

which gives (bac-cab expansion)

    tau_mtq = m x B = -k * (h_w - B_hat (B_hat . h_w)) = -k * h_w_perp

i.e. the perpendicular component of the wheel momentum decays exponentially
with time constant 1/k. The parallel component is untouchable *now*, but the
dipole-model field rotates at twice orbit rate (dynamics/environment.py), so
"parallel to B" is a different direction a quarter-orbit later and the whole
vector drains over an orbit.

Mechanism of the dump: tau_mtq acts on the BODY. The attitude controller
holds the body still by commanding the wheels to counter it, and that
counter-torque is what de-spins the wheels -- momentum leaves the wheels,
through the body, into the field. No wheel-side logic is needed.

Engagement logic: a start/stop hysteresis latch rather than a bare
threshold, so the law does not chatter on/off when |h_w| hovers at the
boundary. Stateful -> reset() before reuse, same contract as the
controllers.

References: Markley & Crassidis, Fundamentals of Spacecraft Attitude
Determination and Control, Sec. 7.3 (momentum management); Sidi, Spacecraft
Dynamics and Control, Ch. 7.
"""

from __future__ import annotations

import numpy as np

from deimos import constants


class MagnetorquerArray:
    """Three orthogonal torquer rods aligned with the body axes.

    max_dipole  : [A m^2] per-axis dipole limit (hardware ceiling, from
                  constants unless a config overrides it for a what-if study)
    k_desat     : [1/s] dump rate -- 1/k is the decay time constant of the
                  perpendicular wheel-momentum component, before dipole
                  clipping stretches it
    threshold   : [N m s] |h_w| at which desaturation engages
    stop_frac   : desaturation disengages when |h_w| falls below
                  stop_frac * threshold (hysteresis)
    """

    def __init__(self,
                 max_dipole: float = constants.MTQ_MAX_DIPOLE,
                 k_desat: float = 5.0e-3,
                 threshold: float = 0.3 * constants.WHEEL_MOMENTUM_CAPACITY,
                 stop_frac: float = 0.5):
        self.max_dipole = float(max_dipole)
        self.k_desat = float(k_desat)
        self.threshold = float(threshold)
        self.stop_frac = float(stop_frac)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def desat_dipole(self, h_w: np.ndarray, B_body: np.ndarray) -> np.ndarray:
        """Commanded dipole moment [A m^2] for the current wheel momentum and
        body-frame field. Zero while the latch is disengaged."""
        h_w = np.asarray(h_w, dtype=float)
        B_body = np.asarray(B_body, dtype=float)

        h = np.linalg.norm(h_w)
        if not self._active and h >= self.threshold:
            self._active = True
        elif self._active and h <= self.stop_frac * self.threshold:
            self._active = False

        if not self._active:
            return np.zeros(3)

        B2 = float(B_body @ B_body)
        if B2 <= 0.0:
            return np.zeros(3)

        m = (self.k_desat / B2) * np.cross(h_w, B_body)
        return np.clip(m, -self.max_dipole, self.max_dipole)

    @staticmethod
    def torque(m: np.ndarray, B_body: np.ndarray) -> np.ndarray:
        """Torque on the body [N m] from dipole m in field B_body."""
        return np.cross(m, B_body)

    def reset(self):
        """Clear the engagement latch. Call before each new simulation run --
        same reuse contract as the controllers' reset()."""
        self._active = False

    def __repr__(self):
        return (f"MagnetorquerArray(max_dipole={self.max_dipole:.2e} A m^2, "
                f"k_desat={self.k_desat:.2e} 1/s, "
                f"threshold={self.threshold:.2e} N m s)")
