"""
results.py
==========

`SimResults` is just `sim.history` (from AttitudeSimulator) wrapped as numpy
arrays, plus a handful of derived quantities (attitude error, momentum norm,
energy) computed once so every plot function doesn't recompute them.

No new physics here — every derived quantity is a direct, undisguised
application of formulas you already have elsewhere (quaternion error angle
from AttitudeVisualizer, momentum/energy from ReactionWheelArray).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from QuaternionLibrary import Quaternion


@dataclass
class SimResults:
    name: str                  # config name, for labeling plots/filenames
    t: np.ndarray               # (T,)
    q: np.ndarray                # (T,4)  wxyz
    omega: np.ndarray            # (T,3)
    Omega: np.ndarray            # (T,N)  wheel speeds
    u: np.ndarray                 # (T,3)  commanded control torque
    g: np.ndarray                  # (T,N) per-wheel motor torque
    q_target: np.ndarray            # (4,)

    # derived, filled in by from_history()
    attitude_error_deg: np.ndarray = None   # (T,)
    wheel_momentum: np.ndarray = None        # (T,3)
    wheel_momentum_norm: np.ndarray = None    # (T,)
    body_momentum: np.ndarray = None           # (T,3)  J*omega
    total_momentum: np.ndarray = None            # (T,3)  J*omega + h_w
    total_momentum_norm: np.ndarray = None         # (T,)
    wheel_kinetic_energy: np.ndarray = None          # (T,)
    body_kinetic_energy: np.ndarray = None            # (T,)
    total_kinetic_energy: np.ndarray = None             # (T,)
    quaternion_norm_error: np.ndarray = None              # (T,) |q| - 1, pre-renormalize drift proxy

    @classmethod
    def from_history(cls, name: str, history: dict, q_target: np.ndarray,
                      inertia_tensor: np.ndarray, wheels) -> "SimResults":
        t = np.asarray(history["t"])
        q = np.asarray(history["q"])
        omega = np.asarray(history["omega"])
        Omega = np.asarray(history["Omega"])
        u = np.asarray(history["u"])
        g = np.asarray(history["g"])

        results = cls(name=name, t=t, q=q, omega=omega, Omega=Omega,
                       u=u, g=g, q_target=np.asarray(q_target, dtype=np.float64))

        # --- attitude error (deg), same formula as AttitudeVisualizer ---
        q_tgt_obj = Quaternion(*results.q_target)
        q_tgt_inv = q_tgt_obj.inverse()
        err_deg = np.empty(len(q))
        for i, qi in enumerate(q):
            q_err = q_tgt_inv * Quaternion(*qi)
            q_err.normalize()
            w_err = np.clip(abs(q_err.q[0]), -1.0, 1.0)
            err_deg[i] = np.degrees(2.0 * np.arccos(w_err))
        results.attitude_error_deg = err_deg

        # --- momentum / energy diagnostics, reusing ReactionWheelArray methods ---
        h_w = np.array([wheels.momentum(Omega_i) for Omega_i in Omega])  # (T,3)
        results.wheel_momentum = h_w
        results.wheel_momentum_norm = np.linalg.norm(h_w, axis=1)

        body_h = (inertia_tensor @ omega.T).T  # (T,3)
        results.body_momentum = body_h
        results.total_momentum = body_h + h_w
        results.total_momentum_norm = np.linalg.norm(results.total_momentum, axis=1)

        wheel_ke = np.array([wheels.kinetic_energy(Omega_i) for Omega_i in Omega])
        results.wheel_kinetic_energy = wheel_ke
        body_ke = 0.5 * np.einsum("ti,ij,tj->t", omega, inertia_tensor, omega)
        results.body_kinetic_energy = body_ke
        results.total_kinetic_energy = wheel_ke + body_ke

        results.quaternion_norm_error = np.linalg.norm(q, axis=1) - 1.0

        return results

    # --- convenience metrics, useful for printing a quick summary or for the report ---

    def settling_time(self, threshold_deg: float = 1.0) -> float | None:
        """First time after which attitude error stays below threshold_deg
        for the remainder of the run. Returns None if it never settles."""
        below = self.attitude_error_deg <= threshold_deg
        for i in range(len(below)):
            if below[i:].all():
                return float(self.t[i])
        return None

    def max_wheel_speed(self) -> float:
        return float(np.max(np.abs(self.Omega)))

    def final_attitude_error_deg(self) -> float:
        return float(self.attitude_error_deg[-1])

    def summary(self) -> str:
        st = self.settling_time()
        st_str = f"{st:.2f} s" if st is not None else "did not settle"
        return (
            f"--- {self.name} ---\n"
            f"  settling time (<1 deg): {st_str}\n"
            f"  final attitude error  : {self.final_attitude_error_deg():.4f} deg\n"
            f"  max wheel speed       : {self.max_wheel_speed():.2f} rad/s "
            f"({self.max_wheel_speed() / 104.72:.0f} rpm)\n"
            f"  max |total momentum|  : {self.total_momentum_norm.max():.6e} kg m^2/s\n"
            f"  total momentum drift  : {self.total_momentum_norm.max() - self.total_momentum_norm[0]:.3e} kg m^2/s\n"
        )
