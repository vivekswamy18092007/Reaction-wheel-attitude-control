"""
results.py
==========

`SimResults` is just `sim.history` (from AttitudeSimulator) wrapped as numpy
arrays, plus a handful of derived quantities (attitude error, momentum norm,
energy) computed once so every plot function doesn't recompute them.

The per-run convenience metrics (settling_time, control_effort, ...) live as
methods here AND as free functions in analysis/metrics.py that just call
these methods -- the free-function form is what analysis/stability.py and
analysis/design_card.py are written against, and what a future consumer that
only has a SimResults (no import of this module) would reach for.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from deimos.math.quaternion import Quaternion


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
    m: np.ndarray = None            # (T,3) magnetorquer dipole command [A m^2]
    tau_mtq: np.ndarray = None      # (T,3) magnetorquer torque on the body [N m]

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

    # --- actuator authority / control-strategy diagnostics ---
    q_error: np.ndarray = None              # (T,4) error quaternion q_t^-1 (x) q
    torque_saturated: np.ndarray = None      # (T,N) bool, |g| at the motor limit
    speed_saturated: np.ndarray = None        # (T,N) bool, |Omega| at the speed limit
    wheel_power: np.ndarray = None             # (T,N) signed mechanical power g*Omega [W]
    eigenaxis_deviation_deg: np.ndarray = None  # (T,) angle between omega and the
                                                 # initial error eigenaxis
    eigenaxis: np.ndarray = None                  # (3,) that initial eigenaxis
    max_torque: float = None                       # [N m] per-wheel motor limit
    max_speed: float = None                         # [rad/s] per-wheel speed limit
    wheel_momentum_capacity: float = None           # [N m s] Iw * max_speed, per wheel
    torque_envelope: np.ndarray = None               # (3,) max deliverable body
                                                      # torque along each body axis

    @classmethod
    def from_history(cls, name: str, history: dict, q_target: np.ndarray,
                      inertia_tensor: np.ndarray, wheels) -> "SimResults":
        t = np.asarray(history["t"])
        q = np.asarray(history["q"])
        omega = np.asarray(history["omega"])
        Omega = np.asarray(history["Omega"])
        u = np.asarray(history["u"])
        g = np.asarray(history["g"])
        # .get with a zero default: histories recorded before the
        # magnetorquer integration (or by a hand-rolled loop) still load.
        m = np.asarray(history.get("m", np.zeros((len(t), 3))))
        tau_mtq = np.asarray(history.get("tau_mtq", np.zeros((len(t), 3))))

        results = cls(name=name, t=t, q=q, omega=omega, Omega=Omega,
                       u=u, g=g, q_target=np.asarray(q_target, dtype=np.float64),
                       m=m, tau_mtq=tau_mtq)

        # --- attitude error (deg) ---
        q_tgt_obj = Quaternion(*results.q_target)
        q_tgt_inv = q_tgt_obj.inverse()
        err_deg = np.empty(len(q))
        q_err_hist = np.empty((len(q), 4))
        for i, qi in enumerate(q):
            q_err = q_tgt_inv * Quaternion(*qi)
            q_err.normalize()
            q_err_hist[i] = q_err.q
            w_err = np.clip(abs(q_err.q[0]), -1.0, 1.0)
            err_deg[i] = np.degrees(2.0 * np.arccos(w_err))
        results.attitude_error_deg = err_deg
        results.q_error = q_err_hist

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

        # --- actuator authority: how much of the wheel envelope was used ---
        # Reuses the array's own limits so these can never drift out of sync
        # with what allocate() actually clipped to during the run.
        results.max_torque = float(wheels.max_torque)
        results.max_speed = float(wheels.max_speed)
        results.wheel_momentum_capacity = float(wheels.Iw[0] * wheels.max_speed)
        # `>=` with a relative tolerance, not `>`: allocate() clips g to exactly
        # max_torque, so a strict `>` would report zero saturation on a run that
        # was pinned against the limit the whole way.
        # Deliverable body-torque ceiling per axis. allocate() uses the
        # min-norm g = -W^+ tau, so tau is feasible iff |W^+ tau|_inf <=
        # max_torque; along a unit axis n that caps |tau| at
        # max_torque / |W^+ n|_inf. Direction-dependent, and much less than
        # 4x the per-wheel limit.
        env = np.empty(3)
        for i in range(3):
            n = np.zeros(3)
            n[i] = 1.0
            env[i] = wheels.max_torque / np.abs(wheels.W_pinv @ n).max()
        results.torque_envelope = env

        results.torque_saturated = np.abs(g) >= results.max_torque * (1.0 - 1e-9)
        results.speed_saturated = np.abs(Omega) >= results.max_speed * (1.0 - 1e-9)
        results.wheel_power = g * Omega

        # --- eigenaxis deviation ---
        # Wie's Case 1 (mu=1, K=k*J) is the *eigenaxis* law: the body should
        # rotate about a single fixed axis, so omega(t) stays parallel to the
        # eigenaxis of the initial error quaternion. Cases 2-4 trade that away
        # for robustness, and this is the quantity that measures how much.
        e0 = q_err_hist[0, 1:]
        n0 = np.linalg.norm(e0)
        results.eigenaxis = e0 / n0 if n0 > 1e-12 else np.zeros(3)
        w_norm = np.linalg.norm(omega, axis=1)
        # Sign-free: a rotation reversing along the same axis is still on-axis,
        # hence |cos|. Undefined where the body is essentially at rest, so those
        # samples are left as NaN rather than reported as a huge deviation.
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_dev = np.abs(omega @ results.eigenaxis) / w_norm
        dev = np.degrees(np.arccos(np.clip(cos_dev, 0.0, 1.0)))
        # Blank the samples where the body is barely rotating. The threshold is
        # 1% of peak |omega|, not machine epsilon: a slew starts and ends at
        # omega = 0, and near those points the *direction* of omega is set by
        # rounding noise. Scoring those samples would report a large deviation
        # for a maneuver that is perfectly on-axis wherever it actually moves.
        moving = w_norm >= 1e-2 * w_norm.max() if w_norm.max() > 0 else np.zeros_like(w_norm, dtype=bool)
        results.eigenaxis_deviation_deg = np.where(moving, dev, np.nan)

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

    # --- actuator authority ---------------------------------------------

    def saturation_fraction(self) -> float:
        """Fraction of timesteps where ANY wheel was at its torque limit.
        The headline number for the report's saturation analysis: near 0 means
        the maneuver fits inside the wheel envelope, near 1 means the gains are
        writing cheques the Maxon ECX Flat 22L can't cash."""
        return float(self.torque_saturated.any(axis=1).mean())

    def per_wheel_saturation_fraction(self) -> np.ndarray:
        """(N,) same thing per wheel -- an uneven split points at the actuator
        geometry favouring some axes over others."""
        return self.torque_saturated.mean(axis=0)

    def speed_saturation_fraction(self) -> float:
        """Fraction of timesteps where any wheel was at its speed limit, i.e.
        momentum-saturated and unable to absorb more (needs desaturation)."""
        return float(self.speed_saturated.any(axis=1).mean())

    def peak_control_torque(self) -> float:
        """Largest commanded body torque magnitude [N m]."""
        return float(np.linalg.norm(self.u, axis=1).max())

    def peak_wheel_torque(self) -> float:
        """Largest per-wheel motor torque actually applied [N m]."""
        return float(np.abs(self.g).max())

    def torque_margin(self) -> float:
        """Peak wheel torque as a fraction of the motor limit. >1 is impossible
        (allocate() clips); ==1 means saturated at some point."""
        return self.peak_wheel_torque() / self.max_torque

    # --- cost / effort ---------------------------------------------------

    def control_effort(self) -> float:
        """Integral of |u| over the run [N m s] -- the standard control-effort
        cost. Lower is cheaper for the same pointing result, which is the
        fair way to compare two controllers that both converge."""
        return float(np.trapezoid(np.linalg.norm(self.u, axis=1), self.t))

    def electrical_energy(self) -> float:
        """Conservative electrical energy estimate [J]: integral of the sum of
        |mechanical power| over all wheels, i.e. assuming no regenerative
        recovery when braking. Upper bound on the true draw."""
        return float(np.trapezoid(np.abs(self.wheel_power).sum(axis=1), self.t))

    # --- transient shape --------------------------------------------------

    def overshoot_deg(self) -> float:
        """How far the attitude error swings back up after first reaching its
        minimum. 0 for a monotone (overdamped/critically damped) slew."""
        i_min = int(np.argmin(self.attitude_error_deg))
        if i_min >= len(self.attitude_error_deg) - 1:
            return 0.0
        return float(self.attitude_error_deg[i_min:].max() - self.attitude_error_deg[i_min])

    def mean_eigenaxis_deviation_deg(self) -> float:
        """Time-average deviation of omega from the initial error eigenaxis,
        over the samples where the body was actually moving. ~0 means the
        maneuver was a true eigenaxis (shortest-path) rotation."""
        d = self.eigenaxis_deviation_deg
        return float(np.nanmean(d)) if np.isfinite(d).any() else float("nan")

    # --- momentum management ---------------------------------------------

    def desat_used(self) -> bool:
        """True if the magnetorquers commanded a nonzero dipole at any point."""
        return self.m is not None and bool(np.any(self.m != 0.0))

    def momentum_dumped(self) -> float:
        """Peak-to-final drop in |h_w| [N m s] -- the headline number for a
        desaturation run. Near zero on a plain slew (whatever the wheels
        take up to perform the maneuver they keep at the end of it)."""
        return float(self.wheel_momentum_norm.max() - self.wheel_momentum_norm[-1])

    def summary(self) -> str:
        st = self.settling_time()
        st_str = f"{st:.2f} s" if st is not None else "did not settle"
        rpm = self.max_wheel_speed() * 60.0 / (2.0 * np.pi)
        return (
            f"--- {self.name} ---\n"
            f"  POINTING\n"
            f"    initial attitude error: {self.attitude_error_deg[0]:.2f} deg\n"
            f"    settling time (<1 deg): {st_str}\n"
            f"    final attitude error  : {self.final_attitude_error_deg():.4f} deg\n"
            f"    overshoot             : {self.overshoot_deg():.3f} deg\n"
            f"    final |omega|         : {np.linalg.norm(self.omega[-1]):.3e} rad/s\n"
            f"    mean eigenaxis dev.   : {self.mean_eigenaxis_deviation_deg():.2f} deg\n"
            f"  ACTUATOR AUTHORITY\n"
            f"    peak |u| commanded    : {self.peak_control_torque():.3e} N m\n"
            f"    peak wheel torque     : {self.peak_wheel_torque():.3e} N m "
            f"({100 * self.torque_margin():.1f}% of {self.max_torque:.1e} limit)\n"
            f"    torque saturation     : {100 * self.saturation_fraction():.2f}% of steps\n"
            f"    speed saturation      : {100 * self.speed_saturation_fraction():.2f}% of steps\n"
            f"    max wheel speed       : {self.max_wheel_speed():.2f} rad/s ({rpm:.0f} rpm)\n"
            f"  COST\n"
            f"    control effort int|u| : {self.control_effort():.4e} N m s\n"
            f"    electrical energy     : {self.electrical_energy():.4e} J\n"
            + (
                f"  MOMENTUM MANAGEMENT (magnetorquer desaturation active)\n"
                f"    peak |h_w|            : {self.wheel_momentum_norm.max():.4e} N m s\n"
                f"    final |h_w|           : {self.wheel_momentum_norm[-1]:.4e} N m s\n"
                f"    momentum dumped       : {self.momentum_dumped():.4e} N m s\n"
                f"    peak |dipole|         : {np.abs(self.m).max():.4e} A m^2\n"
                if self.desat_used() else ""
            )
            + f"  VERIFICATION\n"
            f"    max |total momentum|  : {self.total_momentum_norm.max():.6e} kg m^2/s\n"
            f"    total momentum drift  : {self.total_momentum_norm.max() - self.total_momentum_norm[0]:.3e} kg m^2/s\n"
            f"    max |q| drift         : {np.abs(self.quaternion_norm_error).max():.3e}\n"
        )
