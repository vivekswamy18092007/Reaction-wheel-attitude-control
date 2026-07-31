import numpy as np

from deimos.math.quaternion import quaternion_error


class WieRegulator:
    """
    Wie, Weiss & Arapostathis (1989) quaternion feedback regulator.

        u = mu*(omega x J*omega) - D @ omega - s * (K @ q_ev)

    s = sign(q_e0), frozen ONCE at the start of the maneuver (Remark 5) --
    not recomputed every step, or the controller could jump onto the wrong
    Lyapunov branch mid-maneuver.

    (omega x J*omega) is exactly the term RotationalDynamics.omega_dot
    subtracts from the plant (Euler's eq: J*omega_dot = torque - omega x J*omega).
    mu=1 cancels it in closed loop; mu=0 leaves it in and instead requires a
    matched K -- see `design()`.

    Scalar-first quaternion convention throughout: q = [q0, q1, q2, q3].

    Global stability (paper Sec. III) requires EITHER:
        mu = 1   (any K = K^T > 0, D = D^T > 0), or
        mu = 0   AND   K^-1 = alpha*J + beta*I   (alpha, beta >= 0)
    mu=0 with an unmatched K (e.g. K = k*J) has NO stability guarantee from
    this proof -- the gyroscopic residual in V_dot is not shown to vanish.
    """

    def __init__(self, J, K, D, mu, Ki=None, u_max=None, integral_limit=None,
                 decouple_wheel_momentum=False):
        """
        decouple_wheel_momentum: include the reaction wheels' stored angular
            momentum in the mu-decoupling term, i.e.

                decoupling = mu * omega x (J*omega + h_w)

            instead of mu * omega x (J*omega).

            THIS IS REQUIRED FOR THE EIGENAXIS CASE TO ACTUALLY BE EIGENAXIS
            ON THIS PLANT, and it is off by default only because turning it on
            silently would change every result already recorded with mu != 0.

            The paper models an ideal torque actuator, so its plant is
            J*omega_dot = u - omega x J*omega and mu = 1 cancels the whole
            gyroscopic term. DEIMoS's plant is a momentum-exchange system
            (dynamics/rigid_body.py):

                J*omega_dot = u - omega x (J*omega + h_w)

            so mu = 1 without this flag leaves -omega x h_w uncancelled. That
            residual is not parallel to the error eigenaxis, and it grows
            through the maneuver because h_w is exactly what the wheels have
            been accumulating to perform it. The result is a Case 1 controller
            whose defining property quietly fails: measured on the 40/30/25
            deg slew, mean eigenaxis deviation is 3.65 deg with the residual
            left in and 0.005 deg with it removed -- three orders of
            magnitude, from a term that is invisible in the settling time.

            With the flag on, the cancellation is exact again and the closed
            loop is once more omega_dot = -d*omega - s*k*q_ev.

            Requires h_w to be passed to compute_torque(); sim/runner.py does
            this automatically for any controller advertising
            `needs_wheel_momentum`.

        integral_limit: [N m] per-axis cap on the *contribution* of the
            integral term, |Ki * z|. None (default) = unbounded, which is the
            original behaviour every existing preset relies on.

            Same rationale as PIDController's: capping the contribution
            rather than the raw accumulator keeps the limit physically
            meaningful and independent of Ki's magnitude, so it means the
            same thing for every gain set a search might try. The
            accumulator is clamped to match, because capping only the output
            would still let z grow without bound and then take arbitrarily
            long to unwind.

            This matters far more for a searched gain set than for a
            hand-picked one: the GA will happily propose the top of the
            Ki range, where an unbounded accumulator integrates the entire
            slew transient and demands several times the wheel limit.
        """
        self.J = np.asarray(J, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.mu = float(mu)
        self.Ki = np.asarray(Ki, dtype=float) if Ki is not None else None
        self.u_max = u_max
        self.integral_limit = integral_limit
        self.decouple_wheel_momentum = bool(decouple_wheel_momentum)
        self._sign = None
        self._z = np.zeros(3)

    @property
    def needs_wheel_momentum(self) -> bool:
        """sim/runner.py reads this to decide whether to pass h_w into
        compute_torque(). False when mu = 0, because then there is no
        decoupling term for h_w to enter -- asking for it would be dead
        weight and would suggest cases 2-4 depend on wheel state, which they
        do not."""
        return self.decouple_wheel_momentum and self.mu != 0.0

    # ---------------- gain design (Sec. IV & VI) ----------------

    @staticmethod
    def eigenaxis_gains(J, k, d):
        """Case 1 (`design(case="eigenaxis")`) with k and d given DIRECTLY
        instead of derived from (zeta, settling_time).

            K = k*J,   D = d*J,   mu = 1

        This is the parameterization the NSGA-II search operates on
        (tuning/objectives.py, controller_type="WIE"), and it exists because
        `design()` reaches the same (K, D) only through the large-angle
        relation t_s = 8/(zeta*omega_n). That relation is a *sizing rule* --
        a linearized, disturbance-free, unsaturated estimate. Forcing a
        search to move along it would restrict the reachable (K, D) to the
        one-dimensional curve k = 2*omega_n^2, d = 2*zeta*omega_n, and every
        candidate it rejected would be rejected for obeying a rule of thumb
        rather than for performing badly. Searching (k, d) as free positive
        scalars spans the whole plane; the design rule then becomes the warm
        start (see objectives.encode_config_gains) rather than a constraint.

        Two properties are preserved exactly, and they are the reason the
        search is over two scalars rather than six per-axis gains:

        EIGENAXIS.  With mu=1 the plant's gyroscopic term cancels, leaving
            J*omega_dot = -d*J*omega - s*k*J*q_ev,  i.e.  omega_dot = -d*omega - s*k*q_ev
        -- J drops out entirely and all three components of q_ev obey the
        SAME scalar second-order equation. So omega stays parallel to the
        initial error eigenaxis and the body turns about one fixed axis:
        the shortest angular path. Break the K = k*J tie (per-axis gains)
        and that cancellation is gone, the axes decouple at different rates,
        and the path bows away from the eigenaxis.

        GLOBAL STABILITY.  Sec. III guarantees global asymptotic stability
        for mu=1 and ANY K = K^T > 0, D = D^T > 0. Since J is positive
        definite, every k > 0 and d > 0 satisfies that. So the entire search
        space is globally stable by construction -- the GA cannot return an
        unstable gain set, and no stability constraint has to be bolted onto
        the objectives to stop it trying.
        """
        J = np.asarray(J, dtype=float)
        k, d = float(k), float(d)
        if k <= 0.0 or d <= 0.0:
            raise ValueError(f"k and d must be positive (got k={k}, d={d}) -- "
                             "K = k*J and D = d*J must stay positive definite "
                             "for the Sec. III stability guarantee to apply")
        return k * J, d * J, 1.0

    @staticmethod
    def design(J, case, zeta, settling_time):
        """
        Build (K, D, mu) for one of the four cases in the paper's design
        example (Sec. VI), using the large-angle settling-time relation

            settling_time = 8 / (zeta * omega_n)

        (Sec. IV -- the large-angle 8/.. relation, not the small-angle 4/..
        one, since this targets large slews).

        Assumes J is diagonal-dominant and uses diag(J) as the principal
        moments -- consistent with DEIMoS's confirmed diagonal-dominant
        inertia tensor. Re-derive if products of inertia grow significant.

        case: "eigenaxis" | "robust" | "mortensen" | "near_eigenaxis"
        """
        J = np.asarray(J, dtype=float)
        Ji = np.diag(J).copy()

        omega_n = 8.0 / (zeta * settling_time)
        k = 2.0 * omega_n**2
        d = 2.0 * zeta * omega_n

        if case == "eigenaxis":
            # K = k*J, mu = 1 -- shortest angular path, needs accurate J
            K, D, mu = k * J, d * J, 1.0

        elif case == "robust":
            # K = k*I, mu = 0 -- globally stable under unknown Delta_J
            # (Sec. V), sacrifices the eigenaxis property
            K, D, mu = k * np.eye(3), d * J, 0.0

        elif case == "mortensen":
            # K = k / J_i  (Remark 1, beta=0 extreme: K^-1 = (1/k)*J), mu=0
            K, D, mu = k * np.diag(1.0 / Ji), d * J, 0.0

        elif case == "near_eigenaxis":
            # Eq. 25: minimize sum_i(alpha*Ji + beta - 1/Ji)^2
            # solved directly as ordinary least squares -- equivalent to
            # the closed-form Eq. 26a/26b, done this way to avoid
            # transcribing the OCR'd expression off the scanned paper
            A = np.column_stack([Ji, np.ones(3)])
            b = 1.0 / Ji
            (alpha, beta), *_ = np.linalg.lstsq(A, b, rcond=None)
            K = np.linalg.inv(alpha * J + beta * np.eye(3))
            D, mu = d * J, 0.0

        else:
            raise ValueError(f"unknown case '{case}'")

        return K, D, mu

    # ---------------- quaternion error ----------------

    @staticmethod
    def quaternion_error(q, q_target):
        """q_e = q_target^-1 (x) q, scalar-first."""
        return quaternion_error(q, q_target)

    # ---------------- control ----------------

    def compute_torque(self, q, omega, q_target=None, dt=None, h_w=None):
        """
        q_target=None reproduces the paper's q_e = q simplification
        (regulation to the frame q is already expressed in). Pass a
        target quaternion for slews to an arbitrary attitude.

        Pass dt only if self.Ki was set -- see the class docstring for
        the caveat on integral action.

        h_w: the reaction wheels' stored angular momentum [N m s], needed
        only when decouple_wheel_momentum is on. Omitting it there is a
        silent correctness bug rather than a crash, so it raises instead.
        """
        q_e = q if q_target is None else quaternion_error(q, q_target)
        q0, q_ev = q_e[0], q_e[1:]

        if self._sign is None:
            self._sign = 1.0 if q0 >= 0.0 else -1.0  # Remark 5, frozen once

        # H is the TOTAL angular momentum the plant's gyroscopic term is
        # built from -- body plus wheels. Cancelling only the body part
        # leaves a residual that is not along the eigenaxis; see the class
        # docstring for the measured cost.
        H = self.J @ omega
        if self.needs_wheel_momentum:
            if h_w is None:
                raise ValueError(
                    "decouple_wheel_momentum is on but compute_torque() was "
                    "called without h_w. Silently falling back to body-only "
                    "decoupling would leave -omega x h_w uncancelled and "
                    "break the eigenaxis property without any error -- so "
                    "this raises instead. sim/runner.py passes h_w "
                    "automatically; a hand-rolled loop must too.")
            H = H + np.asarray(h_w, dtype=float)

        decoupling = self.mu * np.cross(omega, H)
        u = decoupling - self.D @ omega - self._sign * (self.K @ q_ev)

        if self.Ki is not None and dt is not None:
            # heuristic integral augmentation -- NOT covered by the Sec. III
            # global proof; keep Ki small relative to K, D or verify
            # stability empirically. For the rigorous version, port Module
            # 3 Sec 3.6's augmented-Lyapunov z-state instead.
            integral = self.Ki * self._z
            if self.integral_limit is not None:
                integral = np.clip(integral, -self.integral_limit,
                                   self.integral_limit)
            u = u - integral
            self._accumulate(q_ev, dt)

        if self.u_max is not None:
            u = self.saturate(u, self.u_max)

        return u

    def _accumulate(self, q_ev, dt):
        """Advance the integral state, clamped if anti-windup is enabled.

        The sign latch multiplies the accumuland for the same reason it
        multiplies the proportional term: z has to accumulate along the
        branch the controller committed to at t=0, or the integral fights
        the proportional term for the whole maneuver.
        """
        self._z = self._z + self._sign * q_ev * dt
        if self.integral_limit is None:
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            z_max = np.where(np.abs(self.Ki) > 0.0,
                             self.integral_limit / np.abs(self.Ki), np.inf)
        self._z = np.clip(self._z, -z_max, z_max)

    def reset(self):
        """
        Call before each new simulation run. Sign and integral state are
        frozen/accumulated per maneuver -- reusing one instance across
        multiple runs (e.g. a case-comparison sweep) without resetting
        carries state across runs incorrectly.
        """
        self._sign = None
        self._z = np.zeros(3)

    # ---------------- saturation ----------------

    @staticmethod
    def saturate(u, u_max):
        """
        Per-axis clip to the wheel torque limit. This is the practical
        engineering choice, not the full case-by-case Lyapunov-consistent
        saturation law from Module 4 Sec 4.2 (tracking / regulator /
        rate-regulator problems treated separately) -- formalizing that
        per-case proof for this specific controller is still open; worth
        doing for the final report if time allows.
        """
        return np.clip(u, -u_max, u_max)


def build_wie_from_config(config, J):
    """
    Factory for the YAML-preset pipeline. Two mutually exclusive ways to get
    (K, D, mu), because a preset and a search want different handles:

      SIZING-RULE FORM (the original) -- keys: case, zeta, settling_time_s.
        K, D, mu come from WieRegulator.design(), i.e. from the large-angle
        relation t_s = 8/(zeta*omega_n). This is how a human specifies a
        controller: "critically damped, settle in 8 seconds".

      DIRECT FORM -- keys: k_scale, d_scale (case must be "eigenaxis", or be
        omitted). K = k*J, D = d*J, mu = 1, via
        WieRegulator.eigenaxis_gains(). This is how the NSGA-II search
        specifies one: two free positive scalars, no sizing rule in the way.
        See eigenaxis_gains() for why the search uses this form.

    Optional in both: Ki ([kx, ky, kz]), u_max, integral_limit,
    decouple_wheel_momentum.
    """
    has_direct = "k_scale" in config or "d_scale" in config
    has_rule = "zeta" in config or "settling_time_s" in config

    if has_direct and has_rule:
        raise ValueError(
            "Wie controller config gives BOTH k_scale/d_scale and "
            "zeta/settling_time_s. These are two different ways to set the "
            "same (K, D) and would silently disagree -- pick one.")

    if has_direct:
        case = config.get("case", "eigenaxis")
        if case != "eigenaxis":
            raise ValueError(
                f"k_scale/d_scale set K = k*J, D = d*J, mu = 1, which IS the "
                f"'eigenaxis' case -- but case='{case}' was requested. Cases "
                f"2-4 derive K from J in ways that have no single scale "
                f"factor, so they cannot be expressed this way; use "
                f"zeta/settling_time_s for those.")
        if "k_scale" not in config or "d_scale" not in config:
            raise KeyError("the direct form needs BOTH k_scale and d_scale")
        K, D, mu = WieRegulator.eigenaxis_gains(
            J, k=config["k_scale"], d=config["d_scale"])
    else:
        K, D, mu = WieRegulator.design(
            J, case=config["case"], zeta=config["zeta"],
            settling_time=config["settling_time_s"],
        )

    Ki = np.array(config["Ki"]) if "Ki" in config else None
    u_max = config.get("u_max")
    integral_limit = config.get("integral_limit")
    return WieRegulator(J, K, D, mu, Ki=Ki, u_max=u_max,
                        integral_limit=integral_limit,
                        decouple_wheel_momentum=config.get(
                            "decouple_wheel_momentum", False))
