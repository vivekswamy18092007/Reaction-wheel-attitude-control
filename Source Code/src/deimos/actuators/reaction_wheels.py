"""
Momentum-exchange actuator model for the satellite ADCS simulation.

Current primary geometry: 3-wheel orthogonal CONE (config="cone")

The design uses three reaction wheels on a cone: each spin axis is canted
theta = arccos(1/sqrt(3)) = 54.7356 deg from the body +z axis, spaced 120 deg
apart in azimuth, with wheel 1's in-plane projection along body +y. The three
axes are MUTUALLY ORTHOGONAL (W^T W = I), so the array behaves as a clean
3-axis set while keeping all motors off the body axes.

This replaces the earlier 4-wheel pyramid (config="pyramid"), which is retained
for reference and comparison. The pyramid provided single-wheel-failure
tolerance at the cost of a redundant fourth wheel; the 3-wheel cone is
minimum-redundancy (lose one wheel = lose one axis) but simpler and lighter.

Geometry:

Each wheel i spins about a unit axis e_i (a column of the 3xN axis matrix W,
expressed in the body frame). Stacking the axes gives the distribution matrix

        W = [ e_1  e_2  ...  e_N ]                         (3 x N)

A positive motor torque g_i accelerates wheel i along +e_i, so its stored
angular momentum is  h_i = I_w,i * Omega_i  along e_i, and the wheel momentum
expressed in the body frame is

        h_w = sum_i (I_w,i * Omega_i) e_i = W (Iw ⊙ Omega)  (3-vector)

Control allocation:

The controller asks for a body control torque tau_c (3-vector). The wheel+body
system conserves angular momentum, so to deliver tau_c onto the body the wheels
must take up momentum at the opposite rate:

        d/dt h_w = -tau_c        =>        W g = -tau_c

We solve this with the Moore-Penrose pseudo-inverse W^+ = W^T (W W^T)^-1:

        g = -W^+ tau_c

For the 3-wheel cone the axes are orthonormal, so W is square and orthogonal
and the pseudo-inverse reduces exactly to the transpose, W^+ = W^T (verified
in code). For the redundant 4-wheel pyramid the system is under-determined and
pinv returns the minimum-||g||^2 solution, matching the Lagrange-multiplier
derivation in Shirazi & Mirshams (2014, eq. 7-10). Using pinv keeps the *same*
allocation code correct for every geometry (cone, pyramid, body-axis orthogonal).

The reaction torque actually delivered to the body is

        tau_body = -W g

(equal to tau_c when tau_c lies in range(W) and no wheel torque is clipped).

Coupling into Euler's equation

With total system momentum H = J*w + h_w, the transport theorem gives

        J w_dot = tau_ext - w x (J*w + h_w) + tau_body
        Omega_dot = g / Iw

Note the gyroscopic term carries h_w, *not* just J*w -- the wheel momentum has
to live inside the cross product. Hold g fixed (zero-order hold) across the RK4
sub-steps and recompute h_w from each sub-step's Omega.

Assumptions (document these in the report)
-----------
- Rigid wheels; spin axes fixed in the body frame.
- The back-coupling term Iw*(e_i . w_dot) is neglected (Iw << J for a CubeSat),
  so Omega_dot_i = g_i / Iw,i. Tighten this only if Iw approaches J.
- Motor torque saturates at +/- max_torque. Wheel speed saturates at
  +/- max_speed and this is ENFORCED in the plant: allocate() zeroes any
  motor torque that would accelerate a pinned wheel further (back-EMF /
  motor-controller cutoff), so control authority along that wheel is
  genuinely lost, not just flagged. Braking a pinned wheel stays allowed --
  that is how it recovers. A torque-speed derating curve (available torque
  falling linearly with speed, per the Maxon datasheet) would be the next
  refinement; the hard cutoff is the conservative first-order model.

References
----------
- Markley & Crassidis, Fundamentals of Spacecraft Attitude Determination and
  Control (reaction-wheel dynamics).
- Shirazi & Mirshams, "Pyramidal reaction wheel arrangement optimization of
  satellite attitude control subsystem for minimizing power consumption",
  Int'l J. Aeronautical & Space Sci., 15(2), 2014 (pyramid geometry,
  min-norm allocation, tilt-angle optimum ~32 deg for their test-bed).
"""

import numpy as np

from deimos import constants


class ReactionWheelArray:

    def __init__(
        self,
        axes=None,
        wheel_inertia=constants.WHEEL_INERTIA,    # [kg m^2] CAD-exported flywheel inertia (Al 7075 hub + rim)
        max_torque=constants.WHEEL_MAX_TORQUE,    # [N m]   software torque cap (Maxon ECX Flat 22L can do ~28.8e-3)
        max_speed=constants.WHEEL_MAX_SPEED,      # [rad/s] 12000 rpm, per ECX Flat 22L datasheet
        tilt_deg=constants.WHEEL_TILT_DEG,        # pyramid tilt angle beta from the body x-y plane
        config=constants.WHEEL_CONFIG,            # "pyramid" (4 wheels), "orthogonal" (3 wheels) or "cone" (3 wheels)
        Omega0=None,             # initial wheel speeds [rad/s], length N
    ):
        if axes is None:
            if config == "pyramid":
                axes = self.pyramid_axes(tilt_deg)
            elif config == "orthogonal":
                axes = self.orthogonal_axes()
            elif config == "cone":
                axes = self.cone_axes()   # angle fixed internally; tilt_deg unused
            else:
                raise ValueError(f"unknown config '{config}'")

        self.W = np.asarray(axes, dtype=float)          # (3, N) axis matrix
        if self.W.ndim != 2 or self.W.shape[0] != 3:
            raise ValueError("axes must be a 3xN matrix (columns are spin axes)")
        self.N = self.W.shape[1]

        # force each column to be a unit vector
        self.W = self.W / np.linalg.norm(self.W, axis=0)

        # sanity: a "cone"/"orthogonal" set must actually be orthogonal
        if config in ("cone", "orthogonal"):
            gram = self.W.T @ self.W
            if not np.allclose(gram, np.eye(self.N), atol=1e-6):
                raise ValueError(
                    f"'{config}' axes are not mutually orthogonal; "
                    f"check the geometry.\nW^T W =\n{gram}"
                )

        # minimum-norm allocation operator  (N x 3)
        self.W_pinv = np.linalg.pinv(self.W)

        # per-wheel inertia, broadcast a scalar to all wheels
        self.Iw = (np.full(self.N, float(wheel_inertia))
                   if np.isscalar(wheel_inertia)
                   else np.asarray(wheel_inertia, dtype=float))

        self.max_torque = float(max_torque)
        self.max_speed = float(max_speed)

        self.Omega = (np.zeros(self.N) if Omega0 is None
                      else np.asarray(Omega0, dtype=float).copy())

    # ----- geometry builders -----

    @staticmethod
    def pyramid_axes(tilt_deg):
        """4 wheels projecting onto +x, +y, -x, -y, each tilted up out of the
        body x-y plane by beta. Columns are the unit spin axes."""
        b = np.radians(tilt_deg)
        cb, sb = np.cos(b), np.sin(b)
        return np.array([
            [cb,  0.0, -cb,  0.0],
            [0.0,  cb, 0.0,  -cb],
            [sb,   sb,  sb,   sb],
        ])

    @staticmethod
    def orthogonal_axes():
        """3 wheels along the body x, y, z axes."""
        return np.eye(3)

    @staticmethod
    def cone_axes():
        """3 wheels on a cone, MUTUALLY ORTHOGONAL (W^T@W = I).

        Each spin axis is canted theta = arccos(1/sqrt(3)) = 54.7356 deg from
        the body +z axis, spaced 120 deg apart in azimuth, with wheel 1's
        in-plane projection along body +y (azimuth measured from +y):
            e = [ sin(theta) sin(phi),  sin(theta) cos(phi),  cos(theta) ]

        The cant angle is FIXED by the orthogonality requirement. it is NOT
        a free parameter and there is deliberately no tilt argument.

        Returns the 3x3 axis matrix (columns = unit spin axes).
        """
        theta = np.arccos(1.0 / np.sqrt(3.0))       # 54.7356 deg from +z, fixed by orthogonality
        st, ct = np.sin(theta), np.cos(theta)
        phis = np.radians([0.0, 120.0, 240.0])      # azimuth from +y
        return np.array([
            [st * np.sin(p) for p in phis],
            [st * np.cos(p) for p in phis],
            [ct, ct, ct],
        ])

    # ----- allocation & torque -----

    def allocate(self, tau_cmd, Omega=None):
        """Map a desired body control torque to per-wheel motor torques.
        Returns (g, torque_saturated_mask). g already carries the reaction
        sign, so body_torque(g) reproduces tau_cmd when unsaturated.

        Omega: current wheel speeds [rad/s]. When given, speed saturation is
        enforced: a wheel at/over max_speed accepts no further accelerating
        torque (that g_i is zeroed) but may still brake. When omitted the
        old torque-clip-only behaviour applies -- callers that integrate
        wheel speeds (the propagator) must pass Omega or the plant will
        happily spin wheels past the motor's physical limit."""
        tau_cmd = np.asarray(tau_cmd, dtype=float)
        g = -(self.W_pinv @ tau_cmd)
        torque_saturated = np.abs(g) > self.max_torque
        g = np.clip(g, -self.max_torque, self.max_torque)
        if Omega is not None:
            Omega = np.asarray(Omega, dtype=float)
            pinned = (np.abs(Omega) >= self.max_speed) & (np.sign(g) == np.sign(Omega))
            g = np.where(pinned, 0.0, g)
        return g, torque_saturated

    def body_torque(self, g):
        """Reaction torque delivered to the body by motor torques g."""
        return -self.W @ np.asarray(g, dtype=float)

    def wheel_accel(self, g):
        """Wheel angular accelerations Omega_dot = g / Iw."""
        return np.asarray(g, dtype=float) / self.Iw

    # ----- state / diagnostics -----

    def momentum(self, Omega=None):
        """Wheel angular momentum h_w in the body frame (3-vector).
        Add this inside the gyroscopic cross product in Euler's equation."""
        Omega = self.Omega if Omega is None else np.asarray(Omega, dtype=float)
        return self.W @ (self.Iw * Omega)

    def speed_saturated(self, Omega=None):
        """Boolean mask of wheels at/over the speed limit (authority lost)."""
        Omega = self.Omega if Omega is None else np.asarray(Omega, dtype=float)
        return np.abs(Omega) >= self.max_speed

    def mechanical_power(self, g, Omega=None):
        """Signed per-wheel mechanical power g*Omega [W]. Negative = braking.
        For a conservative electrical draw estimate (no regen recovery) use
        np.sum(np.abs(mechanical_power(...)))."""
        Omega = self.Omega if Omega is None else np.asarray(Omega, dtype=float)
        return np.asarray(g, dtype=float) * Omega

    def kinetic_energy(self, Omega=None):
        """Rotational KE stored in the wheels: 0.5 * sum(Iw * Omega^2) [J].
        Useful as a conservation/energy-budget check."""
        Omega = self.Omega if Omega is None else np.asarray(Omega, dtype=float)
        return 0.5 * np.sum(self.Iw * Omega ** 2)

    # ----- convenience integrator (standalone testing only) -----

    def step(self, tau_cmd, dt):
        """One forward-Euler update of the wheel speeds. For the coupled
        simulation, integrate Omega inside the RK4 propagator instead (hold g
        as ZOH across sub-steps); use allocate()/wheel_accel()/momentum() there.
        Returns (tau_body, g, torque_saturated, speed_saturated)."""
        g, tsat = self.allocate(tau_cmd, self.Omega)
        self.Omega = self.Omega + self.wheel_accel(g) * dt
        return self.body_torque(g), g, tsat, self.speed_saturated()

    def __repr__(self):
        return (f"ReactionWheelArray(N={self.N}, "
                f"Iw={self.Iw[0]:.2e} kg m^2, "
                f"max_torque={self.max_torque:.2e} N m, "
                f"max_speed={self.max_speed:.1f} rad/s)")
