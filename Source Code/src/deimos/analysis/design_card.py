"""
design_card.py
===============

Turns a SimConfig into a readable description of *the control law it builds*
-- gains, closed-loop modes, which stability theorem is being invoked, and
whether the actuators can physically deliver what the gains ask for.

Nothing here runs a simulation or invents a control law. It reads the same
objects sim.runner.simulate() builds (via _build_controller) and reports on
them, so what it prints is by construction what the plant actually flew.
"""

from __future__ import annotations

import numpy as np

from deimos.sim.runner import _build_controller, _build_wheels
from deimos.sim.config import SimConfig
from deimos.analysis.stability import closed_loop_modes, stability_branch, offdiag_materiality


def torque_envelope(wheels) -> np.ndarray:
    """
    Max deliverable body torque along each body axis [N m], given the
    per-wheel motor limit and the min-norm allocation actually used.

    allocate() sets g = -W^+ tau, so tau is deliverable iff |W^+ tau|_inf <=
    max_torque. Along a unit direction n the largest feasible magnitude is
    therefore max_torque / |W^+ n|_inf. This is the honest ceiling for the
    actuator geometry -- it is NOT N x the per-wheel limit, and it is not the
    same in every direction.
    """
    env = np.empty(3)
    for i in range(3):
        n = np.zeros(3)
        n[i] = 1.0
        env[i] = wheels.max_torque / np.abs(wheels.W_pinv @ n).max()
    return env


def describe(config: SimConfig) -> str:
    """Full text design card for the controller `config` specifies."""
    J = config.satellite.inertia_tensor
    ctype = config.controller.type.strip().lower()
    controller = _build_controller(config)
    wheels = _build_wheels(config)

    L = []
    add = L.append
    add("=" * 74)
    add(f"CONTROLLER DESIGN CARD - {config.name}")
    add("=" * 74)

    add("\nPLANT")
    add(f"  inertia tensor J [kg m^2] (CoM-referenced):")
    for row in J:
        add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
    add(f"  principal moments diag(J) : {np.array2string(np.diag(J), precision=6)}")
    off = J - np.diag(np.diag(J))
    add(f"  max |off-diagonal| / min diag : {np.abs(off).max() / np.diag(J).min():.2e} "
        f"({'diagonal-dominant' if np.abs(off).max() < 0.01 * np.diag(J).min() else 'NOT diagonal-dominant'})")
    _, _, offdiag_note = offdiag_materiality(J)
    add(f"  {offdiag_note}")

    add("\nACTUATORS")
    add(f"  {wheels!r}")
    env = torque_envelope(wheels)
    add(f"  max deliverable body torque per axis [N m]: "
        f"{np.array2string(env, precision=4)}")
    add(f"    (= max_torque / |W^+ n|_inf, i.e. the real actuator ceiling,")
    add(f"     not N x the per-wheel limit)")

    if ctype == "wie":
        K, D, mu = controller.K, controller.D, controller.mu
        add("\nCONTROL LAW - Wie, Weiss & Arapostathis (1989), Sec. III")
        add("  u = mu*(omega x J*omega) - D*omega - sgn(q_e0)*K*q_ev")
        add(f"  case  : {config.controller.case}")
        if config.controller.zeta is not None:
            add(f"  design: zeta = {config.controller.zeta}, "
                f"t_settle = {config.controller.settling_time_s} s "
                f"-> omega_n = 8/(zeta*t_s) = "
                f"{8.0 / (config.controller.zeta * config.controller.settling_time_s):.4f} rad/s")
        else:
            # direct form (k_scale/d_scale) -- written by the NSGA-II search
            # and by hand-tuning off the sizing-rule curve; there is no
            # (zeta, t_s) pair to report, the scale factors ARE the design
            add(f"  design: direct scale factors K = {config.controller.k_scale:.6g} * J, "
                f"D = {config.controller.d_scale:.6g} * J")
        add(f"  mu    : {mu}")
        add("  K [N m]:")
        for row in K:
            add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
        add("  D [N m s]:")
        for row in D:
            add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
        add(f"  Ki    : {'none (base case, per the original proof)' if controller.Ki is None else controller.Ki}")
        add(f"  u_max : {'none (allocator clip only)' if config.controller.u_max is None else config.controller.u_max}")

        status, detail, _ = stability_branch(K, D, J, mu)
        add(f"\nGLOBAL STABILITY: {status}")
        add(f"  {detail}")
        if config.controller.case in ("mortensen", "near_eigenaxis"):
            ratio, material, offdiag_note = offdiag_materiality(J)
            if material:
                add(f"  ADDITIONAL CAVEAT for case='{config.controller.case}': {offdiag_note}")
                add("  The K above may not be the K design() intended -- refit with a")
                add("  full-matrix-aware method before trusting this stability verdict.")
            # else: not material, already stated once in PLANT above -- no
            # need to repeat it here as a caveat on a proof that isn't affected.
        if controller.Ki is not None:
            add("  NOTE: integral action is active. It is a heuristic augmentation")
            add("        and is NOT covered by the Sec. III proof above.")

    elif ctype == "pd":
        K, D = config.controller.Kp, config.controller.Kd
        mu = 0.0
        add("\nCONTROL LAW - quaternion PD")
        add("  u = -Kp*q_ev - Kd*omega")
        add("  Kp [N m]:")
        for row in K:
            add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
        add("  Kd [N m s]:")
        for row in D:
            add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
        add(f"  Ki    : {'none (base PD, no integral action)' if controller.Ki is None else controller.Ki}")
        add("\nGLOBAL STABILITY: not claimed")
        add("  This law has no sign-of-q0 term, so it has two closed-loop")
        add("  equilibria (q_e = +/-[1,0,0,0]) and can take the long way round")
        add("  on slews past 180 deg. Wie's sgn(q_e0) factor is exactly the fix.")
        if controller.Ki is not None:
            add("  NOTE: integral action is active. It is a heuristic augmentation")
            add("        with no anti-windup and no stability analysis of its own --")
            add("        this law already made no global-stability claim, so adding")
            add("        Ki does not change that, it just adds another unproven term.")

    elif ctype == "pid":
        K, D = config.controller.Kp, config.controller.Kd
        mu = 0.0
        add("\nCONTROL LAW - quaternion PID")
        add("  u = -Kp*q_ev - Kd*omega - Ki*integral(q_ev dt)")
        add("  Kp [N m]:")
        for row in K:
            add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
        add("  Kd [N m s]:")
        for row in D:
            add("    [" + "  ".join(f"{v: .6e}" for v in row) + "]")
        add(f"  Ki    : {controller.Ki}  (required for this type -- see control/pid.py)")
        add("\nGLOBAL STABILITY: not claimed")
        add("  Same two-equilibria issue as PD (no sign-of-q0 term), plus the")
        add("  integral term is a heuristic augmentation with no anti-windup and")
        add("  no stability analysis of its own. If the wheels saturate for a long")
        add("  stretch, the accumulator keeps growing regardless -- watch for")
        add("  windup in the eigenaxis_deviation / saturation figures.")

    else:
        add(f"\nCONTROL LAW - '{ctype}' (no design card written for this type yet)")
        return "\n".join(L)

    # --- closed-loop modes, common to both laws ---
    omega_n, zeta = closed_loop_modes(K, D, J)
    add("\nLINEARIZED CLOSED LOOP (per body axis)")
    add("            omega_n [rad/s]    zeta      t_settle = 8/(zeta*omega_n) [s]")
    for i, ax in enumerate("xyz"):
        ts = 8.0 / (zeta[i] * omega_n[i]) if zeta[i] * omega_n[i] > 0 else np.inf
        add(f"    {ax}       {omega_n[i]:12.4f}    {zeta[i]:8.4f}    {ts:12.2f}")
    if zeta.min() < 0.5:
        add(f"  WARNING: min zeta = {zeta.min():.3f} - underdamped, expect ringing.")
    if zeta.max() > 2.0:
        add(f"  NOTE: max zeta = {zeta.max():.3f} - heavily overdamped, slow tail.")

    # --- torque budget at t=0, where the error (and so the demand) is largest ---
    from deimos.math.quaternion import euler_to_quaternion
    from deimos.control.wie import WieRegulator
    q0 = euler_to_quaternion(*np.radians(config.initial.attitude_rpy_deg)).normalize()
    qt = euler_to_quaternion(*np.radians(config.target.attitude_rpy_deg)).normalize()
    q_e0 = WieRegulator.quaternion_error(q0.q, qt.q)
    omega0 = np.asarray(config.initial.omega, dtype=float)
    u0 = -(D @ omega0) - np.sign(q_e0[0] if q_e0[0] != 0 else 1.0) * (K @ q_e0[1:])
    g0, _ = wheels.allocate(u0)
    add("\nTORQUE BUDGET AT t = 0 (largest error => largest demand)")
    add(f"  initial attitude error   : {np.degrees(2 * np.arccos(np.clip(abs(q_e0[0]), -1, 1))):.2f} deg")
    add(f"  commanded u(0)     [N m] : {np.array2string(u0, precision=4)}")
    add(f"  per-wheel g(0)     [N m] : {np.array2string(g0, precision=4)}")
    worst = np.abs(np.linalg.pinv(wheels.W) @ u0).max()
    add(f"  worst wheel demand / limit : {worst / wheels.max_torque:.2f}x "
        f"({'WILL SATURATE at t=0' if worst > wheels.max_torque else 'within envelope'})")

    add("=" * 74)
    return "\n".join(L)
