"""
single_run.py
=============

Each function takes a SimResults and returns a matplotlib Figure. None of
these touch the simulation -- they're pure presentation, operating only on
the arrays already sitting in SimResults. Add a new figure by writing a new
function here and registering it in REGISTRY at the bottom.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from deimos.sim.results import SimResults
from deimos.viz.style import BG, FG, GRID, COLORS, WHEEL_CYCLE, _style_axes, _new_fig, _cumtrapz


# --- Scenario A / general figures -----------------------------------------

def plot_attitude_error(results: SimResults):
    fig, (ax,) = _new_fig(1)
    ax.plot(results.t, results.attitude_error_deg, color=COLORS["main"], linewidth=1.6)
    ax.set_ylabel("Attitude error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Attitude Error vs Time — {results.name}")
    fig.tight_layout()
    return fig


def plot_body_rates(results: SimResults):
    fig, (ax,) = _new_fig(1)
    ax.plot(results.t, results.omega[:, 0], color=COLORS["x"], linewidth=1.3, label=r"$\omega_x$")
    ax.plot(results.t, results.omega[:, 1], color=COLORS["y"], linewidth=1.3, label=r"$\omega_y$")
    ax.plot(results.t, results.omega[:, 2], color=COLORS["z"], linewidth=1.3, label=r"$\omega_z$")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Body Angular Rates vs Time — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID)
    fig.tight_layout()
    return fig


def plot_quaternion_components(results: SimResults):
    fig, (ax,) = _new_fig(1)
    labels = ["w", "x", "y", "z"]
    colors = [COLORS["main"], COLORS["x"], COLORS["y"], COLORS["z"]]
    for i in range(4):
        ax.plot(results.t, results.q[:, i], color=colors[i], linewidth=1.3, label=labels[i])
    ax.set_ylabel("Quaternion component")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Quaternion Components vs Time — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID)
    fig.tight_layout()
    return fig


def plot_wheel_panel(results: SimResults):
    """Wheel speeds and per-wheel motor torque, stacked."""
    fig, axes = _new_fig(2)
    ax_speed, ax_torque = axes

    N = results.Omega.shape[1]
    for i in range(N):
        ax_speed.plot(results.t, results.Omega[:, i], linewidth=1.2, label=f"wheel {i+1}")
        ax_torque.plot(results.t, results.g[:, i], linewidth=1.2, label=f"wheel {i+1}")

    ax_speed.set_ylabel("Wheel speed (rad/s)")
    ax_speed.set_title(f"Reaction Wheel Speeds — {results.name}")
    ax_speed.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)

    ax_torque.set_ylabel("Motor torque (N m)")
    ax_torque.set_xlabel("Time (s)")
    ax_torque.set_title(f"Per-Wheel Motor Torque — {results.name}")
    ax_torque.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)

    fig.tight_layout()
    return fig


def plot_control_torque(results: SimResults):
    fig, (ax,) = _new_fig(1)
    ax.plot(results.t, results.u[:, 0], color=COLORS["x"], linewidth=1.3, label=r"$u_x$")
    ax.plot(results.t, results.u[:, 1], color=COLORS["y"], linewidth=1.3, label=r"$u_y$")
    ax.plot(results.t, results.u[:, 2], color=COLORS["z"], linewidth=1.3, label=r"$u_z$")
    ax.set_ylabel("Commanded control torque (N m)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Control Torque vs Time — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID)
    fig.tight_layout()
    return fig


# --- Control-strategy figures ----------------------------------------------
# These are the ones that say something about the *controller*, as opposed to
# just showing that the spacecraft got where it was going.

def plot_torque_envelope(results: SimResults):
    """Commanded body torque against what the actuators can actually deliver.

    The dashed lines are the per-axis ceiling max_torque / |W^+ n|_inf -- the
    real limit under the min-norm allocation, which is direction-dependent. A
    trace pressed against its dashed line is a controller asking for
    authority the hardware does not have.
    """
    fig, (ax,) = _new_fig(1, figsize=(10, 4.5))
    for i, ax_name in enumerate("xyz"):
        ax.plot(results.t, results.u[:, i], color=COLORS[ax_name], linewidth=1.3,
                label=rf"$u_{ax_name}$")
    for i, ax_name in enumerate("xyz"):
        lim = results.torque_envelope[i]
        ax.axhline(lim, color=COLORS[ax_name], linestyle=":", linewidth=1.0, alpha=0.75)
        ax.axhline(-lim, color=COLORS[ax_name], linestyle=":", linewidth=1.0, alpha=0.75)
    ax.plot([], [], color=FG, linestyle=":", linewidth=1.0,
            label="per-axis envelope")
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.set_ylabel("Commanded body torque (N m)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Control Torque vs Actuator Envelope — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def plot_saturation(results: SimResults):
    """Per-wheel motor torque against the hard limit, plus a saturation
    timeline. The bottom strip is the diagnostic the report's saturation
    analysis needs: black = that wheel was pinned at its limit at that instant.
    """
    fig, axes = _new_fig(2, figsize=(10, 3.2))
    ax_t, ax_s = axes

    lim = results.max_torque
    N = results.g.shape[1]
    # Wheel traces deliberately avoid red -- red is reserved for the limit line,
    # and a wheel drawn in the same colour as the thing it must not cross is
    # exactly the wrong cue.
    for i in range(N):
        ax_t.plot(results.t, results.g[:, i], linewidth=1.1,
                  color=WHEEL_CYCLE[i % len(WHEEL_CYCLE)], label=f"wheel {i+1}")
    ax_t.axhline(lim, color="#ff4b4b", linestyle="--", linewidth=1.2, label="motor limit")
    ax_t.axhline(-lim, color="#ff4b4b", linestyle="--", linewidth=1.2)
    # Keep the limit lines in frame even when the run stays well inside them,
    # so "how much margin was there" is readable, not just "it didn't clip".
    ax_t.set_ylim(-1.15 * lim, 1.15 * lim)
    ax_t.set_ylabel("Motor torque (N m)")
    frac = 100.0 * results.saturation_fraction()
    ax_t.set_title(f"Wheel Torque vs {lim:.1e} N m Limit — "
                   f"{frac:.2f}% of steps saturated — {results.name}")
    ax_t.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8, ncol=2)

    # timeline strip: one row per wheel, shaded where saturated
    ax_s.imshow(results.torque_saturated.T, aspect="auto", cmap="inferno",
                interpolation="nearest", vmin=0, vmax=1,
                extent=[results.t[0], results.t[-1], N + 0.5, 0.5])
    ax_s.set_yticks(range(1, N + 1))
    ax_s.set_yticklabels([f"w{i+1}" for i in range(N)])
    ax_s.set_ylabel("Saturated")
    ax_s.set_xlabel("Time (s)")
    ax_s.grid(False)
    if not results.torque_saturated.any():
        # An all-dark strip is ambiguous: it looks the same as a broken plot.
        # Say so explicitly.
        ax_s.text(0.5, 0.5, "no wheel reached the torque limit at any timestep",
                  transform=ax_s.transAxes, ha="center", va="center",
                  color=FG, fontsize=10, alpha=0.75)
    fig.tight_layout()
    return fig


def plot_phase_portrait(results: SimResults):
    """Attitude error angle vs its rate -- the classic way to read a regulator's
    character. A critically damped law spirals straight into the origin; an
    underdamped one loops around it; a saturated one shows a straight
    constant-deceleration segment before it can turn."""
    fig = plt.figure(figsize=(6, 5.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    err = results.attitude_error_deg
    rate = np.gradient(err, results.t)
    pts = ax.scatter(err, rate, c=results.t, cmap="viridis", s=3)
    ax.plot(err, rate, color=FG, alpha=0.18, linewidth=0.7)
    ax.scatter([err[0]], [rate[0]], color="#ff4b4b", s=45, zorder=5, label="start")
    ax.scatter([err[-1]], [rate[-1]], color="#4bff7a", s=45, zorder=5, label="end")
    cb = fig.colorbar(pts, ax=ax)
    cb.set_label("Time (s)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.get_yticklabels(), color=FG)

    ax.set_xlabel("Attitude error (deg)")
    ax.set_ylabel("Error rate (deg/s)")
    ax.set_title(f"Phase Portrait — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def plot_eigenaxis_deviation(results: SimResults):
    """Angle between omega(t) and the initial error eigenaxis.

    This is the figure that distinguishes Wie's four cases from each other.
    Case 1 (mu=1, K=k*J) is the eigenaxis law and should sit near 0 -- a single
    fixed rotation axis, the shortest angular path. Cases 2-4 give that up in
    exchange for robustness to inertia error, and this plot is how much they
    give up. Samples where the body is essentially at rest are dropped (the
    axis of a non-rotation is undefined), which is why the trace can end early.
    """
    fig, (ax,) = _new_fig(1)
    ax.plot(results.t, results.eigenaxis_deviation_deg,
            color=COLORS["main"], linewidth=1.4)
    mean_dev = results.mean_eigenaxis_deviation_deg()
    ax.axhline(mean_dev, color="#ff4b4b", linestyle="--", linewidth=1.1,
               label=f"mean {mean_dev:.2f} deg")
    ax.set_ylabel("Deviation of $\\omega$ from eigenaxis (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Eigenaxis Deviation — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def plot_error_quaternion(results: SimResults):
    """Error quaternion components. q_e0 -> +/-1 and q_ev -> 0 at convergence;
    which sign q_e0 settles to is the Lyapunov branch the controller committed
    to, which for the Wie law is frozen at t=0 by sgn(q_e0) (Remark 5)."""
    fig, (ax,) = _new_fig(1)
    labels = [r"$q_{e0}$", r"$q_{e1}$", r"$q_{e2}$", r"$q_{e3}$"]
    colors = [COLORS["main"], COLORS["x"], COLORS["y"], COLORS["z"]]
    for i in range(4):
        ax.plot(results.t, results.q_error[:, i], color=colors[i],
                linewidth=1.3, label=labels[i])
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    ax.set_ylabel("Error quaternion component")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Error Quaternion — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8, ncol=4)
    fig.tight_layout()
    return fig


def plot_control_cost(results: SimResults):
    """Cumulative control effort and instantaneous electrical power.

    Two controllers that both converge are separated by what they *spent*
    doing it, so this is the fair comparison axis for the trade study.
    """
    fig, axes = _new_fig(2, figsize=(10, 3.2))
    ax_e, ax_p = axes

    u_norm = np.linalg.norm(results.u, axis=1)
    ax_e.plot(results.t, _cumtrapz(u_norm, results.t),
              color=COLORS["main"], linewidth=1.5)
    ax_e.set_ylabel(r"$\int |u|\,dt$  (N m s)")
    ax_e.set_title(f"Cumulative Control Effort — {results.name}")

    power = np.abs(results.wheel_power).sum(axis=1)
    ax_p.plot(results.t, power, color=COLORS["y"], linewidth=1.3)
    ax_p.set_ylabel("Wheel power (W)")
    ax_p.set_xlabel("Time (s)")
    ax_p.set_title(f"Instantaneous Wheel Power (no regen) — "
                   f"total {results.electrical_energy():.3e} J")
    fig.tight_layout()
    return fig


def plot_momentum_envelope(results: SimResults):
    """Wheel speeds against the speed limit -- the momentum-storage budget.
    Hitting this is a different failure from torque saturation: the wheel can
    no longer absorb momentum at all and the array needs desaturation."""
    fig, (ax,) = _new_fig(1)
    N = results.Omega.shape[1]
    for i in range(N):
        ax.plot(results.t, results.Omega[:, i], linewidth=1.2,
                color=WHEEL_CYCLE[i % len(WHEEL_CYCLE)], label=f"wheel {i+1}")
    ax.axhline(results.max_speed, color="#ff4b4b", linestyle="--",
               linewidth=1.2, label="speed limit")
    ax.axhline(-results.max_speed, color="#ff4b4b", linestyle="--", linewidth=1.2)
    ax.set_ylim(-1.15 * results.max_speed, 1.15 * results.max_speed)
    used = 100.0 * results.max_wheel_speed() / results.max_speed
    ax.set_ylabel("Wheel speed (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Momentum Storage Budget — peak {used:.1f}% of capacity — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# --- Verification figures --------------------------------------------------

def plot_momentum_drift(results: SimResults):
    fig, (ax,) = _new_fig(1)
    drift = results.total_momentum_norm - results.total_momentum_norm[0]
    ax.plot(results.t, drift, color=COLORS["main"], linewidth=1.4)
    ax.set_ylabel(r"$|H(t)| - |H(0)|$  (kg m$^2$/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Total Angular Momentum Drift — {results.name}")
    fig.tight_layout()
    return fig


def plot_energy_drift(results: SimResults):
    fig, (ax,) = _new_fig(1)
    ax.plot(results.t, results.total_kinetic_energy, color=COLORS["main"], linewidth=1.4)
    ax.set_ylabel("Total kinetic energy (J)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"System Kinetic Energy vs Time — {results.name}")
    fig.tight_layout()
    return fig


def plot_quaternion_norm_error(results: SimResults):
    fig, (ax,) = _new_fig(1)
    ax.plot(results.t, results.quaternion_norm_error, color=COLORS["main"], linewidth=1.4)
    ax.set_ylabel(r"$|q| - 1$ (pre-renormalize)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Quaternion Norm Drift — {results.name}")
    fig.tight_layout()
    return fig


def plot_polhode(results: SimResults):
    """omega_x vs omega_y vs omega_z trajectory -- should trace a closed
    curve for torque-free motion (sanity check, most informative when run
    with disturbance and control both disabled)."""
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.plot(results.omega[:, 0], results.omega[:, 1], results.omega[:, 2],
             color=COLORS["main"], linewidth=1.0)
    ax.set_xlabel(r"$\omega_x$", color=FG)
    ax.set_ylabel(r"$\omega_y$", color=FG)
    ax.set_zlabel(r"$\omega_z$", color=FG)
    ax.set_title(f"Polhode — {results.name}", color=FG)
    ax.tick_params(colors=FG)
    fig.tight_layout()
    return fig


# --- momentum-management figures -------------------------------------------

def plot_wheel_momentum(results: SimResults):
    """Wheel momentum |h_w| against the single-wheel capacity Iw*max_speed.

    THE desaturation diagnostic: on a desat run |h_w| should start high
    (pre-loaded wheels), decay while the magnetorquers are active, and level
    off once the latch disengages. On a plain slew it shows how much of the
    momentum envelope the maneuver itself consumes."""
    fig, (ax,) = _new_fig(1)
    for i, ax_name in enumerate("xyz"):
        ax.plot(results.t, results.wheel_momentum[:, i], color=COLORS[ax_name],
                linewidth=1.1, alpha=0.8, label=rf"$h_{{w,{ax_name}}}$")
    ax.plot(results.t, results.wheel_momentum_norm, color=COLORS["main"],
            linewidth=1.8, label=r"$|h_w|$")
    if results.wheel_momentum_capacity is not None:
        ax.axhline(results.wheel_momentum_capacity, color=FG, linestyle=":",
                   linewidth=1.0, alpha=0.75,
                   label=r"single-wheel capacity $I_w \Omega_{max}$")
    ax.set_ylabel("Wheel momentum (N m s)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Stored Wheel Momentum — {results.name}")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def plot_mtq_dipole(results: SimResults):
    """Magnetorquer dipole command against the per-axis hardware limit, plus
    the resulting torque on the body. Flat zero = desaturation never engaged
    (or the scenario doesn't enable magnetorquers at all)."""
    fig, axes = _new_fig(2)
    ax_m, ax_tau = axes

    for i, ax_name in enumerate("xyz"):
        ax_m.plot(results.t, results.m[:, i], color=COLORS[ax_name],
                  linewidth=1.2, label=rf"$m_{ax_name}$")
        ax_tau.plot(results.t, results.tau_mtq[:, i], color=COLORS[ax_name],
                    linewidth=1.2, label=rf"$\tau_{{mtq,{ax_name}}}$")

    m_lim = float(np.abs(results.m).max())
    if m_lim > 0:
        # the clip ceiling is only known to the config; the visible flat-top
        # of a clipped command IS the limit, so draw at the observed max when
        # anything was commanded at all
        ax_m.axhline(m_lim, color=FG, linestyle=":", linewidth=0.9, alpha=0.6)
        ax_m.axhline(-m_lim, color=FG, linestyle=":", linewidth=0.9, alpha=0.6)

    ax_m.set_ylabel("Dipole command (A m$^2$)")
    ax_m.set_title(f"Magnetorquer Dipole — {results.name}")
    ax_m.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)

    ax_tau.set_ylabel("MTQ torque on body (N m)")
    ax_tau.set_xlabel("Time (s)")
    ax_tau.set_title(f"Magnetorquer Torque — {results.name}")
    ax_tau.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)

    fig.tight_layout()
    return fig


# --- registry ---------------------------------------------------------

REGISTRY = {
    "attitude_error": plot_attitude_error,
    "body_rates": plot_body_rates,
    "quaternion_components": plot_quaternion_components,
    "wheel_panel": plot_wheel_panel,
    "control_torque": plot_control_torque,
    "torque_envelope": plot_torque_envelope,
    "saturation": plot_saturation,
    "phase_portrait": plot_phase_portrait,
    "eigenaxis_deviation": plot_eigenaxis_deviation,
    "error_quaternion": plot_error_quaternion,
    "control_cost": plot_control_cost,
    "momentum_envelope": plot_momentum_envelope,
    "momentum_drift": plot_momentum_drift,
    "energy_drift": plot_energy_drift,
    "quaternion_norm_error": plot_quaternion_norm_error,
    "polhode": plot_polhode,
    "wheel_momentum": plot_wheel_momentum,
    "mtq_dipole": plot_mtq_dipole,
}

SCENARIO_A = ["attitude_error", "body_rates", "wheel_panel", "quaternion_components"]
SCENARIO_B = ["attitude_error", "wheel_panel", "control_torque"]
VERIFICATION = ["momentum_drift", "energy_drift", "quaternion_norm_error", "polhode"]
# Everything that says something about the control law rather than the outcome.
CONTROL = ["error_quaternion", "torque_envelope", "saturation", "phase_portrait",
           "eigenaxis_deviation", "control_cost", "momentum_envelope"]
# The desaturation story: pre-loaded momentum draining while the dipole works.
DESAT = ["wheel_momentum", "mtq_dipole", "wheel_panel", "attitude_error"]


def plot(results: SimResults, names="all", save_dir=None, show=True):
    """
    names: "all", a group name ("scenario_a" / "scenario_b" / "verification"),
           or a list of individual figure names from REGISTRY.
    save_dir: if given, saves each figure as PNG into this directory.
    show: if True, calls plt.show() at the end (set False in batch/headless use).
    """
    if names == "all":
        selected = list(REGISTRY.keys())
    elif names == "scenario_a":
        selected = SCENARIO_A
    elif names == "scenario_b":
        selected = SCENARIO_B
    elif names == "verification":
        selected = VERIFICATION
    elif names == "control":
        selected = CONTROL
    elif names == "desat":
        selected = DESAT
    else:
        selected = list(names)

    figs = {}
    for name in selected:
        if name not in REGISTRY:
            raise KeyError(f"Unknown plot name '{name}'. Available: {list(REGISTRY.keys())}")
        fig = REGISTRY[name](results)
        figs[name] = fig
        if save_dir is not None:
            from pathlib import Path
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"{results.name}_{name}.png",
                        facecolor=BG, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    elif save_dir is not None:
        # Pure export mode: the caller isn't displaying these, so release them
        # rather than letting pyplot accumulate open figures across cells.
        # The Figure objects stay usable for savefig if the caller kept them.
        for fig in figs.values():
            plt.close(fig)

    return figs
