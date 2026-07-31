"""
comparison.py
==============

Figures that take a list/dict of SimResults rather than one, overlaying
several runs on shared axes -- driven by `compare()` at the bottom.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from deimos.viz.style import BG, FG, GRID, CYCLE, _style_axes, _new_fig, _cumtrapz


def _as_items(results_list):
    """Accept either a list of SimResults or a {label: SimResults} dict."""
    if isinstance(results_list, dict):
        return list(results_list.items())
    return [(r.name, r) for r in results_list]


def compare_attitude_error(results_list, log=True):
    """Attitude error for every run on shared axes. Log scale by default --
    on a linear axis every converged run collapses onto the same flat line
    near zero and the steady-state differences, which are often the whole
    point, become invisible."""
    items = _as_items(results_list)
    fig, (ax,) = _new_fig(1, figsize=(10, 4.8))
    for i, (label, r) in enumerate(items):
        ax.plot(r.t, r.attitude_error_deg, color=CYCLE[i % len(CYCLE)],
                linewidth=1.5, label=label)
    ax.axhline(1.0, color=FG, linestyle="--", linewidth=0.9, alpha=0.5,
               label="1 deg settling threshold")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("Attitude error (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Attitude Error — controller comparison")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def compare_control_effort(results_list):
    """Cumulative int|u|dt per run -- the cost side of the trade study."""
    items = _as_items(results_list)
    fig, (ax,) = _new_fig(1, figsize=(10, 4.2))
    for i, (label, r) in enumerate(items):
        u_norm = np.linalg.norm(r.u, axis=1)
        ax.plot(r.t, _cumtrapz(u_norm, r.t), color=CYCLE[i % len(CYCLE)],
                linewidth=1.5, label=label)
    ax.set_ylabel(r"$\int |u|\,dt$  (N m s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Cumulative Control Effort — controller comparison")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def compare_saturation(results_list):
    """Saturated-timestep fraction per run, as bars. The single number the
    report's saturation analysis section is built around."""
    items = _as_items(results_list)
    fig, (ax,) = _new_fig(1, figsize=(10, 4.0))
    labels = [lb for lb, _ in items]
    fracs = [100.0 * r.saturation_fraction() for _, r in items]
    bars = ax.bar(range(len(items)), fracs,
                  color=[CYCLE[i % len(CYCLE)] for i in range(len(items))])
    for b, f in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{f:.2f}%",
                ha="center", va="bottom", color=FG, fontsize=9)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Timesteps saturated (%)")
    ax.set_title("Wheel Torque Saturation — controller comparison")
    fig.tight_layout()
    return fig


def compare_tradeoff(results_list):
    """Settling time vs control effort -- the actual design trade. Bottom-left
    is better on both axes; anything up and to the right is dominated. Marker
    size encodes saturation, since a fast cheap run that lives at the torque
    limit is not really comparable to one that doesn't."""
    items = _as_items(results_list)
    fig = plt.figure(figsize=(7, 5.5))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    for i, (label, r) in enumerate(items):
        st = r.settling_time()
        if st is None:
            st = r.t[-1]
            label = label + " (never settled)"
        size = 60 + 400 * r.saturation_fraction()
        ax.scatter(st, r.control_effort(), s=size, color=CYCLE[i % len(CYCLE)],
                   alpha=0.85, edgecolors=FG, linewidths=0.6, label=label)
        ax.annotate(label.split(" (")[0], (st, r.control_effort()),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", color=FG, fontsize=7, alpha=0.85)
    # Log-log: both axes can span 2+ orders of magnitude across a case sweep
    # (e.g. settle times from ~3s to 300s once slow/pathological cases are
    # included) -- on linear axes the fast cases collapse into one illegible
    # cluster near the origin and the labels overlap.
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Settling time to <1 deg (s)")
    ax.set_ylabel(r"Control effort $\int |u|\,dt$  (N m s)")
    ax.set_title("Performance / Cost Trade\n(marker size = saturation fraction)")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def compare_eigenaxis(results_list):
    """Eigenaxis deviation per run -- how far each law departs from the
    shortest-path rotation."""
    items = _as_items(results_list)
    fig, (ax,) = _new_fig(1, figsize=(10, 4.2))
    for i, (label, r) in enumerate(items):
        ax.plot(r.t, r.eigenaxis_deviation_deg, color=CYCLE[i % len(CYCLE)],
                linewidth=1.4, label=f"{label} (mean {r.mean_eigenaxis_deviation_deg():.1f} deg)")
    ax.set_ylabel("Deviation from eigenaxis (deg)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Eigenaxis Deviation — controller comparison")
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def comparison_table(results_list, stability=None) -> str:
    """Fixed-width metrics table across runs. Plain text so it drops straight
    into a report appendix or a commit message without a pandas dependency.

    stability: optional {label: verdict_string} dict (e.g. from
    analysis.stability.stability_branch()'s first return value, or "not
    claimed" for PD). When given, an extra right-hand column is printed so a
    run winning on settle time while sitting in unproven-stability territory
    is visible in the SAME row as its performance numbers, not only in the
    single-run design card. SimResults itself carries no K/D/J/mu (those are
    controller/config internals, not simulation output), which is why this
    is a caller-supplied dict rather than something computed in here.
    """
    items = _as_items(results_list)
    width = max(24, min(60, max(len(lb) for lb, _ in items) + 2))
    hdr = (f"{'run':<{width}}{'settle(s)':>11}{'final(deg)':>12}{'oversh(deg)':>13}"
           f"{'peak|u|(Nm)':>13}{'sat(%)':>9}{'effort(Nms)':>13}{'energy(J)':>12}"
           f"{'eigax(deg)':>12}")
    stab_width = 0
    if stability:
        stab_width = max(28, max(len(v) for v in stability.values()) + 2)
        hdr += f"{'stability':>{stab_width}}"
    lines = [hdr, "-" * len(hdr)]
    for label, r in items:
        st = r.settling_time()
        st_s = f"{st:.2f}" if st is not None else "never"
        row = (
            f"{label[:width - 1]:<{width}}{st_s:>11}{r.final_attitude_error_deg():>12.4f}"
            f"{r.overshoot_deg():>13.3f}{r.peak_control_torque():>13.3e}"
            f"{100 * r.saturation_fraction():>9.2f}{r.control_effort():>13.4e}"
            f"{r.electrical_energy():>12.4e}{r.mean_eigenaxis_deviation_deg():>12.2f}"
        )
        if stability:
            row += f"{stability.get(label, 'n/a'):>{stab_width}}"
        lines.append(row)
    return "\n".join(lines)


COMPARE_REGISTRY = {
    "attitude_error": compare_attitude_error,
    "control_effort": compare_control_effort,
    "saturation": compare_saturation,
    "tradeoff": compare_tradeoff,
    "eigenaxis": compare_eigenaxis,
}


def compare(results_list, names="all", save_dir=None, show=True, prefix="compare"):
    """Same contract as `plot()` in single_run.py, but every figure overlays
    multiple runs.

    results_list: list of SimResults, or {label: SimResults} to control the
                  legend text (useful when two runs share a config name).
    """
    selected = list(COMPARE_REGISTRY.keys()) if names == "all" else list(names)

    figs = {}
    for name in selected:
        if name not in COMPARE_REGISTRY:
            raise KeyError(f"Unknown comparison plot '{name}'. "
                           f"Available: {list(COMPARE_REGISTRY.keys())}")
        fig = COMPARE_REGISTRY[name](results_list)
        figs[name] = fig
        if save_dir is not None:
            from pathlib import Path
            sd = Path(save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            fig.savefig(sd / f"{prefix}_{name}.png", facecolor=BG, dpi=150,
                        bbox_inches="tight")

    if show:
        plt.show()
    elif save_dir is not None:
        for fig in figs.values():
            plt.close(fig)
    return figs
