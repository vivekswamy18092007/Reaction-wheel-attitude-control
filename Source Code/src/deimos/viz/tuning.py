"""
tuning.py
=========

Figures describing a GA *run* -- the Pareto front, how it converged, and
where in gain space it ended up. These take the (pop, obj, decode, ga)
output of tuning.objectives.tune(), not a SimResults; for the time-domain
behaviour of the tuned gains, re-simulate the picks and use
viz.comparison.compare() as usual.

Same dark theme as the rest of viz/, via style.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from deimos.tuning.objectives import (
    OBJECTIVE_LABELS, decoded_to_flat, gene_bounds, gene_labels, make_decode,
    pareto_picks,
)
from deimos.viz.style import BG, FG, GRID, COLORS, CYCLE, _new_fig, _style_axes

# Distinct, saturated colours for the handful of highlighted picks -- these
# must stand out against the semi-transparent cloud of front points.
PICK_COLORS = {
    "fastest": "#ff4b4b",
    "best_tracking": "#ff4b4b",
    "most_accurate": "#4bff7a",
    "cheapest_effort": "#ffb84b",
    "least_saturated": "#4bffe6",
    "knee": "#ffffff",
}


def _labels_for(controller_type):
    return OBJECTIVE_LABELS[str(controller_type).strip().upper()]


def _pick_color(label, i):
    return PICK_COLORS.get(label, CYCLE[i % len(CYCLE)])


def plot_pareto_projections(obj, controller_type="PD", picks=None):
    """The three pairwise 2D projections of a 3-objective Pareto front.

    Easier to read quantitatively than the 3D view: each panel is an honest
    2D trade curve, and a point that looks dominated here but isn't is being
    saved by the third objective -- which is the whole reason to keep all
    three panels rather than picking a favourite pair.
    """
    obj = np.asarray(obj, dtype=float)
    labels = _labels_for(controller_type)
    picks = pareto_picks(obj, controller_type) if picks is None else picks

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(BG)

    for ax, (i, j) in zip(np.atleast_1d(axes), pairs):
        _style_axes(ax)
        ax.scatter(obj[:, i], obj[:, j], s=34, color=COLORS["main"],
                   alpha=0.55, edgecolors="none", label="Pareto front")
        for k, (label, idx) in enumerate(picks.items()):
            ax.scatter(obj[idx, i], obj[idx, j], s=130, marker="*",
                       color=_pick_color(label, k), edgecolors=FG,
                       linewidths=0.6, zorder=5, label=label)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])

    axes[0].legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)
    fig.suptitle(f"Pareto front projections — {controller_type} gain search",
                 color=FG)
    fig.tight_layout()
    return fig


def plot_pareto_3d(obj, controller_type="PD", picks=None):
    """All three objectives at once. The front should read as a surface;
    a front collapsed to a line or a point means one objective is doing no
    work at these bounds (or the search has not spread yet)."""
    obj = np.asarray(obj, dtype=float)
    labels = _labels_for(controller_type)
    picks = pareto_picks(obj, controller_type) if picks is None else picks

    fig = plt.figure(figsize=(7.5, 6.2))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)

    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=38, color=COLORS["main"],
               alpha=0.6, edgecolors="none", depthshade=True)
    for k, (label, idx) in enumerate(picks.items()):
        ax.scatter([obj[idx, 0]], [obj[idx, 1]], [obj[idx, 2]], s=150,
                   marker="*", color=_pick_color(label, k), edgecolors=FG,
                   linewidths=0.6, depthshade=False, label=label)

    ax.set_xlabel(labels[0], color=FG)
    ax.set_ylabel(labels[1], color=FG)
    ax.set_zlabel(labels[2], color=FG)
    ax.set_title(f"Pareto front — {controller_type} gain search", color=FG)
    ax.tick_params(colors=FG)
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)
    fig.tight_layout()
    return fig


def plot_convergence(history, controller_type="PD"):
    """Best value of each objective vs generation, plus the front size.

    Generation 0 is the random initial population, so the drop from 0 to 1
    is what the GA bought over random sampling. A curve flat from early on
    means those extra generations were wasted -- the honest thing to report
    rather than quietly running 80 of them.
    """
    labels = _labels_for(controller_type)
    gens = np.array([h["generation"] for h in history])
    best = np.array([h["best_per_objective"] for h in history])
    front = np.array([h["front_size"] for h in history])

    n_obj = best.shape[1]
    fig, axes = plt.subplots(1, n_obj + 1, figsize=(4.2 * (n_obj + 1), 4.0))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)

    for i in range(n_obj):
        ax = axes[i]
        _style_axes(ax)
        ax.plot(gens, best[:, i], color=CYCLE[i % len(CYCLE)], linewidth=1.7,
                marker="o", markersize=3)
        ax.set_xlabel("generation")
        ax.set_ylabel(f"best {labels[i]}")
        ax.set_title(f"best {labels[i]}", fontsize=9)
        # Objectives here span orders of magnitude (effort ~1e-3, settle ~1e1);
        # log scale only when the data is strictly positive.
        if np.all(best[:, i] > 0) and best[:, i].max() / best[:, i].min() > 20:
            ax.set_yscale("log")

    ax = axes[-1]
    _style_axes(ax)
    ax.plot(gens, front, color=COLORS["main"], linewidth=1.7, marker="o", markersize=3)
    ax.set_xlabel("generation")
    ax.set_ylabel("Pareto front size")
    ax.set_title("front size", fontsize=9)

    fig.suptitle(f"NSGA-II convergence — {controller_type} gain search", color=FG)
    fig.tight_layout()
    return fig


def plot_gain_parallel_coordinates(pop, decode, controller_type="PD", obj=None,
                                    picks=None):
    """Where on the Pareto front each gain actually sits, as parallel
    coordinates.

    Each polyline is one Pareto solution; each vertical axis is one gene,
    normalized to its own log-uniform search bounds so 0 = lower bound and
    1 = upper bound. This is the plot that answers "did the search pin a gain
    against its bound?" (a flat line at 0 or 1 means the bounds are wrong,
    not that the gain is optimal) and "which gains actually vary along the
    front?" (a gene whose lines all overlap is not a live trade).
    """
    pop = np.asarray(pop, dtype=float)
    names = gene_labels(controller_type)
    bounds = gene_bounds(controller_type)

    flat = np.array([decoded_to_flat(decode(ind), controller_type) for ind in pop])
    lo = np.array([np.log10(b[0]) for b in bounds])
    hi = np.array([np.log10(b[1]) for b in bounds])
    norm = (np.log10(flat) - lo) / (hi - lo)

    picks = (pareto_picks(obj, controller_type)
             if picks is None and obj is not None else (picks or {}))

    fig, (ax,) = _new_fig(1, figsize=(11, 5.0))
    x = np.arange(len(names))

    for row in norm:
        ax.plot(x, row, color=COLORS["main"], alpha=0.22, linewidth=1.0)
    for k, (label, idx) in enumerate(picks.items()):
        ax.plot(x, norm[idx], color=_pick_color(label, k), linewidth=2.0,
                marker="o", markersize=4, label=label, zorder=5)

    for xi in x:
        ax.axvline(xi, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("position within search bounds (log-scaled)")
    ax.set_title(f"Pareto gain sets — {controller_type}\n"
                 f"0 = lower bound, 1 = upper bound of each gene's log-uniform range",
                 fontsize=10)
    if picks:
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    fig.tight_layout()
    return fig


def _series(history, key, default=np.nan):
    return np.array([h.get(key, default) for h in history], dtype=float)


def plot_hypervolume(history, controller_type="PD"):
    """Hypervolume of the front vs generation -- the honest convergence plot.

    Per-objective bests (plot_convergence) only see the ENDPOINTS of the
    front, and those can sit flat for many generations while the interior of
    the front is still filling in and the knee is still moving. Hypervolume
    is monotone with Pareto dominance, so it is sensitive to convergence,
    spread and cardinality at once: if this curve is still climbing, the run
    is still buying something, and if it has plateaued it genuinely has not.

    The lower panel is the relative gain over the trailing stall window --
    i.e. literally the quantity the stopping rule thresholds, so the decision
    to stop is visible rather than asserted.
    """
    gens = _series(history, "generation")
    hv = _series(history, "hypervolume")
    gain = _series(history, "hv_relative_gain")

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        _style_axes(ax)

    axes[0].plot(gens, hv, color=COLORS["main"], linewidth=1.8, marker="o",
                 markersize=3)
    axes[0].set_ylabel("hypervolume (fraction of worst-case box)")
    axes[0].set_title(f"Front hypervolume — {controller_type} gain search", fontsize=10)

    if np.isfinite(gain).any():
        axes[1].plot(gens, gain, color=COLORS["y"], linewidth=1.4)
        axes[1].axhline(0.0, color=FG, linewidth=0.8, alpha=0.4)
        finite = gain[np.isfinite(gain)]
        if finite.size and (finite > 0).any():
            axes[1].set_yscale("symlog", linthresh=1e-6)
    axes[1].set_ylabel("relative HV gain\nover stall window")
    axes[1].set_xlabel("generation")

    fig.tight_layout()
    return fig


def plot_objective_spread(history, controller_type="PD"):
    """Best vs population-median for each objective, over generations.

    The gap between the two lines is selection pressure made visible. Early
    on the median sits far above the best; as the population converges onto
    the front the two close. A median that never moves while the best does
    means the search is riding a handful of lucky individuals and the rest of
    the population is dead weight -- worth knowing, and invisible in a
    best-only convergence plot.
    """
    labels = _labels_for(controller_type)
    gens = _series(history, "generation")
    best = np.array([h["best_per_objective"] for h in history], dtype=float)
    med = np.array([h.get("median_per_objective",
                          np.full(best.shape[1], np.nan)) for h in history],
                   dtype=float)

    n_obj = best.shape[1]
    fig, axes = plt.subplots(1, n_obj, figsize=(4.6 * n_obj, 4.0))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        _style_axes(ax)
        ax.plot(gens, best[:, i], color=CYCLE[i % len(CYCLE)], linewidth=1.7,
                label="front best")
        if np.isfinite(med[:, i]).any():
            ax.plot(gens, med[:, i], color=FG, linewidth=1.1, alpha=0.6,
                    linestyle="--", label="population median")
            ax.fill_between(gens, best[:, i], med[:, i],
                            color=CYCLE[i % len(CYCLE)], alpha=0.15)
        ax.set_xlabel("generation")
        ax.set_ylabel(labels[i])
        stack = np.concatenate([best[:, i], med[:, i]])
        stack = stack[np.isfinite(stack) & (stack > 0)]
        if stack.size and stack.max() / stack.min() > 20:
            ax.set_yscale("log")
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)

    fig.suptitle(f"Selection pressure — {controller_type} gain search", color=FG)
    fig.tight_layout()
    return fig


def plot_seed_comparison(histories, fronts, labels=None, controller_type="PD",
                          merged=None):
    """Cross-seed view: hypervolume trajectories, final HV spread, and every
    seed's front overlaid on one trade projection.

    This is the figure that turns three runs into a result. One stochastic
    trajectory tells you nothing about whether the answer is reproducible;
    three overlaid fronts show directly how much of the trade surface is
    seed-independent, and the final-HV bar chart is the variance number to
    quote rather than hand-wave.

    histories: list of ga.history lists, one per seed.
    fronts: list of (n_i, M) objective arrays, one per seed.
    merged: optional (n, M) merged non-dominated front across all seeds.
    """
    obj_labels = _labels_for(controller_type)
    labels = labels or [f"seed {i}" for i in range(len(histories))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        _style_axes(ax)

    # --- 1. HV trajectories -------------------------------------------
    finals = []
    for i, h in enumerate(histories):
        gens = _series(h, "generation")
        hv = _series(h, "hypervolume")
        axes[0].plot(gens, hv, color=CYCLE[i % len(CYCLE)], linewidth=1.6,
                     label=labels[i])
        finals.append(hv[-1] if len(hv) else np.nan)
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("hypervolume fraction")
    axes[0].set_title("convergence per seed", fontsize=10)
    axes[0].legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)

    # --- 2. final HV spread -------------------------------------------
    finals = np.array(finals, dtype=float)
    axes[1].bar(range(len(finals)), finals,
                color=[CYCLE[i % len(CYCLE)] for i in range(len(finals))])
    if np.isfinite(finals).sum() > 1:
        mu, sd = np.nanmean(finals), np.nanstd(finals)
        axes[1].axhline(mu, color=FG, linestyle="--", linewidth=1.0,
                        label=f"mean {mu:.4f} ± {sd:.4f}")
        axes[1].legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=8)
    axes[1].set_xticks(range(len(finals)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("final hypervolume fraction")
    axes[1].set_title("run-to-run variance", fontsize=10)

    # --- 3. fronts overlaid -------------------------------------------
    # Merged front drawn as hollow rings UNDER the seed points: a filled
    # marker on top would hide exactly the information this panel exists to
    # show, namely which seed each surviving point came from.
    if merged is not None and len(merged):
        m = np.atleast_2d(np.asarray(merged, dtype=float))
        axes[2].scatter(m[:, 0], m[:, 1], s=150, marker="o", facecolors="none",
                        edgecolors="#ffffff", linewidths=1.3, zorder=3,
                        label="merged front")
    for i, f in enumerate(fronts):
        f = np.atleast_2d(np.asarray(f, dtype=float))
        axes[2].scatter(f[:, 0], f[:, 1], s=34, alpha=0.85, edgecolors="none",
                        color=CYCLE[i % len(CYCLE)], zorder=5, label=labels[i])
    axes[2].set_xlabel(obj_labels[0])
    axes[2].set_ylabel(obj_labels[1])
    axes[2].set_xscale("log")
    axes[2].set_title("fronts overlaid", fontsize=10)
    axes[2].legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)

    fig.suptitle(f"Multi-seed summary — {controller_type} gain search", color=FG)
    fig.tight_layout()
    return fig


def _diag_labels(history):
    for h in history:
        if "diagnostic_labels" in h:
            return list(h["diagnostic_labels"])
    return []


# The engineering quantities worth their own panel, and how to render them.
# ITAE is deliberately absent: it is objective 0 for both controller types and
# already has its own panel in plot_convergence. Repeating it here would show
# the same quantity twice under two different aggregations, which is exactly
# the kind of thing that makes a figure argue with itself.
_DIAG_PANELS = [
    ("settling_time_s", "settling time (s)", False),
    ("steady_state_error_deg", "steady-state error (deg)", True),
    ("control_effort_Nms", r"control effort $\int|u|dt$ (N m s)", True),
    ("overshoot_deg", "overshoot (deg)", False),
    ("saturation_fraction", "torque saturation (fraction)", False),
    ("peak_wheel_torque_Nm", "peak wheel torque (N m)", True),
]


def plot_diagnostic_history(history, controller_type="PD"):
    """Settling time, steady-state error, control effort and friends, per
    generation -- best on the front and population median.

    These are recorded for every individual but steer nothing (see
    tuning/objectives.py: settling time is a sentinel as a search signal and
    ties every non-settling candidate). They are the numbers a reader
    actually understands, so they are worth plotting even though the search
    is sorting on ITAE.

    The settling-time panel additionally shows how many individuals settled
    at all, on a twin axis. That count rising from zero is the most legible
    evidence the search is working, and it is invisible in objective space
    precisely because ITAE never ties.
    """
    labels = _diag_labels(history)
    if not labels:
        fig, (ax,) = _new_fig(1, figsize=(9, 3.2))
        ax.text(0.5, 0.5, "no diagnostics recorded for this run\n"
                          "(evaluator was built with with_diagnostics=False)",
                ha="center", va="center", color=FG, transform=ax.transAxes)
        return fig

    gens = _series(history, "generation")
    n_settled = _series(history, "n_settled")
    best = np.array([h.get("best_per_diagnostic", np.full(len(labels), np.nan))
                     for h in history], dtype=float)
    med = np.array([h.get("median_per_diagnostic", np.full(len(labels), np.nan))
                    for h in history], dtype=float)

    panels = [(k, t, lg) for k, t, lg in _DIAG_PANELS if k in labels]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(panels):]:
        ax.axis("off")

    for ax, (key, title, log) in zip(axes, panels):
        _style_axes(ax)
        j = labels.index(key)
        # Markers, not just a line: settling time is NaN for any generation
        # where nothing settled, and an isolated finite value between two NaNs
        # draws no line segment at all -- it would silently vanish.
        ax.plot(gens, best[:, j], color=CYCLE[0], linewidth=1.8, marker="o",
                markersize=3, label="best in population")
        if np.isfinite(med[:, j]).any():
            ax.plot(gens, med[:, j], color=FG, linewidth=1.0, alpha=0.55,
                    linestyle="--", label="population median")
        ax.set_xlabel("generation")
        ax.set_ylabel(title)
        col = np.concatenate([best[:, j], med[:, j]])
        col = col[np.isfinite(col) & (col > 0)]
        if log and col.size and col.max() / col.min() > 20:
            ax.set_yscale("log")

        if key == "settling_time_s":
            ax.axhline(0, color=GRID, linewidth=0.6)
            twin = ax.twinx()
            twin.plot(gens, n_settled, color=COLORS["y"], linewidth=1.2,
                      alpha=0.8)
            twin.set_ylabel("individuals that settled", color=COLORS["y"],
                            fontsize=8)
            twin.tick_params(colors=COLORS["y"], labelsize=8)
            twin.set_facecolor("none")
            for s in twin.spines.values():
                s.set_color(GRID)
            if not np.isfinite(best[:, j]).any():
                ax.text(0.5, 0.5, "nothing settled below 1 deg\nwithin the window",
                        transform=ax.transAxes, ha="center", va="center",
                        color=FG, alpha=0.75, fontsize=9)
        ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7,
                  loc="best")

    fig.suptitle(f"Engineering metrics per generation — {controller_type}\n"
                 f"recorded for every individual, but none of these steer the "
                 f"search; 'best' is over the whole population, so it can rise "
                 f"when a good-but-dominated individual is dropped",
                 color=FG, fontsize=10)
    fig.tight_layout()
    return fig


def plot_gain_history(history, controller_type="PD"):
    """Every gain's trajectory across generations.

    One panel per gene. The band is the min-to-max spread of that gain across
    the whole Pareto front at each generation; the line is the front median.
    A band that collapses means the front agrees on that gain -- it is not a
    live trade, and you can quote a single value for it. A band that stays
    wide means the gain is genuinely trading against something, and quoting
    one number for it would be hiding the trade.

    The dashed lines are the search bounds. A median pressed against the top
    one is the pinning signature discussed in tuning/report.py -- expected at
    a Pareto extreme, and worth checking against the actuator there.
    """
    ctype = str(controller_type).strip().upper()
    decode = make_decode(ctype)
    names = gene_labels(ctype)
    bounds = gene_bounds(ctype)

    gens, lo_b, med_b, hi_b = [], [], [], []
    for h in history:
        if "front_genes" not in h:
            continue
        flat = np.array([decoded_to_flat(decode(g), ctype)
                         for g in np.atleast_2d(h["front_genes"])])
        gens.append(h["generation"])
        lo_b.append(flat.min(axis=0))
        med_b.append(np.median(flat, axis=0))
        hi_b.append(flat.max(axis=0))

    if not gens:
        fig, (ax,) = _new_fig(1, figsize=(9, 3.2))
        ax.text(0.5, 0.5, "no front genes recorded", ha="center", va="center",
                color=FG, transform=ax.transAxes)
        return fig

    gens = np.array(gens)
    lo_b, med_b, hi_b = map(np.array, (lo_b, med_b, hi_b))

    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow),
                             sharex=True)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(names):]:
        ax.axis("off")

    for k, (ax, name) in enumerate(zip(axes, names)):
        _style_axes(ax)
        colour = CYCLE[k % len(CYCLE)]
        ax.fill_between(gens, lo_b[:, k], hi_b[:, k], color=colour, alpha=0.25,
                        linewidth=0)
        ax.plot(gens, med_b[:, k], color=colour, linewidth=1.7)
        ax.axhline(bounds[k][0], color=FG, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axhline(bounds[k][1], color=FG, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_yscale("log")
        ax.set_ylim(bounds[k][0] * 0.7, bounds[k][1] * 1.4)
        ax.set_ylabel(name)
        if k >= len(names) - ncol:
            ax.set_xlabel("generation")

    fig.suptitle(f"Gain trajectories — {controller_type}\n"
                 f"band = Pareto-front min–max, line = front median, "
                 f"dotted = search bounds", color=FG)
    fig.tight_layout()
    return fig


REGISTRY = {
    "pareto_projections": plot_pareto_projections,
    "pareto_3d": plot_pareto_3d,
    "convergence": plot_convergence,
    "gain_coordinates": plot_gain_parallel_coordinates,
    "hypervolume": plot_hypervolume,
    "objective_spread": plot_objective_spread,
    "diagnostic_history": plot_diagnostic_history,
    "gain_history": plot_gain_history,
}


def plot_tuning(pop, obj, decode, history, controller_type="PD",
                 save_dir=None, show=True, prefix="ga"):
    """Every GA figure at once. Same contract as viz.single_run.plot():
    returns {name: Figure}, optionally saving PNGs and/or showing."""
    picks = pareto_picks(obj, controller_type)

    figs = {
        "pareto_projections": plot_pareto_projections(obj, controller_type, picks),
        "pareto_3d": plot_pareto_3d(obj, controller_type, picks),
        "convergence": plot_convergence(history, controller_type),
        "gain_coordinates": plot_gain_parallel_coordinates(
            pop, decode, controller_type, picks=picks),
        "hypervolume": plot_hypervolume(history, controller_type),
        "objective_spread": plot_objective_spread(history, controller_type),
        "diagnostic_history": plot_diagnostic_history(history, controller_type),
        "gain_history": plot_gain_history(history, controller_type),
    }

    if save_dir is not None:
        from pathlib import Path
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(sd / f"{prefix}_{name}.png", facecolor=BG, dpi=150,
                        bbox_inches="tight")

    if show:
        plt.show()
    elif save_dir is not None:
        for fig in figs.values():
            plt.close(fig)
    return figs
