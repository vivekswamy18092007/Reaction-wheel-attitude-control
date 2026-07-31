"""
Shared dark-theme styling for every figure in single_run.py / comparison.py.
"""

from __future__ import annotations

import numpy as np

BG = "#0d1117"
FG = "#e6e6e6"
GRID = "#3a3f47"
COLORS = {"x": "#ff4b4b", "y": "#4bff7a", "z": "#4b9bff", "main": "#3f7cac"}
# Per-wheel traces, chosen to exclude red so the red limit lines stay legible.
WHEEL_CYCLE = ["#4b9bff", "#4bff7a", "#ffb84b", "#b84bff", "#4bffe6", "#e6e6e6"]
# Comparison-plot per-run cycle.
CYCLE = ["#3f7cac", "#ff4b4b", "#4bff7a", "#ffb84b", "#b84bff", "#4bffe6"]


def _style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.3, color=GRID)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral, same length as y, starting at 0.
    Written out rather than imported: numpy has no cumulative trapezoid and
    pulling in scipy for six lines isn't worth a new dependency."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _new_fig(nrows=1, figsize=(10, 4)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, 1, figsize=(figsize[0], figsize[1] * nrows), sharex=True)
    fig.patch.set_facecolor(BG)
    axes = np.atleast_1d(axes)
    for ax in axes:
        _style_axes(ax)
    return fig, axes
