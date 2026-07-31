"""
Live dashboard for a running NSGA-II tuning job.

Run this in a SEPARATE terminal/process from the GA itself:

    python overnight_tune.py                     # terminal 1
    python -m deimos.tuning.monitor_ga runs/overnight_latest    # terminal 2

It polls the checkpoint files NSGA2._checkpoint() writes every generation and
redraws. It never touches the GA process, imports nothing from it, and if you
close the window the optimization keeps running untouched -- this is a
read-only viewer over pickles on disk.

Point it at either:
  * a single checkpoint .pkl, or
  * a run directory containing seed*/checkpoint.pkl, in which case it tracks
    the seed that is currently advancing and overlays every seed's
    hypervolume trajectory in the cross-seed panel.

Panels (15):
    hypervolume + the stall rule that will stop the run
    Pareto front size
    live status text (generation, ETA, deadline, evaluations, s/gen, and the
        best settling time / steady-state error / effort reached so far)
    best vs population median, one panel per objective          (3)
    settling time (+ how many individuals settled), steady-state error and
        control effort per generation -- recorded, not optimized  (3)
    Pareto front pairwise projections                           (3)
    gain parallel coordinates for the current front
    gain trajectories: each gain's front median vs generation
    cross-seed hypervolume comparison

Headless use (no display, e.g. leaving it running overnight and checking a
PNG from your phone):

    python -m deimos.tuning.monitor_ga runs/overnight_latest \
        --save runs/overnight_latest/live_monitor.png --no-window
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Optional: if the deimos package is importable, use its labels and gain
# decoding. If not (monitoring from a bare checkout), fall back to generic
# labels so the monitor still runs standalone.
try:
    from deimos.tuning.objectives import (
        OBJECTIVE_LABELS, decoded_to_flat, gene_bounds, gene_labels, make_decode,
    )
    _HAVE_DEIMOS = True
except ImportError:                                   # pragma: no cover
    OBJECTIVE_LABELS = None
    _HAVE_DEIMOS = False

BG = "#0d1117"
FG = "#e6e6e6"
GRID = "#3a3f47"
MAIN = "#3f7cac"
CYCLE = ["#3f7cac", "#ff4b4b", "#4bff7a", "#ffb84b", "#b84bff", "#4bffe6"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _load(path):
    """Tolerates the file not existing yet (the run hasn't reached generation
    0) and a transient read failure. _checkpoint uses os.replace so a
    half-written file is never visible at `path`, but a poll loop should
    never crash on a race regardless."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, PermissionError):
        return None


def _discover(target: Path):
    """-> (list of (label, path)) for every checkpoint we should watch."""
    if target.is_file():
        return [(target.stem, target)]
    found = sorted(target.glob("seed*/checkpoint.pkl"))
    if not found:
        found = sorted(target.glob("**/checkpoint.pkl"))
    return [(p.parent.name, p) for p in found]


def _labels(n_obj, controller_type):
    if OBJECTIVE_LABELS is not None and controller_type in OBJECTIVE_LABELS:
        labs = list(OBJECTIVE_LABELS[controller_type])
        if len(labs) == n_obj:
            return labs
    return [f"objective {i}" for i in range(n_obj)]


def _series(history, key):
    return np.array([h.get(key, np.nan) for h in history], dtype=float)


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(True, alpha=0.25, color=GRID)
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, color=FG, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=FG, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=8)


def _build_figure():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(19, 13.5))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(5, 6, figure=fig, hspace=0.45, wspace=0.34,
                  left=0.05, right=0.98, top=0.94, bottom=0.05)

    ax = {
        "hv":      fig.add_subplot(gs[0, 0:2]),
        "front":   fig.add_subplot(gs[0, 2:4]),
        "status":  fig.add_subplot(gs[0, 4:6]),
        "obj0":    fig.add_subplot(gs[1, 0:2]),
        "obj1":    fig.add_subplot(gs[1, 2:4]),
        "obj2":    fig.add_subplot(gs[1, 4:6]),
        # the engineering row: the numbers a human reads, none of which
        # steer the search
        "settle":  fig.add_subplot(gs[2, 0:2]),
        "sse":     fig.add_subplot(gs[2, 2:4]),
        "effort":  fig.add_subplot(gs[2, 4:6]),
        "p01":     fig.add_subplot(gs[3, 0:2]),
        "p02":     fig.add_subplot(gs[3, 2:4]),
        "p12":     fig.add_subplot(gs[3, 4:6]),
        "par":     fig.add_subplot(gs[4, 0:3]),
        "gains":   fig.add_subplot(gs[4, 3:5]),
        "seeds":   fig.add_subplot(gs[4, 5:6]),
    }
    try:
        fig.canvas.manager.set_window_title("DEIMoS — NSGA-II live monitor")
    except Exception:
        pass
    return fig, ax


def _draw(fig, ax, data, all_series, ctype, active_label, deadline_note=""):
    """Full redraw of every panel.

    Deliberately clear-and-redraw rather than the set_data() dance the old
    monitor used. A generation here takes tens of seconds and the poll
    interval is seconds, so the redraw cost is irrelevant, and blitting 11
    heterogeneous panels (scatter, bars, text, parallel coordinates whose
    line COUNT changes as the front grows) is a large amount of fragile
    bookkeeping for no measurable gain.
    """
    history = data["history"]
    obj = np.asarray(data["obj"], dtype=float)
    pop = np.asarray(data["pop"], dtype=float)

    gens = _series(history, "generation")
    hv = _series(history, "hypervolume")
    gain = _series(history, "hv_relative_gain")
    front_size = _series(history, "front_size")
    elapsed = _series(history, "elapsed_s")
    n_evals = _series(history, "n_evaluations")
    best = np.array([h["best_per_objective"] for h in history], dtype=float)
    med = np.array([h.get("median_per_objective",
                          np.full(best.shape[1], np.nan)) for h in history],
                   dtype=float)
    n_obj = best.shape[1]
    labels = _labels(n_obj, ctype)
    latest = history[-1]

    for a in ax.values():
        a.clear()

    # --- hypervolume ---------------------------------------------------
    _style(ax["hv"], "hypervolume (fraction of worst-case box)", "generation")
    ax["hv"].plot(gens, hv, color=MAIN, linewidth=1.8)
    if np.isfinite(hv).any():
        ax["hv"].scatter([gens[-1]], [hv[-1]], s=45, color="#ffffff", zorder=5)
        ax["hv"].annotate(f"{hv[-1]:.6f}", (gens[-1], hv[-1]),
                          textcoords="offset points", xytext=(-8, 10),
                          color=FG, fontsize=8, ha="right")
    g = gain[-1] if len(gain) else np.nan
    ax["hv"].text(0.02, 0.06,
                  ("stall gain: n/a (warming up)" if not np.isfinite(g)
                   else f"trailing relative gain: {g:.2e}"),
                  transform=ax["hv"].transAxes, color=FG, fontsize=8)

    # --- front size ----------------------------------------------------
    _style(ax["front"], "Pareto front size", "generation")
    ax["front"].plot(gens, front_size, color="#4bff7a", linewidth=1.6)
    ax["front"].text(0.02, 0.06,
                     "front == population is normal: once everything is\n"
                     "mutually non-dominated, crowding distance takes over",
                     transform=ax["front"].transAxes, color=FG, fontsize=7,
                     alpha=0.75)

    # --- status text ---------------------------------------------------
    ax["status"].set_facecolor(BG)
    ax["status"].axis("off")
    per_gen = (elapsed[-1] / max(gens[-1], 1)) if np.isfinite(elapsed[-1]) else np.nan
    lines = [
        f"watching      {active_label}",
        f"generation    {int(latest['generation'])}",
        f"front size    {int(latest['front_size'])}",
        f"evaluations   {int(n_evals[-1]) if np.isfinite(n_evals[-1]) else 0}",
        f"elapsed       {elapsed[-1]/60:.1f} min" if np.isfinite(elapsed[-1]) else "elapsed       n/a",
        f"per gen       {per_gen:.1f} s" if np.isfinite(per_gen) else "per gen       n/a",
        "",
        "best so far:",
    ]
    for i, lab in enumerate(labels):
        short = lab.split(" (")[0][:16]
        lines.append(f"  {short:<16} {best[-1, i]:.4e}")

    # Short labels and a tight width: this panel shares a row with two plots,
    # and a long monospace line silently overruns into the neighbouring axes
    # rather than wrapping.
    _SHORT = {"settling_time_s": "settle (s)",
              "steady_state_error_deg": "sse (deg)",
              "overshoot_deg": "overshoot (deg)",
              "saturation_fraction": "saturation",
              "peak_wheel_torque_Nm": "peak wheel tau"}
    if "diagnostic_labels" in latest and "best_per_diagnostic" in latest:
        lines += ["", "best reported:"]
        for lab, v in zip(latest["diagnostic_labels"], latest["best_per_diagnostic"]):
            if lab in _SHORT:
                txt = "none settled" if not np.isfinite(v) else f"{v:.4g}"
                lines.append(f"  {_SHORT[lab]:<16} {txt}")
        lines.append(f"  {'settled':<16} {latest.get('n_settled', 0)} / {len(pop)}")

    if deadline_note:
        lines += ["", deadline_note]
    if data.get("stop_reason"):
        lines += ["", f"stopped: {data['stop_reason']}"]
    ax["status"].text(0.0, 0.99, "\n".join(lines), transform=ax["status"].transAxes,
                      color=FG, fontsize=8, va="top", family="monospace",
                      clip_on=True)

    # --- per-objective best vs median ----------------------------------
    for i in range(min(3, n_obj)):
        a = ax[f"obj{i}"]
        _style(a, labels[i], "generation")
        a.plot(gens, best[:, i], color=CYCLE[i % len(CYCLE)], linewidth=1.6,
               label="front best")
        if np.isfinite(med[:, i]).any():
            a.plot(gens, med[:, i], color=FG, linewidth=1.0, alpha=0.55,
                   linestyle="--", label="pop median")
        stack = np.concatenate([best[:, i], med[:, i]])
        stack = stack[np.isfinite(stack) & (stack > 0)]
        if stack.size and stack.max() / stack.min() > 20:
            a.set_yscale("log")
        a.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)

    # --- engineering metrics (recorded, not optimized) ------------------
    diag_labels = []
    for h in history:
        if "diagnostic_labels" in h:
            diag_labels = list(h["diagnostic_labels"])
            break

    diag_panels = [("settle", "settling_time_s", "settling time (s)", False),
                   ("sse", "steady_state_error_deg", "steady-state error (deg)", True),
                   ("effort", "control_effort_Nms", "control effort (N m s)", True)]

    if diag_labels:
        d_best = np.array([h.get("best_per_diagnostic",
                                 np.full(len(diag_labels), np.nan))
                           for h in history], dtype=float)
        d_med = np.array([h.get("median_per_diagnostic",
                                np.full(len(diag_labels), np.nan))
                          for h in history], dtype=float)
        n_settled = _series(history, "n_settled")

        for key, dkey, title, log in diag_panels:
            a = ax[key]
            _style(a, title, "generation")
            if dkey not in diag_labels:
                a.axis("off")
                continue
            j = diag_labels.index(dkey)
            # markers: settling time is NaN whenever nothing settled, and an
            # isolated finite value draws no line segment
            a.plot(gens, d_best[:, j], color=CYCLE[3], linewidth=1.7,
                   marker="o", markersize=3, label="best in population")
            if np.isfinite(d_med[:, j]).any():
                a.plot(gens, d_med[:, j], color=FG, linewidth=1.0, alpha=0.55,
                       linestyle="--", label="pop median")
            col = np.concatenate([d_best[:, j], d_med[:, j]])
            col = col[np.isfinite(col) & (col > 0)]
            if log and col.size and col.max() / col.min() > 20:
                a.set_yscale("log")

            if dkey == "settling_time_s":
                twin = a.twinx()
                twin.plot(gens, n_settled, color="#4bff7a", linewidth=1.2,
                          alpha=0.85)
                twin.set_ylabel("num settled", color="#4bff7a", fontsize=8)
                twin.tick_params(colors="#4bff7a", labelsize=7)
                twin.set_facecolor("none")
                for s in twin.spines.values():
                    s.set_color(GRID)
                if not np.isfinite(d_best[:, j]).any():
                    a.text(0.5, 0.5, "nothing has settled below 1 deg yet",
                           transform=a.transAxes, ha="center", va="center",
                           color=FG, alpha=0.75, fontsize=9)
            a.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)
    else:
        for key, _, title, _ in diag_panels:
            _style(ax[key], title)
            ax[key].text(0.5, 0.5, "no diagnostics in this checkpoint",
                         transform=ax[key].transAxes, ha="center", color=FG,
                         fontsize=8, alpha=0.7)

    # --- pareto projections --------------------------------------------
    for key, (i, j) in {"p01": (0, 1), "p02": (0, 2), "p12": (1, 2)}.items():
        a = ax[key]
        if n_obj <= max(i, j):
            a.axis("off")
            continue
        _style(a, None, labels[i], labels[j])
        a.scatter(obj[:, i], obj[:, j], s=30, color=MAIN, alpha=0.6,
                  edgecolors="none")
        for k, m in enumerate(np.argmin(obj, axis=0)):
            a.scatter([obj[m, i]], [obj[m, j]], s=110, marker="*",
                      color=CYCLE[(k + 1) % len(CYCLE)], edgecolors=FG,
                      linewidths=0.5, zorder=5)
        for axis_i, setter in ((i, a.set_xscale), (j, a.set_yscale)):
            col = obj[:, axis_i]
            col = col[np.isfinite(col) & (col > 0)]
            if col.size and col.max() / col.min() > 50:
                setter("log")

    # --- parallel coordinates -------------------------------------------
    a = ax["par"]
    _style(a, "current front in gain space  (0 = lower search bound, 1 = upper)",
           None, "position within bounds")
    if _HAVE_DEIMOS and pop.shape[1] in (6, 9):
        try:
            decode = make_decode(ctype)
            names = gene_labels(ctype)
            bounds = gene_bounds(ctype)
            flat = np.array([decoded_to_flat(decode(ind), ctype) for ind in pop])
            lo = np.array([np.log10(b[0]) for b in bounds])
            hi = np.array([np.log10(b[1]) for b in bounds])
            norm = (np.log10(flat) - lo) / (hi - lo)
            x = np.arange(len(names))
            for row in norm:
                a.plot(x, row, color=MAIN, alpha=0.20, linewidth=1.0)
            for k, m in enumerate([np.argmin(obj[:, c]) for c in range(n_obj)]):
                a.plot(x, norm[m], color=CYCLE[(k + 1) % len(CYCLE)],
                       linewidth=1.8, marker="o", markersize=3,
                       label=f"best {labels[k]}")
            for xi in x:
                a.axvline(xi, color=GRID, linewidth=0.7, alpha=0.6)
            a.set_xticks(x)
            a.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
            a.set_ylim(-0.05, 1.05)
            a.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7,
                     ncol=3)
        except Exception as e:                        # pragma: no cover
            a.text(0.5, 0.5, f"gain decode unavailable: {e}", color=FG,
                   ha="center", transform=a.transAxes, fontsize=8)
    else:
        a.text(0.5, 0.5, "gain decoding needs the deimos package importable",
               color=FG, ha="center", transform=a.transAxes, fontsize=8)

    # --- gain trajectories ------------------------------------------------
    a = ax["gains"]
    _style(a, "gain trajectories (front median, normalized to bounds)",
           "generation", "position within bounds")
    if _HAVE_DEIMOS and pop.shape[1] in (6, 9):
        try:
            decode = make_decode(ctype)
            names = gene_labels(ctype)
            bounds = gene_bounds(ctype)
            lo = np.array([np.log10(b[0]) for b in bounds])
            hi = np.array([np.log10(b[1]) for b in bounds])
            tg, tmed = [], []
            for h in history:
                if "front_genes" not in h:
                    continue
                flat = np.array([decoded_to_flat(decode(g), ctype)
                                 for g in np.atleast_2d(h["front_genes"])])
                tg.append(h["generation"])
                tmed.append(np.median((np.log10(flat) - lo) / (hi - lo), axis=0))
            if tg:
                tmed = np.array(tmed)
                for k, name in enumerate(names):
                    a.plot(tg, tmed[:, k], linewidth=1.3,
                           color=CYCLE[k % len(CYCLE)],
                           linestyle=["-", "--", ":"][k // len(CYCLE) % 3],
                           label=name)
                a.set_ylim(-0.05, 1.05)
                a.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID,
                         fontsize=6, ncol=3)
        except Exception as e:                        # pragma: no cover
            a.text(0.5, 0.5, f"unavailable: {e}", color=FG, ha="center",
                   transform=a.transAxes, fontsize=8)

    # --- cross-seed ------------------------------------------------------
    a = ax["seeds"]
    _style(a, "hypervolume by seed", "generation")
    for k, (label, series) in enumerate(all_series.items()):
        if series is None or not len(series[0]):
            continue
        a.plot(series[0], series[1], color=CYCLE[k % len(CYCLE)], linewidth=1.5,
               label=label, alpha=1.0 if label == active_label else 0.65)
    if all_series:
        a.legend(facecolor=BG, labelcolor=FG, edgecolor=GRID, fontsize=7)

    fig.suptitle(
        f"DEIMoS — NSGA-II {ctype} gain search   |   {active_label}   |   "
        f"gen {int(latest['generation'])}   |   front {int(latest['front_size'])}"
        + (f"   |   HV {hv[-1]:.6f}" if np.isfinite(hv[-1]) else ""),
        color=FG, fontsize=12)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target",
                    help="a checkpoint .pkl, or a run directory containing "
                         "seed*/checkpoint.pkl")
    ap.add_argument("--controller-type", default="PID", choices=["PD", "PID", "WIE"])
    ap.add_argument("--poll-interval", type=float, default=5.0,
                    help="seconds between checking the checkpoints for updates")
    ap.add_argument("--save", default=None,
                    help="also write the dashboard to this PNG on every refresh")
    ap.add_argument("--no-window", action="store_true",
                    help="headless: render to --save only, no GUI window")
    ap.add_argument("--once", action="store_true",
                    help="render a single frame and exit")
    args = ap.parse_args(argv)

    import matplotlib
    if args.no_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(args.target)
    print(f"watching {target} (polling every {args.poll_interval}s, Ctrl+C to stop)")

    if not args.no_window:
        plt.ion()
    fig = ax = None
    last_signature = None

    while True:
        sources = _discover(target)
        if not sources:
            print(f"\rwaiting for a checkpoint under {target} ...", end="", flush=True)
            time.sleep(args.poll_interval)
            continue

        loaded = [(label, _load(p)) for label, p in sources]
        loaded = [(label, d) for label, d in loaded
                  if d is not None and d.get("history")]
        if not loaded:
            print(f"\rwaiting for the first generation to be written ...",
                  end="", flush=True)
            time.sleep(args.poll_interval)
            continue

        # The "active" run is whichever checkpoint has advanced furthest in
        # wall time -- with sequential seeds that is the one currently going.
        def _recency(item):
            h = item[1]["history"][-1]
            return (h.get("generation", 0), h.get("elapsed_s", 0.0))

        active_label, data = max(loaded, key=_recency)
        all_series = {label: (_series(d["history"], "generation"),
                              _series(d["history"], "hypervolume"))
                      for label, d in loaded}

        signature = tuple((label, len(d["history"])) for label, d in loaded)
        if signature == last_signature and not args.once:
            if fig is not None and not args.no_window:
                plt.pause(args.poll_interval)
            else:
                time.sleep(args.poll_interval)
            continue
        last_signature = signature

        # A sibling status.json, if the driver wrote one, carries the deadline
        # -- the single most useful number to see at 2 a.m.
        deadline_note = ""
        status_file = Path(sources[0][1]).parent / "status.json"
        if status_file.exists():
            try:
                import json
                st = json.loads(status_file.read_text())
                if st.get("deadline_in_s") is not None:
                    deadline_note = (f"seed deadline in "
                                     f"{st['deadline_in_s']/60:.0f} min")
            except Exception:
                pass

        if fig is None:
            fig, ax = _build_figure()
        _draw(fig, ax, data, all_series, args.controller_type, active_label,
              deadline_note)

        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, facecolor=BG, dpi=110, bbox_inches="tight")

        if not args.no_window:
            fig.canvas.draw_idle()
            plt.pause(0.01)      # lets the GUI event loop actually paint

        h = data["history"][-1]
        print(f"\r{active_label}  gen {h['generation']:4d}  "
              f"front={h['front_size']:3d}  HV={h.get('hypervolume', float('nan')):.6f}",
              end="", flush=True)

        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nmonitor stopped (the GA run, if still active, is unaffected)")
        sys.exit(0)
