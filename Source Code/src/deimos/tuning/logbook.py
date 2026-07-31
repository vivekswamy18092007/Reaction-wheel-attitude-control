"""
The generation-by-generation record, in text.

The GA's history already carries every individual's genes, objectives and
diagnostics for every generation. This module turns that into files you can
open in Excel, grep, or paste into a report -- and a terminal table you can
print against a live checkpoint while the run is still going.

Three outputs, deliberately at different granularities:

  generations_summary.csv   one row per GENERATION. Best and median of every
                            metric, how many individuals settled, hypervolume,
                            wall time. This is the file you plot from and the
                            one that answers "what did the search actually
                            buy between generation 20 and 80".

  generations_front.csv     one row per PARETO-FRONT INDIVIDUAL per
                            generation, with its absolute gains and its
                            metrics. This is the file that answers "what were
                            the gains at generation 40" -- the question a
                            summary cannot answer, because a front is a set of
                            trade-offs, not one design.

  generations_full.csv      one row per INDIVIDUAL per generation, front or
                            not. Large (pop_size x generations rows) but it is
                            the only record of what the search REJECTED, which
                            is what you need to say anything about the shape of
                            the search space rather than just its optimum.

Every gain is written ABSOLUTE (Kp, Kd, Ki as the controller uses them), not
as the internal [0,1] gene or the Ki/Kp ratio the search operates on, so a row
can be pasted into a config without a conversion step.
"""

from __future__ import annotations

import csv
import pickle
import time
from pathlib import Path

import numpy as np

from deimos.tuning import objectives as O


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _gain_names(ctype):
    """Column names for the decoded gains, in the order _absolute_gains
    produces them. Delegated to objectives so a new controller type declares
    its gene layout in exactly one place -- WIE's three scalars (k, d, ki)
    are not a per-axis triple and hardcoding x/y/z here would have silently
    mislabelled them."""
    return O.absolute_gain_labels(ctype)


def _absolute_gains(genes, decode):
    """Genes -> the physical gains, in the order _gain_names gives."""
    return np.array([np.concatenate(decode(g)) for g in np.atleast_2d(genes)])


def _diag_labels(history):
    for h in history:
        if "diagnostic_labels" in h:
            return list(h["diagnostic_labels"])
    return []


def generation_rows(history, controller_type="PD", front_only=True):
    """One dict per individual per generation.

    front_only=True uses front_genes/front_objectives, which NSGA2 always
    stores. front_only=False needs store_full_population=True (the default)
    and yields the whole population including everything that was rejected.
    """
    ctype = O._normalize_type(controller_type)
    decode = O.make_decode(ctype)
    gain_names = _gain_names(ctype)
    obj_labels = list(O.OBJECTIVE_LABELS[ctype])
    diag_labels = _diag_labels(history)

    gkey = "front_genes" if front_only else "population_genes"
    okey = "front_objectives" if front_only else "population_objectives"
    dkey = "front_diagnostics" if front_only else "population_diagnostics"

    rows = []
    for h in history:
        if gkey not in h:
            continue
        genes = np.atleast_2d(h[gkey])
        obj = np.atleast_2d(h[okey])
        diag = np.atleast_2d(h[dkey]) if dkey in h else None
        gains = _absolute_gains(genes, decode)

        for i in range(len(genes)):
            row = {"generation": int(h["generation"]), "individual": i,
                   "on_front": bool(front_only)}
            row.update({name: float(gains[i, j])
                        for j, name in enumerate(gain_names)})
            row.update({lab: float(obj[i, j]) for j, lab in enumerate(obj_labels)})
            if diag is not None and i < len(diag):
                row.update({lab: float(diag[i, j])
                            for j, lab in enumerate(diag_labels)})
            rows.append(row)
    return rows


def summary_rows(history, controller_type="PD"):
    """One dict per generation: best and median of every metric.

    Medians are over the whole population where it was stored, and nan-aware:
    a candidate that never settled carries NaN settling time by design, and a
    plain median would report NaN for the entire generation because of it.
    """
    ctype = O._normalize_type(controller_type)
    obj_labels = list(O.OBJECTIVE_LABELS[ctype])
    diag_labels = _diag_labels(history)

    rows = []
    for h in history:
        row = {
            "generation": int(h["generation"]),
            "front_size": int(h["front_size"]),
            "hypervolume": float(h.get("hypervolume", np.nan)),
            "hv_relative_gain": float(h.get("hv_relative_gain", np.nan)),
            "n_settled": int(h.get("n_settled", -1)),
            "n_evaluations": int(h.get("n_evaluations", -1)),
            "elapsed_s": float(h.get("elapsed_s", np.nan)),
        }
        for j, lab in enumerate(obj_labels):
            row[f"best_{lab}"] = float(h["best_per_objective"][j])
            if "median_per_objective" in h:
                row[f"median_{lab}"] = float(h["median_per_objective"][j])
        for j, lab in enumerate(diag_labels):
            if "best_per_diagnostic" in h:
                row[f"best_{lab}"] = float(h["best_per_diagnostic"][j])
            if "median_per_diagnostic" in h:
                row[f"median_{lab}"] = float(h["median_per_diagnostic"][j])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def write_logbook(out_dir, history, controller_type="PD", full=True):
    """All three CSVs. Returns {name: path}."""
    out_dir = Path(out_dir)
    written = {}
    written["summary"] = _write_csv(out_dir / "generations_summary.csv",
                                    summary_rows(history, controller_type))
    written["front"] = _write_csv(
        out_dir / "generations_front.csv",
        generation_rows(history, controller_type, front_only=True))
    if full:
        try:
            written["full"] = _write_csv(
                out_dir / "generations_full.csv",
                generation_rows(history, controller_type, front_only=False))
        except Exception:
            # store_full_population was off, or this is an old checkpoint --
            # the front-level record is still complete, so this is not fatal.
            written["full"] = None
    return {k: v for k, v in written.items() if v is not None}


# --------------------------------------------------------------------------
# terminal view
# --------------------------------------------------------------------------

_TERMINAL_COLUMNS = [
    ("gen", "generation", "{:>4d}"),
    ("front", "front_size", "{:>5d}"),
    ("settled", "n_settled", "{:>7d}"),
    ("HV", "hypervolume", "{:>9.6f}"),
    ("settle (s)", "best_settling_time_s", "{:>10.2f}"),
    ("sse (deg)", "best_steady_state_error_deg", "{:>10.4f}"),
    ("effort", "best_control_effort_Nms", "{:>11.4e}"),
    ("ITAE", "best_itae_deg_s2", "{:>11.4e}"),
    ("sat", "best_saturation_fraction", "{:>7.3f}"),
    ("min", "elapsed_s", "{:>6.1f}"),
]


def format_table(history, controller_type="PD", last=None) -> str:
    """The per-generation table, as text. `last` limits to the final N rows."""
    rows = summary_rows(history, controller_type)
    if last is not None:
        rows = rows[-last:]
    if not rows:
        return "(no generations recorded yet)"

    header = "".join(f"{h:>{max(len(h), 6)}}  " for h, _, _ in _TERMINAL_COLUMNS)
    out = [header, "-" * len(header)]
    for r in rows:
        cells = []
        for head, key, fmt in _TERMINAL_COLUMNS:
            v = r.get(key, float("nan"))
            if key == "elapsed_s":
                v = v / 60.0
            try:
                if "d}" in fmt:
                    cells.append(fmt.format(int(v)))
                elif not np.isfinite(v):
                    cells.append(f"{'--':>{max(len(head), 6)}}")
                else:
                    cells.append(fmt.format(v))
            except (ValueError, TypeError, OverflowError):
                cells.append(f"{'--':>{max(len(head), 6)}}")
        out.append("".join(f"{c}  " for c in cells))

    note = ("\nsettle/sse/effort/ITAE are the BEST value anything in the "
            "population reached that generation.\n'settled' counts "
            "individuals getting below 1 deg and staying there; '--' means "
            "none did.\nPer-front-individual gains and metrics are in "
            "generations_front.csv.")
    return "\n".join(out) + note


def format_front(history, controller_type="PD", generation=-1) -> str:
    """The gains and metrics of one generation's whole Pareto front."""
    ctype = O._normalize_type(controller_type)
    rows = [r for r in generation_rows(history, ctype, front_only=True)]
    if not rows:
        return "(no front recorded yet)"
    gens = sorted({r["generation"] for r in rows})
    target = gens[generation] if isinstance(generation, int) and generation < 0 \
        else generation
    rows = [r for r in rows if r["generation"] == target]

    gain_names = _gain_names(ctype)
    out = [f"=== generation {target}: Pareto front ({len(rows)} solutions) ==="]
    for r in rows:
        line = "  " + _format_gain_row(r, gain_names, ctype)
        st = r.get("settling_time_s", float("nan"))
        line += ("\n      settle=" + ("did not settle" if not np.isfinite(st)
                                      else f"{st:.2f} s")
                 + f"   sse={r.get('steady_state_error_deg', float('nan')):.4f} deg"
                 + f"   effort={r.get('control_effort_Nms', float('nan')):.4e} N m s"
                 + f"   sat={100*r.get('saturation_fraction', float('nan')):.2f}%")
        if ctype == "WIE":
            # The number that says whether the eigenaxis property survived
            # the wheel limit and the disturbance -- meaningless to omit from
            # a front dump for the eigenaxis law specifically.
            line += (f"   eigax_dev="
                     f"{r.get('mean_eigenaxis_deviation_deg', float('nan')):.2f} deg")
        out.append(line)
    return "\n".join(out)


def _format_gain_row(row, gain_names, ctype):
    """Gains of one CSV row as text. WIE's three genes are scalars (k, d, ki),
    not per-axis triples, so they are printed as scalars -- printing them as
    `k=[x y z]` would suggest a per-axis freedom the eigenaxis law forbids."""
    if O._normalize_type(ctype) == "WIE":
        return "  ".join(f"{n}={row[n]:.4e}" for n in gain_names)
    parts = []
    for name, start in (("Kp", 0), ("Kd", 3), ("Ki", 6)):
        if start >= len(gain_names):
            break
        v = [row[n] for n in gain_names[start:start + 3]]
        parts.append(f"{name}=[{v[0]:.4e} {v[1]:.4e} {v[2]:.4e}]")
    return "  ".join(parts)


# --------------------------------------------------------------------------
# one line per Pareto "pick" (the fastest/cheapest/knee summary), per
# generation -- the compact view worth watching scroll by live
# --------------------------------------------------------------------------

def format_picks(history, controller_type="PD", generation=-1) -> str:
    """One block: the handful of representative Pareto picks (best_tracking /
    cheapest_effort / least_saturated or most_accurate / knee -- whatever
    pareto_picks() returns for this controller type) with their gains and
    metrics. This is the compact per-generation view, versus format_front's
    full front dump.

    Prefers the recorded DIAGNOSTICS (real settling time, steady-state error,
    control effort, saturation -- see tuning/objectives.py) when the
    checkpoint has them. Falls back to the raw objective tuple for an older
    checkpoint recorded before diagnostics existed, so this still works
    against a pre-upgrade run.
    """
    ctype = O._normalize_type(controller_type)
    decode = O.make_decode(ctype)
    obj_labels = list(O.OBJECTIVE_LABELS[ctype])

    fronts = [h for h in history if "front_genes" in h]
    if not fronts:
        return "(no front recorded yet)"
    entry = fronts[generation] if isinstance(generation, int) and generation < 0 \
        else next((h for h in fronts if h["generation"] == generation), fronts[-1])

    genes = np.atleast_2d(entry["front_genes"])
    obj = np.atleast_2d(entry["front_objectives"])
    diag = (np.atleast_2d(entry["front_diagnostics"])
            if "front_diagnostics" in entry else None)
    diag_labels = entry.get("diagnostic_labels")

    picks = O.pareto_picks(obj, ctype)
    out = [f"\n=== generation {entry['generation']}  "
           f"(front size {entry['front_size']}, {len(fronts)} recorded) ==="]

    for label, idx in picks.items():
        decoded = decode(genes[idx])
        if ctype == "WIE":
            k, d, ki = (float(np.asarray(b).ravel()[0]) for b in decoded)
            gain_str = f"k={k:.6g}  d={d:.6g}  ki={ki:.6g}"
        else:
            gain_str = (f"Kp={np.round(decoded[0], 6)}  "
                        f"Kd={np.round(decoded[1], 6)}")
            if ctype == "PID":
                gain_str += f"  Ki={np.round(decoded[2], 6)}"

        if diag is not None and diag_labels:
            d = dict(zip(diag_labels, diag[idx]))
            st = d.get("settling_time_s", float("nan"))
            metric_str = (
                ("did not settle" if not np.isfinite(st) else f"settle={st:.2f}s")
                + f"  sse={d.get('steady_state_error_deg', float('nan')):.4g} deg"
                + f"  effort={d.get('control_effort_Nms', float('nan')):.4g} N m s"
                + f"  sat={100*d.get('saturation_fraction', float('nan')):.2f}%"
            )
        else:
            metric_str = ", ".join(f"{lab}={v:.5g}" for lab, v in zip(obj_labels, obj[idx]))

        out.append(f"  [{label:16s}] {gain_str}  ({metric_str})")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI: works against a live checkpoint, mid-run
# --------------------------------------------------------------------------

def _load(path):
    """Tolerates the file not existing yet and a transient read race with
    NSGA2._checkpoint()'s atomic os.replace -- a watch loop must never crash
    on either."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, PermissionError):
        return None


def watch(path, controller_type="PID", interval=10.0, mode="picks"):
    """Poll `path` and print a new block every time a generation lands --
    the live, scrolling view. Ctrl+C to stop; the GA process is untouched.

    mode: "picks" (compact -- one line per representative solution, this is
        the results.txt style) or "front" (every solution on the front, with
        settle/sse/effort/sat for each).
    """
    print(f"watching {path} every {interval:.0f}s (Ctrl+C to stop)\n")
    last_gen = None
    try:
        while True:
            data = _load(path)
            if data is not None and data.get("history"):
                latest = data["history"][-1]
                if "front_genes" in latest and latest["generation"] != last_gen:
                    fn = format_front if mode == "front" else format_picks
                    print(fn(data["history"], controller_type, generation=-1),
                          flush=True)
                    last_gen = latest["generation"]
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped watching (the GA run, if still active, is unaffected)")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Per-generation gains and metrics from a GA checkpoint. "
                    "Read-only and safe to run while the GA is still going.")
    ap.add_argument("checkpoint", help="a checkpoint.pkl or final.pkl")
    ap.add_argument("--controller-type", default="PID", choices=["PD", "PID", "WIE"])
    ap.add_argument("--last", type=int, default=25,
                    help="show only the last N generations (0 = all)")
    ap.add_argument("--front", type=int, default=None, metavar="GEN",
                    help="also print the full Pareto front of this generation "
                         "(-1 = latest)")
    ap.add_argument("--picks", type=int, default=None, metavar="GEN",
                    help="print the compact pick summary for this generation "
                         "(-1 = latest) -- the fastest/cheapest/knee view")
    ap.add_argument("--csv", default=None, metavar="DIR",
                    help="write the three CSVs into this directory")
    ap.add_argument("--watch", action="store_true",
                    help="keep the process alive and print a new block every "
                         "time a generation lands, instead of exiting once")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="seconds between checkpoint polls in --watch mode")
    ap.add_argument("--watch-mode", default="picks", choices=["picks", "front"],
                    help="--watch prints the compact picks (default) or the "
                         "whole front each generation")
    args = ap.parse_args(argv)

    if args.watch:
        watch(args.checkpoint, args.controller_type, args.interval, args.watch_mode)
        return

    import pickle as _pickle
    with open(args.checkpoint, "rb") as f:
        data = _pickle.load(f)
    history = data["history"]
    ctype = args.controller_type

    print(format_table(history, ctype, last=(args.last or None)))
    if args.picks is not None:
        print(format_picks(history, ctype, generation=args.picks))
    if args.front is not None:
        print()
        print(format_front(history, ctype, generation=args.front))
    if args.csv:
        written = write_logbook(args.csv, history, ctype)
        print("\nwrote:")
        for name, path in written.items():
            print(f"  {name:<8} {path}")


if __name__ == "__main__":
    main()
