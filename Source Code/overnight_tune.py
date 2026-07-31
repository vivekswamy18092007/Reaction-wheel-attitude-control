"""
Unattended overnight NSGA-II gain search for DEIMoS.

    python overnight_tune.py                    # 8 hours, 3 seeds, PID, 2 scenarios
    python overnight_tune.py --hours 6 --seeds 0 1
    python overnight_tune.py --type PD --scenarios configs/scenarios/slew_40_30_25.yaml

Watch it from a second terminal:

    python -m deimos.tuning.monitor_ga runs/overnight

Design decisions that matter for an unattended run:

  A REAL WALL-CLOCK BUDGET.  Every seed gets a hard deadline and the GA stops
  at the next generation boundary once it is reached. The budget is
  re-divided over the seeds that remain after each one finishes, so a seed
  that plateaus early hands its unused time to the next seed instead of
  wasting it. You get N finished seeds, not one truncated one.

  A PREFLIGHT.  Before committing eight hours, one candidate is evaluated end
  to end on the real configs. That catches a broken config, a missing
  disturbance, or an import error in thirty seconds rather than at 7 a.m.
  The measured evaluation time is then used to print an honest estimate of
  how many generations the budget actually buys.

  RESULTS WRITTEN AS THEY ARE EARNED.  Figures, CSVs and pickles are written
  after EVERY seed, not at the end. If the laptop dies at 4 a.m. the seeds
  that finished are complete and usable.

  FAILURE ISOLATION.  A seed that raises is logged and skipped; the remaining
  seeds still run. Waking up to two good seeds beats waking up to a traceback.

  ITAE, NOT SETTLING TIME.  See tuning/objectives.py -- a settling time capped
  at the run duration ties every non-settling candidate at the same value and
  makes selection among them random.

  HYPERVOLUME STOPPING.  See tuning/hypervolume.py -- per-objective bests only
  see the endpoints of the front, which can sit flat while the front's
  interior is still improving.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # unattended: never try to open a window

import numpy as np

from deimos.dynamics.disturbances import gravity_gradient_torque
from deimos.sim.config import compose_config
from deimos.tuning import objectives as O
from deimos.tuning import report as R
from deimos.tuning.logbook import format_table, write_logbook

HERE = Path(__file__).resolve().parent

DEFAULT_SCENARIOS = [
    HERE / "configs" / "scenarios" / "slew_40_30_25.yaml",
    HERE / "configs" / "scenarios" / "slew_55_65_15.yaml",
]


# --------------------------------------------------------------------------

def build_configs(scenarios, controller, ctype, gravity_gradient=True):
    """One SimConfig per scenario, all sharing the same plant and controller
    preset. Worst-case aggregation across these is what makes the resulting
    gains defensible as not-overfit-to-one-slew."""
    configs = []
    for sc in scenarios:
        cfg = compose_config(sc, controller)
        if gravity_gradient:
            # Integral action only earns its keep against a steady bias. On a
            # disturbance-free slew the PD terms drive the error to zero on
            # their own and the search correctly but uselessly drives Ki to
            # its lower bound -- i.e. spends a 9-gene search rediscovering a
            # PD. The gravity-gradient torque is the physically motivated
            # bias this spacecraft actually experiences.
            tau = gravity_gradient_torque(cfg.satellite.inertia_tensor)
            cfg.disturbance.enabled = True
            cfg.disturbance.constant_torque = np.full(3, tau)
        configs.append(cfg)
    return configs


def preflight(configs, ctype, duration):
    """Evaluate one mid-range candidate on the real configs and time it.

    Cheap insurance: a config typo, an unexpected exception inside simulate(),
    or a scenario that produces non-finite results shows up here in seconds
    instead of silently costing a whole night (the evaluator catches
    exceptions and returns worst-case, which is right for the search but
    would hide a systematic failure behind a front full of worst-case ties).
    """
    ev = O.Evaluator(configs, controller_type=ctype, duration=duration)
    decode = O.make_decode(ctype)
    genes = np.full(O.n_genes(ctype), 0.5)

    t0 = time.perf_counter()
    out = ev(decode(genes))
    dt = time.perf_counter() - t0

    ok = all(np.isfinite(v) for v in out)
    at_worst = tuple(np.round(out, 12)) == tuple(np.round(ev.worst, 12))
    return {"seconds_per_evaluation": dt, "objectives": [float(v) for v in out],
            "finite": bool(ok), "midrange_candidate_is_worst_case": bool(at_worst),
            "worst_case_reference": [float(v) for v in ev.worst]}


def run_seed(seed, configs, ctype, pop_size, generations, duration, workers,
             seed_dir, deadline, stall_patience, stall_tol, stall_min_gen,
             eval_timeout=None, saturation_limit=0.10):
    seed_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"\n{'='*78}\nSEED {seed}   deadline "
          f"{datetime.fromtimestamp(deadline).strftime('%H:%M:%S')}   "
          f"({(deadline - time.time())/60:.0f} min)\n{'='*78}", flush=True)

    pop, obj, decode, ga = O.tune(
        configs,
        controller_type=ctype,
        pop_size=pop_size,
        n_generations=generations,
        seed=seed,
        verbose=True,
        warm_start=True,
        n_workers=workers,
        duration=duration,
        checkpoint_path=seed_dir / "checkpoint.pkl",
        status_path=seed_dir / "status.json",
        save_path=seed_dir / "final.pkl",
        deadline=deadline,
        stall_patience=stall_patience,
        stall_tol=stall_tol,
        stall_min_generations=stall_min_gen,
        eval_timeout=eval_timeout,
        saturation_limit=saturation_limit,
    )
    wall = time.perf_counter() - t0
    if ga.n_timeouts:
        print(f"  ** {ga.n_timeouts} candidate(s) exceeded eval_timeout during "
              f"this seed and were scored worst-case -- see the WARNING lines "
              f"above for which generation(s) **", flush=True)

    # --- per-seed figures, written now rather than at the end -----------
    try:
        from deimos.viz.tuning import plot_tuning
        plot_tuning(pop, obj, decode, ga.history, controller_type=ctype,
                    save_dir=seed_dir / "figures", show=False)
    except Exception:
        print("  (per-seed figures failed)\n" + traceback.format_exc(), flush=True)

    # --- per-seed front CSV ---------------------------------------------
    gain_names = ["Kp_x", "Kp_y", "Kp_z", "Kd_x", "Kd_y", "Kd_z"]
    if ctype == "PID":
        gain_names += ["Ki_x", "Ki_y", "Ki_z"]
    absolute = np.array([np.concatenate(decode(ind)) for ind in pop])
    cols, header = [obj, absolute], list(O.OBJECTIVE_LABELS[ctype]) + gain_names
    if getattr(ga, "final_diagnostics", None) is not None:
        cols.append(ga.final_diagnostics)
        header += list(O.DIAGNOSTIC_LABELS)
    np.savetxt(seed_dir / "pareto_front.csv",
               np.hstack(cols), delimiter=",", header=",".join(header),
               comments="")

    # --- the generation-by-generation record ----------------------------
    try:
        written = write_logbook(seed_dir, ga.history, controller_type=ctype)
        print("  logbook: " + ", ".join(Path(p).name for p in written.values()),
              flush=True)
        # The last few generations as a table, straight into the run log --
        # so the terminal scrollback itself is a readable record.
        print("\n" + format_table(ga.history, ctype, last=15), flush=True)
    except Exception:
        print("  (logbook failed)\n" + traceback.format_exc(), flush=True)

    print(f"\nseed {seed} done in {wall/60:.1f} min -- {ga.stop_reason}", flush=True)
    return {"seed": seed, "wall_s": wall, "generations": len(ga.history) - 1,
            "evaluations": ga.n_evaluations, "front_size": int(len(pop)),
            "final_hypervolume": float(ga.history[-1].get("hypervolume", np.nan)),
            "stop_reason": ga.stop_reason, "n_timeouts": ga.n_timeouts}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0,
                    help="total wall-clock budget (default 8)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="RNG seeds, one independent run each (default 0 1 2)")
    ap.add_argument("--extra-seeds", type=int, default=2,
                    help="additional seeds to start if the budget outlasts the "
                         "planned ones (default 2). Free extra evidence when "
                         "plateau stopping ends the planned seeds early.")
    ap.add_argument("--type", default="PID", choices=["PD", "PID", "WIE"],
                    help="PID tunes Kp, Kd and Ki (9 genes); PD tunes Kp, Kd "
                         "(6); WIE tunes the Wie Case 1 eigenaxis regulator's "
                         "k, d and ki (3 scalars, K = k*J, D = d*J, mu = 1)")
    ap.add_argument("--pop-size", type=int, default=60)
    ap.add_argument("--generations", type=int, default=400,
                    help="ceiling only -- plateau stopping and the deadline "
                         "normally end a seed well before this")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="simulated seconds per evaluation")
    ap.add_argument("--saturation-limit", type=float, default=0.10,
                    help="[fraction 0-1, PID/WIE only] a candidate whose "
                         "worst-case wheel-torque saturation exceeds this is "
                         "scored worst-case and dominated out of the front "
                         "(default 0.10). Lower it to push the whole front "
                         "away from the actuator limit -- e.g. 0.0 rejects "
                         "any candidate that ever saturates a wheel.")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: cores - 1)")
    ap.add_argument("--scenarios", nargs="+",
                    default=[str(p) for p in DEFAULT_SCENARIOS],
                    help="objectives are aggregated WORST-CASE over these")
    ap.add_argument("--controller", default=None,
                    help="controller preset to warm-start from (default: "
                         "wie_eigenaxis.yaml for --type WIE, else "
                         "pd_baseline.yaml)")
    ap.add_argument("--no-gravity-gradient", dest="gravity_gradient",
                    action="store_false", default=True)
    ap.add_argument("--stall-patience", type=int, default=25,
                    help="stop a seed when hypervolume has gained less than "
                         "--stall-tol over this many generations")
    ap.add_argument("--stall-tol", type=float, default=1e-4)
    ap.add_argument("--stall-min-generations", type=int, default=50)
    ap.add_argument("--out", default=str(HERE / "runs" / "overnight"))
    ap.add_argument("--eval-timeout", type=float, default=None,
                    help="[s] hard ceiling on one candidate's evaluation "
                         "time; a candidate that exceeds it is scored "
                         "worst-case rather than blocking its whole "
                         "generation (see nsga2.py:NSGA2 docstring for why "
                         "this exists). Default: 15x the preflight-measured "
                         "seconds-per-evaluation, floor 10s. Only applies "
                         "with --workers > 1 (always true unless overridden "
                         "to 1). --skip-preflight without an explicit value "
                         "here disables it, since there is nothing to "
                         "calibrate the default against.")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--min-seed-seconds", type=float, default=120.0,
                    help="don't start a seed with less than this much budget "
                         "left -- a seed that cannot reach a few generations "
                         "produces a front no better than random sampling")
    args = ap.parse_args()

    ctype = args.type.upper()
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    # The warm-start preset has to be the same FAMILY as the search, or
    # encode_config_gains has nothing to seed from and the run silently
    # starts cold -- which throws away the one property worth having from an
    # elitist GA: that the returned front cannot be worse than the design you
    # started with. A PD preset carries no k/d, and a Wie preset carries no
    # Kp/Kd, so the default has to follow --type.
    _default_preset = {"WIE": "wie_eigenaxis.yaml"}.get(ctype, "pd_baseline.yaml")
    controller = args.controller or str(
        HERE / "configs" / "controllers" / _default_preset)

    # --- output dir; an existing one is archived, never clobbered --------
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        stamp = datetime.fromtimestamp(out.stat().st_mtime).strftime("%Y-%m-%dT%H-%M")
        archived = out.parent / f"{out.name}_{stamp}"
        shutil.move(str(out), str(archived))
        print(f"archived the previous run to {archived}")
    out.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args.scenarios, controller, ctype,
                            args.gravity_gradient)

    t_start = time.time()
    t_end = t_start + args.hours * 3600.0

    print(f"\n{'='*78}")
    print(f"DEIMoS overnight {ctype} gain search")
    print(f"{'='*78}")
    print(f"  genes         : {O.n_genes(ctype)}  ({', '.join(O.gene_labels(ctype))})")
    print(f"  objectives    : {', '.join(O.OBJECTIVE_LABELS[ctype])}")
    print(f"  scenarios     : {', '.join(c.name for c in configs)}  "
          f"(worst-case aggregated)")
    if args.gravity_gradient:
        print(f"  disturbance   : gravity gradient, "
              f"{configs[0].disturbance.constant_torque[0]:.3e} N m per axis")
    else:
        print("  disturbance   : none  (Ki will trend to its lower bound)")
    print(f"  population    : {args.pop_size}")
    if ctype in ("PID", "WIE"):
        print(f"  sat. limit    : {args.saturation_limit:.1%} "
              f"(candidates saturating more than this are dominated out)")
    print(f"  seeds         : {args.seeds} (+ up to {args.extra_seeds} extra if time allows)")
    print(f"  workers       : {workers} of {os.cpu_count()} cores")
    print(f"  eval duration : {args.duration:.0f} simulated seconds")
    print(f"  budget        : {args.hours:.1f} h, finishing by "
          f"{datetime.fromtimestamp(t_end).strftime('%a %H:%M')}")
    print(f"  output        : {out}")
    print(f"\n  monitor with:  python -m deimos.tuning.monitor_ga {out}\n")

    # --- preflight --------------------------------------------------------
    pf = None
    if not args.skip_preflight:
        print("preflight: evaluating one mid-range candidate on the real configs...",
              flush=True)
        pf = preflight(configs, ctype, args.duration)
        print(f"  {pf['seconds_per_evaluation']:.2f} s per evaluation "
              f"({len(configs)} scenario(s))")
        print(f"  objectives: {np.array(pf['objectives'])}")
        if not pf["finite"]:
            raise SystemExit("preflight produced non-finite objectives -- fix the "
                             "config before committing the night to it")
        if pf["midrange_candidate_is_worst_case"]:
            print("  WARNING: a mid-range candidate scored WORST-CASE. Either the\n"
                  "  simulation is failing or every candidate is infeasible. The\n"
                  "  search will still run but is likely to return nothing useful.")
        per_gen = pf["seconds_per_evaluation"] * args.pop_size / workers
        total_gens = (t_end - time.time()) / per_gen
        print(f"  estimate: ~{per_gen:.0f} s per generation, "
              f"~{total_gens:.0f} generations total across "
              f"{len(args.seeds)} seeds "
              f"(~{total_gens/max(len(args.seeds),1):.0f} each)\n", flush=True)

    eval_timeout = args.eval_timeout
    if eval_timeout is None and pf is not None and workers > 1:
        # 15x the measured baseline: generous enough that ordinary variance
        # across candidates (a limit-cycling gain set can genuinely take a
        # few x longer than a well-behaved one) never trips it, but firmly
        # below the ~500x-baseline stall that motivated eval_timeout to
        # exist at all -- that one cost an entire generation, this caps the
        # damage at one candidate.
        eval_timeout = max(10.0, 15.0 * pf["seconds_per_evaluation"])
        print(f"  eval_timeout  : {eval_timeout:.0f}s (15x preflight, auto)\n", flush=True)
    elif eval_timeout is not None:
        print(f"  eval_timeout  : {eval_timeout:.0f}s (explicit)\n", flush=True)

    # --- the seeds --------------------------------------------------------
    planned = list(args.seeds)
    extra = [max(planned) + 1 + i for i in range(args.extra_seeds)]
    summaries, failures = [], []
    queue = list(planned)
    done = 0

    while queue:
        seed = queue.pop(0)
        remaining_seeds = 1 + len(queue)
        time_left = t_end - time.time()
        if time_left < args.min_seed_seconds:
            print(f"\nout of budget ({time_left/60:.1f} min left) -- skipping "
                  f"seed {seed} and everything after it")
            break
        # Re-divided every time: a seed that plateaued early gives its unused
        # minutes to the seeds still to come rather than leaving them on the
        # table.
        seed_deadline = time.time() + time_left / remaining_seeds

        try:
            summaries.append(run_seed(
                seed, configs, ctype, args.pop_size, args.generations,
                args.duration, workers, out / f"seed{seed}", seed_deadline,
                args.stall_patience, args.stall_tol, args.stall_min_generations,
                eval_timeout=eval_timeout,
                saturation_limit=args.saturation_limit))
            done += 1
        except Exception:
            tb = traceback.format_exc()
            print(f"\nSEED {seed} FAILED -- continuing with the rest\n{tb}",
                  flush=True)
            failures.append({"seed": seed, "traceback": tb})
            (out / f"seed{seed}_FAILED.txt").write_text(tb)

        # Interim cross-seed report after every seed, so the results on disk
        # are always complete for the seeds that have finished.
        _write_summary(out, summaries, failures, configs, ctype, args, pf,
                       t_start, workers, controller, final=False)

        # Budget left over? Spend it on another independent trajectory.
        if not queue and extra and (t_end - time.time()) > max(
                600.0, 0.15 * args.hours * 3600.0):
            nxt = extra.pop(0)
            print(f"\n{(t_end - time.time())/60:.0f} min still in budget -- "
                  f"adding seed {nxt}")
            queue.append(nxt)

    _write_summary(out, summaries, failures, configs, ctype, args, pf,
                   t_start, workers, controller, final=True)

    print(f"\n{'='*78}")
    print(f"DONE after {(time.time()-t_start)/3600:.2f} h -- "
          f"{len(summaries)} seed(s) completed, {len(failures)} failed")
    print(f"read {out / 'results.md'} first")
    print(f"{'='*78}\n")


def _write_summary(out, summaries, failures, configs, ctype, args, pf,
                   t_start, workers, controller, final):
    """Cross-seed figures + results.md. Called after every seed so the
    directory on disk is always a complete, readable result set."""
    runs = R.load_seed_runs(out)
    if not runs:
        return

    meta = {
        "controller_type": ctype,
        "genes": ", ".join(O.gene_labels(ctype)),
        "objectives": ", ".join(O.OBJECTIVE_LABELS[ctype]),
        "scenarios": ", ".join(c.name for c in configs) + " (worst-case aggregated)",
        "disturbance": ("gravity gradient, "
                        f"{configs[0].disturbance.constant_torque[0]:.3e} N m/axis"
                        if configs[0].disturbance.enabled else "none"),
        "warm start from": Path(controller).name,
        "population": args.pop_size,
        "evaluation duration": f"{args.duration:.0f} s simulated",
        "saturation limit": (f"{args.saturation_limit:.1%} worst-case wheel "
                             "torque saturation (candidates over this are "
                             "dominated out)" if ctype in ("PID", "WIE") else "n/a"),
        "parallel workers": workers,
        "stall rule": (f"stop when hypervolume gains < {args.stall_tol:g} "
                       f"over {args.stall_patience} generations "
                       f"(min {args.stall_min_generations})"),
        "wall clock": f"{(time.time()-t_start)/3600:.2f} h of a "
                      f"{args.hours:.1f} h budget",
        "completed seeds": len(summaries),
        "failed seeds": len(failures),
    }
    if pf:
        meta["seconds per evaluation"] = f"{pf['seconds_per_evaluation']:.2f}"

    with open(out / "run_summary.json", "w") as f:
        json.dump({"summaries": summaries, "failures": failures, "meta": meta,
                   "preflight": pf, "final": final}, f, indent=2, default=str)

    try:
        from deimos.viz.tuning import plot_seed_comparison
        from deimos.viz.style import BG
        histories = [d["history"] for _, d in runs]
        fronts = [np.atleast_2d(d["obj"]) for _, d in runs]
        labels = [f"seed {d.get('seed', i)}" for i, (_, d) in enumerate(runs)]
        merged, _, _, _ = R.merge_seed_fronts(runs)
        fig = plot_seed_comparison(histories, fronts, labels, ctype, merged=merged)
        figdir = out / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / "seed_comparison.png", facecolor=BG, dpi=150,
                    bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        print("(cross-seed figure failed)\n" + traceback.format_exc(), flush=True)

    try:
        R.write_report(out, runs, configs, controller_type=ctype,
                       duration=args.duration, run_meta=meta)
    except Exception:
        print("(results.md failed)\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    # Required on Windows: worker processes re-import this module, and without
    # the guard each of them would start its own overnight run.
    main()
