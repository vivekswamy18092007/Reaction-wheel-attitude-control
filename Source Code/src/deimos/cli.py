"""
cli.py
======

    deimos run     --scenario configs/scenarios/X.yaml --controller configs/controllers/Y.yaml
    deimos compare --scenario configs/scenarios/X.yaml --controller Y1.yaml --controller Y2.yaml
    deimos tune    --scenario configs/scenarios/X.yaml

Every command composes the constants-derived base layer (satellite + wheels
+ magnetorquers, see sim/config.py:_base_raw) with a scenario and, for
run/compare, a controller block (sim.config.compose_config), then writes a
timestamped run directory under runs/:

    runs/<timestamp>_<name>/
        config.yaml     snapshot of the exact composed config
        manifest.json   git SHA, package versions, wall time
        metrics.json    the headline SimResults numbers
        figures/        PNGs (unless --no-plots)

This is orchestration only -- no physics or control logic lives here, only
calls into sim.runner, analysis.design_card and viz.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from deimos.sim.config import compose_config, SimConfig
from deimos.sim.runner import simulate
from deimos.sim.results import SimResults
from deimos.analysis.design_card import describe


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip()
    except Exception:
        return None


def _versions() -> dict:
    import matplotlib
    versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                "matplotlib": matplotlib.__version__}
    return versions


def _config_to_dict(config: SimConfig) -> dict:
    """Snapshot a SimConfig back into a plain YAML-serializable dict, close
    to the schema compose_config()/load_config() read, for the run
    directory's config.yaml."""
    c = config.controller
    controller = {"type": c.type}
    if c.Kp is not None:
        controller["Kp"] = np.diag(c.Kp).tolist()
    if c.Kd is not None:
        controller["Kd"] = np.diag(c.Kd).tolist()
    if c.case is not None:
        controller["case"] = c.case
    if c.zeta is not None:
        controller["zeta"] = c.zeta
    if c.settling_time_s is not None:
        controller["settling_time_s"] = c.settling_time_s
    if c.Ki is not None:
        controller["Ki"] = c.Ki.tolist()
    if c.u_max is not None:
        controller["u_max"] = c.u_max

    initial = {
        "attitude_rpy_deg": config.initial.attitude_rpy_deg.tolist(),
        "omega": config.initial.omega.tolist(),
    }
    if config.initial.wheel_speeds is not None:
        initial["wheel_speeds"] = config.initial.wheel_speeds.tolist()

    return {
        "satellite": {"inertia_tensor": config.satellite.inertia_tensor.tolist()},
        "initial": initial,
        "target": {"attitude_rpy_deg": config.target.attitude_rpy_deg.tolist()},
        "controller": controller,
        "wheels": {
            "config": config.wheels.config,
            "tilt_deg": config.wheels.tilt_deg,
            "wheel_inertia": config.wheels.wheel_inertia,
            "max_torque": config.wheels.max_torque,
            "max_speed": config.wheels.max_speed,
        },
        "magnetorquers": {
            "enabled": config.magnetorquers.enabled,
            "max_dipole": config.magnetorquers.max_dipole,
            "k_desat": config.magnetorquers.k_desat,
            "threshold_frac": config.magnetorquers.threshold_frac,
        },
        "disturbance": {
            "enabled": config.disturbance.enabled,
            "constant_torque": config.disturbance.constant_torque.tolist(),
        },
        "sim": {"dt": config.sim.dt, "duration": config.sim.duration},
    }


def _metrics_dict(results: SimResults) -> dict:
    return {
        "settling_time_s": results.settling_time(),
        "final_attitude_error_deg": results.final_attitude_error_deg(),
        "overshoot_deg": results.overshoot_deg(),
        "peak_control_torque_Nm": results.peak_control_torque(),
        "peak_wheel_torque_Nm": results.peak_wheel_torque(),
        "torque_margin": results.torque_margin(),
        "saturation_fraction": results.saturation_fraction(),
        "speed_saturation_fraction": results.speed_saturation_fraction(),
        "control_effort_Nms": results.control_effort(),
        "electrical_energy_J": results.electrical_energy(),
        "mean_eigenaxis_deviation_deg": results.mean_eigenaxis_deviation_deg(),
    }


def _make_run_dir(out: str, name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
    run_dir = Path(out) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def _write_manifest(run_dir: Path, wall_time_s: float, extra: dict | None = None):
    manifest = {"git_sha": _git_sha(), "versions": _versions(),
                "wall_time_s": wall_time_s, "generated_at": datetime.now().isoformat()}
    if extra:
        manifest.update(extra)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# --- commands ---------------------------------------------------------

def _cmd_run(args):
    from deimos.viz.single_run import plot
    from deimos.analysis.power import power_card

    config = compose_config(args.scenario, args.controller)
    t0 = time.perf_counter()
    results = simulate(config)
    wall = time.perf_counter() - t0

    print(results.summary())
    print(describe(config))
    print(power_card(results))

    run_dir = _make_run_dir(args.out, config.name)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(_config_to_dict(config), f, sort_keys=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(_metrics_dict(results), f, indent=2)
    _write_manifest(run_dir, wall)

    if not args.no_plots:
        plot(results, names=args.plots, save_dir=run_dir / "figures", show=args.show)

    print(f"\nwrote {run_dir}")


def _cmd_compare(args):
    from deimos.viz.comparison import compare, comparison_table
    from deimos.analysis.stability import stability_branch

    runs = {}
    stab = {}
    t0 = time.perf_counter()
    for controller_path in args.controller:
        # name must be unique per controller, not just the scenario stem --
        # otherwise two controllers sharing one scenario collide on the same
        # dict key and the earlier run silently disappears from the table.
        name = f"{Path(args.scenario).stem}_{Path(controller_path).stem}"
        config = compose_config(args.scenario, controller_path, name=name)
        results = simulate(config)
        label = config.name
        runs[label] = results

        ctype = config.controller.type.strip().lower()
        if ctype == "wie":
            from deimos.sim.runner import _build_controller
            c = _build_controller(config)
            status, _, _ = stability_branch(c.K, c.D, config.satellite.inertia_tensor, c.mu)
            stab[label] = status
        else:
            stab[label] = f"not claimed ({config.controller.type})"

    wall = time.perf_counter() - t0
    table = comparison_table(runs, stability=stab)
    print(table)

    run_dir = _make_run_dir(args.out, f"compare_{Path(args.scenario).stem}")
    with open(run_dir / "comparison_table.txt", "w") as f:
        f.write(table + "\n")
    _write_manifest(run_dir, wall, extra={"controllers": list(args.controller)})

    if not args.no_plots:
        compare(runs, names="all", save_dir=run_dir / "figures", show=args.show)

    print(f"\nwrote {run_dir}")


def _cmd_tune(args):
    from deimos.dynamics.disturbances import gravity_gradient_torque
    from deimos.tuning.objectives import (
        OBJECTIVE_LABELS, apply_gains, make_decode, tune, to_runs_dict,
    )

    ctype = args.type.strip().upper()
    base_controller = args.controller or _first_controller_config()
    base_config = compose_config(args.scenario, base_controller)

    # Integral action only earns its keep against a steady bias -- on a
    # disturbance-free slew the search would just drive Ki to its lower
    # bound. Default a gravity-gradient torque on for PID, and say so.
    if args.gravity_gradient:
        tau_gg = gravity_gradient_torque(base_config.satellite.inertia_tensor)
        base_config.disturbance.enabled = True
        base_config.disturbance.constant_torque = np.full(3, tau_gg)
        print(f"disturbance: gravity-gradient, {tau_gg:.4e} N m per axis "
              f"(500 km LEO, from J's asymmetry)")
    elif ctype == "PID" and not base_config.disturbance.enabled:
        print("WARNING: tuning PID with no disturbance enabled -- Ki will trend\n"
              "         to its lower bound (see tuning/objectives.py). Consider\n"
              "         dropping --no-gravity-gradient.")

    labels = OBJECTIVE_LABELS[ctype]
    n_evals = args.pop_size * (args.generations + 1)
    print(f"\ntuning {ctype} gains on {Path(args.scenario).stem}")
    print(f"  objectives : {', '.join(labels)}")
    print(f"  budget     : {args.pop_size} pop x {args.generations + 1} gens "
          f"= {n_evals} sims of {args.duration:.0f} s each\n")

    t0 = time.perf_counter()
    pop, obj, decode, ga = tune(base_config, controller_type=ctype,
                                 pop_size=args.pop_size,
                                 n_generations=args.generations,
                                 duration=args.duration)
    wall = time.perf_counter() - t0

    picks = to_runs_dict(pop, obj, decode, controller_type=ctype)
    print(f"\nPareto front: {len(pop)} solutions   ({wall:.1f} s, {n_evals} evaluations)")
    print(f"\n{'pick':<22}" + "".join(f"{lab:>26}" for lab in labels))
    print("-" * (22 + 26 * len(labels)))
    for label, p in picks.items():
        print(f"{label:<22}" + "".join(f"{v:>26.4e}" for v in p["objectives"]))

    run_dir = _make_run_dir(args.out, f"tune_{ctype}_{Path(args.scenario).stem}")

    payload = {}
    for label, p in picks.items():
        entry = {"type": p["type"], "objectives": list(p["objectives"]),
                 "objective_labels": list(labels),
                 "Kp": np.diag(p["Kp"]).tolist(), "Kd": np.diag(p["Kd"]).tolist()}
        if p["Ki"] is not None:
            entry["Ki"] = np.asarray(p["Ki"]).tolist()
        payload[label] = entry
    with open(run_dir / "pareto_picks.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Full front, so a different pick can be made later without re-running.
    # Gains are written as ABSOLUTE values (Kp, Kd, Ki), not the Ki/Kp ratio
    # the search internally operates on -- this file is meant to be read
    # straight into a config, not fed back into the GA.
    absolute_gains = np.array([np.concatenate(decode(ind)) for ind in pop])
    gain_names = ["Kp_x", "Kp_y", "Kp_z", "Kd_x", "Kd_y", "Kd_z"]
    if ctype == "PID":
        gain_names += ["Ki_x", "Ki_y", "Ki_z"]
    np.savetxt(run_dir / "pareto_front.csv",
               np.hstack([obj, absolute_gains]), delimiter=",",
               header=",".join(list(labels) + gain_names), comments="")

    _write_manifest(run_dir, wall, extra={
        "controller_type": ctype, "pop_size": args.pop_size,
        "generations": args.generations, "eval_duration_s": args.duration,
        "evaluations": n_evals, "pareto_size": len(pop),
        "objective_labels": list(labels),
        "disturbance_enabled": bool(base_config.disturbance.enabled),
    })

    if not args.no_plots:
        from deimos.viz.tuning import plot_tuning
        from deimos.viz.comparison import compare, comparison_table

        plot_tuning(pop, obj, decode, ga.history, controller_type=ctype,
                     save_dir=run_dir / "figures", show=False)

        # Re-simulate the picks alongside the untuned baseline: the Pareto
        # front is objective-space only, and the time-domain response is what
        # actually goes in the report.
        runs = {f"baseline ({Path(base_controller).stem})": simulate(base_config)}
        for label, p in picks.items():
            cfg = copy.deepcopy(base_config)
            apply_gains(cfg, decode(pop[p["index"]]), ctype)
            cfg.name = label
            runs[label] = simulate(cfg)

        table = comparison_table(runs)
        print("\n" + table)
        with open(run_dir / "tuned_comparison.txt", "w") as f:
            f.write(table + "\n")
        compare(runs, names="all", save_dir=run_dir / "figures",
                prefix="tuned", show=args.show)

    print(f"\nwrote {run_dir}")


def _cmd_animate(args):
    from deimos.viz.attitude3d import animate_results

    config = compose_config(args.scenario, args.controller)
    results = simulate(config)
    print(results.summary())

    if args.save is None:
        print("opening interactive animation window (pass --save out.mp4 "
              "to export the demo-video asset instead)")
    animate_results(results, save_path=args.save, stl_path=args.stl,
                    fps=args.fps, stride=args.stride)
    if args.save:
        print(f"wrote {args.save}")


def _first_controller_config() -> str:
    here = Path(__file__).resolve().parents[2] / "configs" / "controllers"
    candidates = sorted(here.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"no controller yaml found in {here}; pass --controller explicitly")
    return str(candidates[0])


# --- argument parsing ---------------------------------------------------

def _add_compose_args(p, controller_multi=False):
    p.add_argument("--scenario", required=True)
    if controller_multi:
        p.add_argument("--controller", dest="controller", action="append", required=True,
                        help="repeatable")
    else:
        p.add_argument("--controller", required=True)
    p.add_argument("--out", default="runs")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-plots", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="deimos")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="simulate one scenario+controller and plot it")
    _add_compose_args(run_p)
    run_p.add_argument("--plots", default="all",
                        help="'all', a group name, or comma-separated figure names")

    compare_p = sub.add_parser("compare", help="simulate one scenario under several controllers")
    _add_compose_args(compare_p, controller_multi=True)

    tune_p = sub.add_parser("tune", help="NSGA-II PD/PID gain search against a scenario")
    tune_p.add_argument("--scenario", required=True)
    tune_p.add_argument("--type", default="PID", choices=["PD", "PID", "pd", "pid"],
                         help="which control law to tune (default: PID)")
    tune_p.add_argument("--controller", default=None,
                         help="seed controller block (its gains get overridden); "
                              "defaults to the first file in configs/controllers/")
    tune_p.add_argument("--pop-size", dest="pop_size", type=int, default=24)
    tune_p.add_argument("--generations", type=int, default=20)
    tune_p.add_argument("--duration", type=float, default=60.0,
                         help="simulated seconds per candidate evaluation")
    tune_p.add_argument("--gravity-gradient", dest="gravity_gradient",
                         action="store_true", default=True,
                         help="apply a gravity-gradient bias so integral action "
                              "has something to reject (default: on)")
    tune_p.add_argument("--no-gravity-gradient", dest="gravity_gradient",
                         action="store_false")
    tune_p.add_argument("--out", default="runs")
    tune_p.add_argument("--show", action="store_true")
    tune_p.add_argument("--no-plots", action="store_true")

    animate_p = sub.add_parser(
        "animate", help="3D attitude animation of one scenario+controller "
                        "(the demo-video asset)")
    animate_p.add_argument("--scenario", required=True)
    animate_p.add_argument("--controller", required=True)
    animate_p.add_argument("--save", default=None,
                            help="output .mp4 (GIF fallback without ffmpeg); "
                                 "omit for an interactive window")
    animate_p.add_argument("--stl", default=None,
                            help="CAD STL to render instead of the schematic "
                                 "cube (needs `pip install deimos[viz3d]`)")
    animate_p.add_argument("--fps", type=int, default=30)
    animate_p.add_argument("--stride", type=int, default=None,
                            help="keep every Nth sample (default: real-time "
                                 "playback at --fps, capped at ~1200 frames)")

    args = parser.parse_args(argv)
    if isinstance(getattr(args, "plots", None), str) and "," in args.plots:
        args.plots = args.plots.split(",")

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "compare":
        _cmd_compare(args)
    elif args.command == "tune":
        _cmd_tune(args)
    elif args.command == "animate":
        _cmd_animate(args)


if __name__ == "__main__":
    main()
