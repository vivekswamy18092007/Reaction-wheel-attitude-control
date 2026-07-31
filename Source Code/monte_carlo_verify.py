"""
Monte Carlo consistency / verification study for DEIMoS controllers.

    python monte_carlo_verify.py --n-runs 20                      # dry run first
    python monte_carlo_verify.py --n-runs 500                     # the real thing
    python monte_carlo_verify.py --n-runs 500 --controllers wie_eigenaxis
    python monte_carlo_verify.py --n-runs 500 --angle-min 0 --angle-max 8 \\
        --scenario-template configs/scenarios/hold_under_disturbance.yaml \\
        --controllers wie_eigenaxis wie_eigenaxis_hold
        # small-angle, fine-pointing regime under hold_under_disturbance.yaml's
        # own realistic bias -- the regime wie_eigenaxis_hold's integral action
        # was added for, run matched-pairs against the no-integral manoeuvre
        # preset so the effect of switching Ki on is visible over an ENSEMBLE,
        # not just the one hand-picked slew Scenario B plots

For each run: sample ONE random slew (uniform random rotation axis + rotation
angle uniform in [--angle-min, --angle-max] degrees -- this is the
statistically unbiased way to sample "a random rotation of roughly this size"
without clustering towards particular orientations the way independent-per-axis
roll/pitch/yaw sampling would; the same sampler covers both the default
large-angle regime and a small-angle regime via --angle-min/--angle-max), then
simulate it under EVERY controller in --controllers, so every controller sees
the EXACT SAME slew for run i -- a matched-pairs design, not an
independently-sampled one per controller.

DISTURBANCE.  Whatever --scenario-template itself specifies, by default: if
its `disturbance.enabled` block is set, that constant per-axis bias is applied
to every run; if not, there is none. This is deliberately the same rule every
other DEIMoS entry point uses (compose_config's own scenario+controller
merge), so pointing --scenario-template at hold_under_disturbance.yaml picks
up its bias automatically with no extra flag. --gravity-gradient overrides
this with the real, physically-motivated bound instead --
dynamics.disturbances.gravity_gradient_torque(J), the same
(3/2)*n^2*|J_max-J_min| bound overnight_tune.py's PID/WIE searches and the
report use -- and --disturbance-torque overrides it with an explicit per-axis
value. --no-disturbance forces it off regardless of what the scenario
template says. Whichever source wins, it is one constant vector for the whole
ensemble (a scenario config has no per-run variation to draw from), applied
every step as tau_ext, and printed in the startup banner rather than repeated
per CSV row.

Each run seeks: settling time, control effort, and steady-state error
(SSE). Metric definitions are NOT reinvented here -- they are the same ones
tuning/objectives.py already uses (int|u|dt for effort; SSE = mean attitude
error over the trailing steady_state_frac of whatever was actually
simulated), so this study's numbers are directly comparable to anything
`deimos tune` / overnight_tune.py already produced.

EARLY STOPPING, not a blind fixed duration.  A run stops as soon as the
attitude error has stayed at or below --settle-threshold for a full
--confirm-window seconds (settled, by construction, in the same sense
SimResults.settling_time() checks), or once --max-duration is hit (flagged
NOT SETTLED -- settling_time recorded as NaN, never guessed). This is why a
fast controller costs far less wall-clock than a slow or non-convergent one
per run, rather than every run paying for the worst case every time.

REPRODUCIBLE.  Every run's slew is derived from --seed and its own run
index via a SeedSequence, independent of worker scheduling order --
re-running with the same --seed reproduces the exact same ensemble, and any
single run_id can be replayed in isolation later (see replay_run() at the
bottom) once a plot flags one as interesting.

OUTPUT.  One CSV row per (run_id, controller) pair, written incrementally
as results complete -- a killed/interrupted run keeps whatever finished, and
you can `tail -f` the file to watch it progress. Extra diagnostic columns
beyond the three requested (peak wheel torque, saturation fraction, mean
eigenaxis deviation) ride along for free, since SimResults already computes
them -- useful for "does saturation correlate with slew angle" style plots
later without re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from deimos.sim.config import compose_config
from deimos.sim.runner import _build_controller, _build_wheels, _make_u_func
from deimos.dynamics.propagator import AttitudeSimulator
from deimos.dynamics.disturbances import gravity_gradient_torque
from deimos.sim.results import SimResults
from deimos.math.quaternion import Quaternion, quaternion_error

HERE = Path(__file__).resolve().parent
IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])

FIELDS = [
    "run_id", "controller", "seed", "angle_deg", "axis_x", "axis_y", "axis_z",
    "initial_error_deg", "settled", "settling_time_s", "control_effort_Nms",
    "sse_deg", "final_error_deg", "peak_wheel_torque_Nm", "saturation_fraction",
    "mean_eigenaxis_deviation_deg", "sim_seconds_run", "wall_seconds",
]


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def random_axis_angle_quat(rng: np.random.Generator, angle_range_deg):
    """Uniform random rotation axis (unbiased on the sphere via a normalized
    Gaussian vector) + rotation angle ~ Uniform(angle_range_deg). This is the
    correct way to sample "a large random rotation" without the clustering
    artifacts independent roll/pitch/yaw sampling would introduce."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_deg = rng.uniform(*angle_range_deg)
    angle_rad = np.radians(angle_deg)
    w = np.cos(angle_rad / 2.0)
    xyz = axis * np.sin(angle_rad / 2.0)
    return np.array([w, *xyz]), axis, angle_deg


# --------------------------------------------------------------------------
# one run: simulate with early stopping, no blind fixed duration
# --------------------------------------------------------------------------

def _attitude_error_deg(q, q_target):
    q_e = quaternion_error(q, q_target)
    w_err = np.clip(abs(q_e[0]), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w_err))


def run_one(controller_name: str, q0: np.ndarray, dt: float, max_duration: float,
            settle_threshold_deg: float, confirm_window_s: float,
            steady_state_frac: float, scenario_template: str,
            disturbance_torque: np.ndarray | None = None):
    """Simulate q0 -> identity under one controller preset, stopping early
    once settled. Returns a dict matching FIELDS (minus run_id/seed/angle/
    axis, added by the caller).

    disturbance_torque: constant body-frame bias [N m], held fixed for the
        whole run (tau_ext every step). None (default) reproduces the
        original disturbance-free behaviour."""
    cfg = compose_config(scenario_template,
                          str(HERE / "configs" / "controllers" / f"{controller_name}.yaml"))
    J = cfg.satellite.inertia_tensor
    controller = _build_controller(cfg)
    wheels = _build_wheels(cfg)
    sim = AttitudeSimulator(inertia_tensor=J, q0=q0, omega0=np.zeros(3),
                             wheel_array=wheels, dt=dt)
    u_func = _make_u_func(cfg, controller, Quaternion(*IDENTITY_Q), wheels=wheels)

    tau_ext = (np.zeros(3) if disturbance_torque is None
              else np.asarray(disturbance_torque, dtype=float))
    confirm_steps = max(1, int(round(confirm_window_s / dt)))
    max_steps = int(round(max_duration / dt))
    consecutive = 0

    t_wall0 = time.perf_counter()
    for _ in range(max_steps):
        u_cmd = u_func(sim.t, sim.q, sim.omega, sim.Omega)
        sim.step(u_cmd, tau_ext=tau_ext)
        err = _attitude_error_deg(sim.q, IDENTITY_Q)
        consecutive = consecutive + 1 if err <= settle_threshold_deg else 0
        if consecutive >= confirm_steps:
            break
    wall = time.perf_counter() - t_wall0

    results = SimResults.from_history(name=controller_name, history=sim.history,
                                       q_target=IDENTITY_Q, inertia_tensor=J, wheels=wheels)
    settle = results.settling_time(settle_threshold_deg)
    tail = results.attitude_error_deg[
        int((1.0 - steady_state_frac) * len(results.t)):]

    return {
        "controller": controller_name,
        "initial_error_deg": float(results.attitude_error_deg[0]),
        "settled": settle is not None,
        "settling_time_s": float("nan") if settle is None else float(settle),
        "control_effort_Nms": results.control_effort(),
        "sse_deg": float(np.mean(tail)),
        "final_error_deg": results.final_attitude_error_deg(),
        "peak_wheel_torque_Nm": results.peak_wheel_torque(),
        "saturation_fraction": results.saturation_fraction(),
        "mean_eigenaxis_deviation_deg": results.mean_eigenaxis_deviation_deg(),
        "sim_seconds_run": float(results.t[-1]),
        "wall_seconds": wall,
    }


def _worker(payload):
    """Top-level (picklable) entry point for ProcessPoolExecutor."""
    (run_id, controller_name, seed, angle_range, dt, max_duration,
     settle_threshold_deg, confirm_window_s, steady_state_frac, scenario_template,
     disturbance_torque) = payload
    rng = np.random.default_rng(np.random.SeedSequence([seed, run_id]))
    q0, axis, angle_deg = random_axis_angle_quat(rng, angle_range)
    row = run_one(controller_name, q0, dt, max_duration, settle_threshold_deg,
                  confirm_window_s, steady_state_frac, scenario_template,
                  disturbance_torque=disturbance_torque)
    row.update(run_id=run_id, seed=seed, angle_deg=angle_deg,
               axis_x=axis[0], axis_y=axis[1], axis_z=axis[2])
    return row


# --------------------------------------------------------------------------
# replay a single run_id later, e.g. to plot the worst outlier in detail
# --------------------------------------------------------------------------

def replay_run(run_id: int, controller_name: str, seed: int = 20260730,
                angle_range=(0.0, 8.0), dt: float = 0.01, max_duration: float = 200.0,
                settle_threshold_deg: float = 1.0, confirm_window_s: float = 10.0,
                steady_state_frac: float = 0.2,
                scenario_template: str = "configs/scenarios/hold_under_disturbance.yaml",
                disturbance_torque=None):
    """Reproduce exactly one (run_id, controller) pair's slew for closer
    inspection -- same seeding rule _worker() uses, so run_id N always maps
    to the same quaternion regardless of how the batch was parallelized."""
    return _worker((run_id, controller_name, seed, angle_range, dt, max_duration,
                     settle_threshold_deg, confirm_window_s, steady_state_frac,
                     scenario_template, disturbance_torque))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", nargs="+",
                    default=["wie_eigenaxis", "pid_example"],
                    help="controller preset names (no .yaml), from configs/controllers/")
    ap.add_argument("--n-runs", type=int, default=500,
                    help="number of random slews (each run under EVERY controller)")
    ap.add_argument("--seed", type=int, default=20260730,
                    help="master seed -- run_id N always samples the same slew")
    ap.add_argument("--angle-min", type=float, default=90.0)
    ap.add_argument("--angle-max", type=float, default=179.0,
                    help="kept below 180 to avoid the q0=0 sign-ambiguity edge case")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--max-duration", type=float, default=200.0,
                    help="[s] hard cap per run; calibration showed both default "
                         "presets settle a 150-160 deg slew in 20-40 s, so this "
                         "leaves ~5x margin before a run is flagged NOT SETTLED")
    ap.add_argument("--settle-threshold-deg", type=float, default=1.0)
    ap.add_argument("--confirm-window-s", type=float, default=10.0,
                    help="how long error must stay under threshold before a "
                         "run is called settled and stops early")
    ap.add_argument("--steady-state-frac", type=float, default=0.2,
                    help="trailing fraction of the (possibly early-stopped) "
                         "run averaged for SSE -- same convention as "
                         "tuning/objectives.py")
    ap.add_argument("--scenario-template",
                    default="configs/scenarios/slew_160_40_30.yaml",
                    help="supplies dt/hardware defaults AND, by default, the "
                         "disturbance (see DISTURBANCE above); initial/target "
                         "are overridden per run regardless")
    ap.add_argument("--gravity-gradient", action="store_true",
                    help="override --scenario-template's own disturbance with "
                         "the real orbital bound, "
                         "dynamics.disturbances.gravity_gradient_torque(J), "
                         "as a constant per-axis bias for the whole ensemble")
    ap.add_argument("--disturbance-torque", type=float, nargs=3, default=None,
                    metavar=("TX", "TY", "TZ"),
                    help="[N m] override with an explicit constant body-frame "
                         "bias instead of the scenario's own or the "
                         "gravity-gradient bound")
    ap.add_argument("--no-disturbance", action="store_true",
                    help="force disturbance off even if --scenario-template "
                         "has one enabled")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: cores - 1")
    ap.add_argument("--out", default=str(HERE / "runs" / "monte_carlo_verify.csv"))
    args = ap.parse_args(argv)

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    angle_range = (args.angle_min, args.angle_max)

    # J and the scenario's own disturbance block are the same regardless of
    # which controller preset is probed (constants.py is the single source
    # for J; the disturbance block lives in the scenario file, not the
    # controller), so probing the first --controllers entry is enough to
    # resolve both for the whole ensemble.
    probe_cfg = compose_config(
        args.scenario_template,
        str(HERE / "configs" / "controllers" / f"{args.controllers[0]}.yaml"))

    disturbance_source = "scenario"
    if args.no_disturbance:
        disturbance_torque, disturbance_source = None, "none (forced)"
    elif args.disturbance_torque is not None:
        disturbance_torque = np.array(args.disturbance_torque, dtype=float)
        disturbance_source = "explicit"
    elif args.gravity_gradient:
        tau_gg = gravity_gradient_torque(probe_cfg.satellite.inertia_tensor)
        disturbance_torque = np.full(3, tau_gg)
        disturbance_source = "gravity-gradient bound"
    elif getattr(probe_cfg.disturbance, "enabled", False):
        disturbance_torque = np.asarray(probe_cfg.disturbance.constant_torque,
                                        dtype=float)
    else:
        disturbance_torque, disturbance_source = None, "none (scenario has none enabled)"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payloads = [
        (run_id, controller, args.seed, angle_range, args.dt, args.max_duration,
         args.settle_threshold_deg, args.confirm_window_s, args.steady_state_frac,
         args.scenario_template, disturbance_torque)
        for run_id in range(args.n_runs)
        for controller in args.controllers
    ]
    total = len(payloads)

    print(f"Monte Carlo verification: {args.n_runs} random slews "
          f"({args.angle_min:.0f}-{args.angle_max:.0f} deg) x "
          f"{len(args.controllers)} controller(s) = {total} runs, "
          f"{workers} worker process(es)")
    if disturbance_torque is None:
        print(f"disturbance: {disturbance_source}")
    elif np.all(disturbance_torque == disturbance_torque[0]):
        print(f"disturbance: {disturbance_torque[0]:.3e} N m/axis constant "
              f"({disturbance_source})")
    else:
        print(f"disturbance: {disturbance_torque} N m constant ({disturbance_source})")
    print(f"writing incrementally to {out_path}")

    t0 = time.perf_counter()
    done = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for fut in as_completed(futures):
                row = fut.result()
                writer.writerow(row)
                f.flush()
                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    elapsed = time.perf_counter() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else float("nan")
                    print(f"  {done}/{total}  elapsed={elapsed:6.1f}s  "
                          f"eta={eta:6.1f}s  ({rate:.2f} runs/s)")

    wall = time.perf_counter() - t0
    print(f"\ndone: {total} runs in {wall:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
