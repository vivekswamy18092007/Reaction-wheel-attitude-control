"""
Post-run analysis for a multi-seed NSGA-II gain search.

Everything here reads the pickles the run already wrote -- nothing re-runs
the GA. Three things it produces that the raw front does not:

  1. A merged front across seeds, plus the run-to-run variance statement that
     turns several stochastic trajectories into a result.

  2. The gene-bounded vs actuator-bounded verdict for the fast endpoint (see
     bound_diagnostic). This is a viva question, not a plotting nicety.

  3. Re-simulated picks with the time-domain numbers the report actually
     quotes -- settling time, overshoot, saturation -- which objective space
     alone cannot give you.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np

from deimos.sim.runner import simulate
from deimos.tuning import objectives as O


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_run(path):
    """Load a checkpoint or final pickle. Both carry pop/obj/history."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_seed_runs(root, pattern="seed*/final.pkl"):
    """All per-seed pickles under a run directory, sorted by seed."""
    root = Path(root)
    runs = []
    for p in sorted(root.glob(pattern)):
        try:
            runs.append((p, load_run(p)))
        except (EOFError, pickle.UnpicklingError):
            continue
    return runs


# --------------------------------------------------------------------------
# the viva question: gene-bounded or actuator-bounded?
# --------------------------------------------------------------------------

# A gene sitting this close to the top of its log-uniform range is "pinned".
# Not 1.0 exactly: SBX and polynomial mutation both clip at the bound, so a
# pinned gene lands ON the bound often enough, but a gene that the search has
# pushed to 0.99 is making the same statement.
PIN_THRESHOLD = 0.97

# Above this saturation fraction the wheels are meaningfully torque-limited
# for that maneuver. Well below the 0.10 PID feasibility limit on purpose:
# this is "is the actuator binding at all", not "is this candidate feasible".
SATURATION_ACTIVE = 0.02


def _SPEC_FIRST_BLOCK_SIZE(ctype):
    """How many genes the proportional-gain block occupies: 3 for PD/PID's
    per-axis Kp, 1 for WIE's scalar k."""
    return O._SPEC[O._normalize_type(ctype)][0][2]


def bound_diagnostic(base_configs, decoded, controller_type="PD",
                     duration=120.0, scale_factors=(2.0, 5.0)):
    """Is the fast endpoint of the front limited by the GENE BOUND or by the
    4e-3 N m WHEEL TORQUE LIMIT?

    Gains pinning at a bound is the expected signature of a Pareto extreme,
    not evidence of a badly chosen decode range -- the minimum-effort endpoint
    *should* sit at minimum Kp, and the fastest endpoint *should* push Kp up
    until something stops it. The question worth answering is what stops it,
    because the two answers have opposite consequences:

        gene-bounded     the search wanted more Kp and the search space
                         refused. Widening KP_BOUNDS would move the endpoint.
                         The reported extreme is an artefact of the bounds.

        actuator-bounded the wheels saturate before more Kp helps. The bound
                         is irrelevant; widening it changes nothing, and the
                         endpoint is a real hardware limit.

    Two independent pieces of evidence are gathered rather than one, because
    either alone can mislead:

      static  -- where the gains sit inside their bounds, and how saturated
                 the run is
      active  -- what actually happens if you scale Kp up past the bound and
                 re-simulate. If ITAE improves, the bound was binding. If it
                 does not (or gets worse), the actuator was. This is the
                 decisive test, and it costs a handful of simulations.

    Returns a dict, including a plain-English `verdict` suitable for quoting.
    """
    configs = (list(base_configs) if isinstance(base_configs, (list, tuple))
               else [base_configs])
    ctype = O._normalize_type(controller_type)

    # The first gene block is the proportional gain for every controller
    # type, but it is not always three genes wide: WIE's is the single scalar
    # k. Slicing [:3] would have swept d and the ki ratio into the "is Kp
    # pinned" test and reported a bound verdict about the wrong genes.
    n_first = _SPEC_FIRST_BLOCK_SIZE(ctype)
    gain_name = "k" if ctype == "WIE" else "Kp"

    pos = O.normalized_gene_position(decoded, ctype)
    kp_pos = pos[:n_first]
    kp_pinned = bool(np.max(kp_pos) >= PIN_THRESHOLD)

    ev = O.Evaluator(configs, controller_type=ctype, duration=duration,
                     saturation_limit=None)   # diagnostic must SEE saturation,
                                              # not have it hidden behind the
                                              # feasibility short-circuit
    baseline = ev(decoded)
    baseline_itae = float(baseline[0])

    # saturation, measured directly rather than inferred from the objective
    # tuple (PID's objective vector does not carry it)
    sat = []
    for cfg in configs:
        c = O.apply_gains(copy.deepcopy(cfg), decoded, ctype)
        c.sim.duration = duration
        try:
            sat.append(simulate(c).saturation_fraction())
        except Exception:
            sat.append(float("nan"))
    saturation = float(np.nanmax(sat)) if len(sat) else float("nan")
    actuator_active = bool(np.isfinite(saturation) and saturation > SATURATION_ACTIVE)

    # --- the active test: push Kp past its bound and see if it buys anything
    trials = []
    for s in scale_factors:
        scaled = list(copy.deepcopy(list(decoded)))
        scaled[0] = np.asarray(scaled[0]) * s        # proportional gain only
        if ctype in ("PID", "WIE"):
            # The integral gain was decoded as a RATIO of the proportional
            # one, so scaling the proportional gain alone would silently
            # change the ratio the search actually operates on. Scaling both
            # keeps the ratio fixed and isolates the gain under test.
            #
            # For WIE this also keeps the trial inside Case 1: k and ki both
            # scale, d does not, so K = k*J and Ki = ki*diag(J) stay matched
            # to J and the trial candidate is still an eigenaxis controller
            # -- just an over-gained one. A trial that broke the K = k*J tie
            # would be answering a question about a different control law.
            scaled[2] = np.asarray(scaled[2]) * s
        out = ev(tuple(scaled))
        trials.append({"kp_scale": float(s), "itae": float(out[0]),
                       "itae_ratio": float(out[0] / baseline_itae)
                       if baseline_itae > 0 else float("nan")})

    best_ratio = min(t["itae_ratio"] for t in trials) if trials else float("nan")
    # 2% is a deliberately loose bar: anything smaller is not a real
    # improvement at this noise level and should not be called one.
    more_kp_helps = bool(np.isfinite(best_ratio) and best_ratio < 0.98)

    bound_name = "K_SCALE_BOUNDS" if ctype == "WIE" else "KP_BOUNDS"

    if more_kp_helps and kp_pinned:
        verdict = (f"GENE-BOUNDED. {gain_name} is pinned at the top of its search range "
                   f"and scaling it up by {min(scale_factors):g}-{max(scale_factors):g}x "
                   f"still improves ITAE (best {100*(1-best_ratio):.1f}% better). "
                   f"The reported fast endpoint is set by {bound_name}, not by the "
                   "hardware -- widening the bound would move it.")
    elif more_kp_helps:
        verdict = (f"{gain_name} is interior to its bounds but scaling it up still helps, "
                   "so this endpoint is limited by the search having converged, "
                   "not by either the bound or the actuator.")
    elif actuator_active:
        verdict = ("ACTUATOR-BOUNDED. The wheels are at their 4e-3 N m torque "
                   f"limit for {100*saturation:.1f}% of the run and scaling {gain_name} up "
                   f"does not improve ITAE (best {best_ratio:.3f}x). The gene "
                   "bound is irrelevant here -- widening it changes nothing, "
                   "because torque saturation binds first.")
    else:
        verdict = (f"Neither bound is active: {gain_name} is interior, the wheels are not "
                   f"saturating, and more {gain_name} does not help. This endpoint is a "
                   "genuine interior optimum of the ITAE objective.")

    return {
        "gain_under_test": gain_name,
        "kp_normalized_position": kp_pos.tolist(),
        "kp_pinned_at_upper_bound": kp_pinned,
        "pin_threshold": PIN_THRESHOLD,
        "saturation_fraction": saturation,
        "actuator_limit_active": actuator_active,
        "baseline_itae": baseline_itae,
        "scaled_kp_trials": trials,
        "more_kp_still_helps": more_kp_helps,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# merged front + variance
# --------------------------------------------------------------------------

def seed_summary(runs):
    """One row per seed: generations, evaluations, final HV, front size,
    why it stopped."""
    rows = []
    for path, data in runs:
        hist = data["history"]
        hv = [h.get("hypervolume", np.nan) for h in hist]
        rows.append({
            "seed": data.get("seed"),
            "path": str(path),
            "generations": int(hist[-1]["generation"]) if hist else 0,
            "evaluations": int(hist[-1].get("n_evaluations", 0)) if hist else 0,
            "wall_s": float(hist[-1].get("elapsed_s", np.nan)) if hist else np.nan,
            "front_size": int(len(data["obj"])),
            "final_hypervolume": float(hv[-1]) if hv else np.nan,
            "stop_reason": data.get("stop_reason", "unknown"),
        })
    return rows


def merge_seed_fronts(runs):
    """Pool every seed's front and keep the non-dominated survivors.

    Also reports, per seed, how many of its points survived into the merged
    front. That contribution count IS the variance statement: if one seed
    supplies almost everything, the others converged to a strictly worse
    surface and the result is seed-sensitive; if all three contribute
    comparably, the trade surface is reproducible.
    """
    objs = [np.atleast_2d(d["obj"]) for _, d in runs]
    genes = [np.atleast_2d(d["pop"]) for _, d in runs]
    obj, pop, source = O.merge_fronts(objs, genes)

    # `source` from merge_fronts is a POSITIONAL index into `runs` (0, 1, 2,
    # ...), not the GA's actual seed number -- those only coincide when every
    # planned seed succeeds. The moment one fails (overnight_tune.py isolates
    # seed failures and keeps going -- see its run_seed try/except), the
    # loaded `runs` list simply omits it, so position 2 can be seed 3's data
    # while seed 2 contributed nothing. Mapping through the real seed number
    # once here, rather than downstream, means the CSV column, picks.json and
    # the markdown can never attribute a result to the wrong (or a failed)
    # seed.
    seed_of_position = np.array(
        [data.get("seed", i) for i, (_, data) in enumerate(runs)])
    seed_ids = seed_of_position[source]

    contributions = {}
    for i, (path, data) in enumerate(runs):
        label = f"seed {data.get('seed', i)}"
        contributions[label] = {
            "front_size": int(len(objs[i])),
            "survived_into_merged": int(np.sum(source == i)),
            "share_of_merged": float(np.mean(source == i)) if len(source) else 0.0,
        }
    return obj, pop, seed_ids, contributions


# --------------------------------------------------------------------------
# markdown report
# --------------------------------------------------------------------------

def _wie_scalars(decoded):
    return tuple(float(np.asarray(b).ravel()[0]) for b in decoded)


def _fmt_gains(decoded, ctype):
    if ctype == "WIE":
        k, d, ki = _wie_scalars(decoded)
        # omega_n and zeta are the reader-facing translation of (k, d): the
        # search operates on the scale factors, but "critically damped,
        # settles in 12 s" is what anyone reviewing the design will ask for.
        # Inverting the same relation design() uses forward.
        omega_n = np.sqrt(k / 2.0)
        zeta = d / (2.0 * omega_n) if omega_n > 0 else float("nan")
        t_s = 8.0 / (zeta * omega_n) if (zeta > 0 and omega_n > 0) else float("nan")
        return (f"k  = {k:.6e}     (K = k*J)\n"
                f"d  = {d:.6e}     (D = d*J)\n"
                f"ki = {ki:.6e}     (Ki = ki*diag(J),  ki/k = {ki/k:.3e})\n"
                f"mu = 1           (eigenaxis case: gyroscopic term cancelled)\n"
                f"\nequivalent sizing-rule design:  omega_n = {omega_n:.4f} rad/s, "
                f"zeta = {zeta:.4f}, t_s = 8/(zeta*omega_n) = {t_s:.2f} s")

    kp, kd = decoded[0], decoded[1]
    txt = (f"Kp = [{kp[0]:.4e}, {kp[1]:.4e}, {kp[2]:.4e}]\n"
           f"Kd = [{kd[0]:.4e}, {kd[1]:.4e}, {kd[2]:.4e}]")
    if ctype == "PID":
        ki = decoded[2]
        ratio = np.asarray(ki) / np.asarray(kp)
        txt += (f"\nKi = [{ki[0]:.4e}, {ki[1]:.4e}, {ki[2]:.4e}]"
                f"   (Ki/Kp = [{ratio[0]:.3e}, {ratio[1]:.3e}, {ratio[2]:.3e}])")
    return txt


def _yaml_block(decoded, ctype, J=None):
    if ctype == "WIE":
        k, d, ki = _wie_scalars(decoded)
        lines = ["controller:", "  type: wie", "  case: eigenaxis",
                 f"  k_scale: {k:.6e}", f"  d_scale: {d:.6e}"]
        if J is not None:
            # Ki is written out ALREADY multiplied by diag(J), because that
            # is what WieRegulator consumes -- it applies Ki elementwise, so
            # a bare scalar here would produce a non-J-matched integral term
            # and quietly break the eigenaxis property the rest of the
            # preset exists to guarantee. See objectives._apply_wie_gains.
            Jd = np.diag(np.asarray(J, dtype=float))
            kv = ki * Jd
            lines.append(f"  Ki: [{kv[0]:.6e}, {kv[1]:.6e}, {kv[2]:.6e}]"
                         f"   # = ki*diag(J), ki = {ki:.6e}")
            lines.append("  integral_limit: 1.0e-3")
        return "\n".join(lines)

    kp, kd = decoded[0], decoded[1]
    lines = ["controller:", f"  type: {ctype}",
             f"  Kp: [{kp[0]:.6e}, {kp[1]:.6e}, {kp[2]:.6e}]",
             f"  Kd: [{kd[0]:.6e}, {kd[1]:.6e}, {kd[2]:.6e}]"]
    if ctype == "PID":
        ki = decoded[2]
        lines.append(f"  Ki: [{ki[0]:.6e}, {ki[1]:.6e}, {ki[2]:.6e}]")
        lines.append("  integral_limit: 1.0e-3")
    return "\n".join(lines)


def write_report(out_dir, runs, base_configs, controller_type="PD",
                  duration=120.0, run_meta=None):
    """The full post-run writeup: merged front, per-seed variance, picks with
    their re-simulated time-domain metrics, and the bound diagnostic.

    Returns the path to results.md.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctype = O._normalize_type(controller_type)
    decode = O.make_decode(ctype)
    labels = O.OBJECTIVE_LABELS[ctype]
    configs = (list(base_configs) if isinstance(base_configs, (list, tuple))
               else [base_configs])

    rows = seed_summary(runs)
    obj, pop, source, contributions = merge_seed_fronts(runs)
    picks = O.pareto_picks(obj, ctype)

    # --- merged front CSV -------------------------------------------------
    gain_names = O.absolute_gain_labels(ctype)
    absolute = np.array([np.concatenate(decode(ind)) for ind in pop])
    np.savetxt(out_dir / "pareto_front_merged.csv",
               np.hstack([source[:, None], obj, absolute]), delimiter=",",
               header=",".join(["seed_index"] + list(labels) + gain_names),
               comments="")

    # --- re-simulate the picks on every scenario --------------------------
    pick_rows = {}
    for label, idx in picks.items():
        decoded = decode(pop[idx])
        per_scenario = []
        for cfg in configs:
            c = O.apply_gains(copy.deepcopy(cfg), decoded, ctype)
            c.sim.duration = duration
            c.name = f"{label}@{cfg.name}"
            try:
                r = simulate(c)
                st = r.settling_time(1.0)
                per_scenario.append({
                    "scenario": cfg.name,
                    "settling_time_s": (None if st is None else float(st)),
                    "final_error_deg": r.final_attitude_error_deg(),
                    "overshoot_deg": r.overshoot_deg(),
                    "itae": O.itae(r.attitude_error_deg, r.t),
                    "control_effort": r.control_effort(),
                    "saturation_fraction": r.saturation_fraction(),
                    "peak_wheel_torque": r.peak_wheel_torque(),
                    "torque_margin": r.torque_margin(),
                    "max_wheel_speed_rpm": r.max_wheel_speed() * 60.0 / (2 * np.pi),
                    "mean_eigenaxis_deviation_deg":
                        r.mean_eigenaxis_deviation_deg(),
                })
            except Exception as e:      # a pick should never fail, but a
                per_scenario.append({"scenario": cfg.name, "error": str(e)})
        pick_rows[label] = {
            "index": int(idx),
            "seed_index": int(source[idx]),
            "objectives": [float(v) for v in obj[idx]],
            "gains": (dict(zip(("k", "d", "ki"), _wie_scalars(decoded)))
                      if ctype == "WIE" else
                      {"Kp": decoded[0].tolist(), "Kd": decoded[1].tolist(),
                       **({"Ki": decoded[2].tolist()} if ctype == "PID" else {})}),
            "normalized_gene_position":
                O.normalized_gene_position(decoded, ctype).tolist(),
            "per_scenario": per_scenario,
        }

    # --- bound diagnostic on the fast endpoint ----------------------------
    fast_idx = picks["best_tracking"]
    diag = bound_diagnostic(configs, decode(pop[fast_idx]), ctype, duration=duration)

    with open(out_dir / "picks.json", "w") as f:
        json.dump({"picks": pick_rows, "bound_diagnostic": diag,
                   "seeds": rows, "contributions": contributions,
                   "meta": run_meta or {}}, f, indent=2)

    # --- markdown ---------------------------------------------------------
    L = []
    L.append(f"# NSGA-II {ctype} gain search — results\n")
    if run_meta:
        L.append("## Run configuration\n")
        for k, v in run_meta.items():
            L.append(f"- **{k}**: {v}")
        L.append("")

    L.append("## Objectives\n")
    for i, lab in enumerate(labels):
        L.append(f"{i+1}. {lab}")
    L.append("")
    L.append("The first objective is ITAE, `∫ t·|θ_err| dt`, not a thresholded "
             "settling time. A settling time capped at the run duration is a "
             "constant for every candidate that never settles, so the whole "
             "non-settling part of the population ties and selection among "
             "them is random. ITAE is finite and strictly ordered for every "
             "candidate, and its `t` weight makes late error expensive, so "
             "non-settling runs score badly by construction rather than by a "
             "tuned penalty constant.\n")

    L.append("## Per-seed runs\n")
    L.append("| seed | generations | evaluations | wall (min) | front | final HV | stopped because |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['seed']} | {r['generations']} | {r['evaluations']} | "
                 f"{r['wall_s']/60:.1f} | {r['front_size']} | "
                 f"{r['final_hypervolume']:.6f} | {r['stop_reason']} |")
    hvs = np.array([r["final_hypervolume"] for r in rows], dtype=float)
    if np.isfinite(hvs).sum() > 1:
        L.append(f"\nFinal hypervolume across seeds: **{np.nanmean(hvs):.6f} ± "
                 f"{np.nanstd(hvs):.6f}** (spread "
                 f"{100*np.nanstd(hvs)/np.nanmean(hvs):.2f}% of the mean).\n")
        L.append("Hypervolume is measured against a FIXED reference point — the "
                 "worst-case objective tuple, which depends only on controller "
                 "type and evaluation duration — so these numbers are directly "
                 "comparable across seeds and across reruns.\n")

    # --- per-generation record for the longest-running seed --------------
    from deimos.tuning.logbook import format_table
    longest = max(runs, key=lambda r: len(r[1]["history"]))
    L.append("## Generation-by-generation record\n")
    L.append(f"Seed {longest[1].get('seed')} (the longest run of the set). "
             f"Full per-generation data — including every Pareto-front "
             f"individual's gains and metrics — is in each seed's "
             f"`generations_summary.csv`, `generations_front.csv` and "
             f"`generations_full.csv`.\n")
    L.append("```")
    L.append(format_table(longest[1]["history"], ctype, last=30))
    L.append("```\n")
    L.append("Settling time, steady-state error and control effort are "
             "recorded for every individual but steer nothing — the search "
             "sorts on ITAE. They are here because they are the numbers a "
             "reader understands, and because 'how many individuals settled' "
             "rising from zero is the most legible evidence the search is "
             "working (it is invisible in objective space, precisely because "
             "ITAE never ties).\n")

    L.append("## Merged front\n")
    L.append(f"Pooling all seeds' fronts and keeping the non-dominated "
             f"survivors gives **{len(obj)}** solutions.\n")
    L.append("| seed | own front | survived into merged | share |")
    L.append("|---|---|---|---|")
    for label, c in contributions.items():
        L.append(f"| {label} | {c['front_size']} | {c['survived_into_merged']} | "
                 f"{100*c['share_of_merged']:.1f}% |")
    L.append("\nA seed contributing almost nothing to the merged front "
             "converged to a strictly worse trade surface; roughly even "
             "contributions mean the surface is reproducible and not an "
             "artefact of one lucky trajectory.\n")

    L.append("## Selected solutions\n")
    for label, row in pick_rows.items():
        decoded = decode(pop[row["index"]])
        L.append(f"### {label}  (from seed {row['seed_index']})\n")
        L.append("```")
        L.append(_fmt_gains(decoded, ctype))
        L.append("```")
        L.append("| " + " | ".join(labels) + " |")
        L.append("|" + "---|" * len(labels))
        L.append("| " + " | ".join(f"{v:.4e}" for v in row["objectives"]) + " |")
        L.append("")
        L.append("Re-simulated per scenario:\n")
        L.append("| scenario | settle (s) | final err (deg) | overshoot (deg) | "
                 "effort (N m s) | saturation | peak wheel τ | max speed (rpm) | "
                 "eigenaxis dev (deg) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s in row["per_scenario"]:
            if "error" in s:
                L.append(f"| {s['scenario']} | simulation failed: {s['error']} | | | | | | | |")
                continue
            st = "did not settle" if s["settling_time_s"] is None else f"{s['settling_time_s']:.2f}"
            L.append(f"| {s['scenario']} | {st} | {s['final_error_deg']:.4f} | "
                     f"{s['overshoot_deg']:.3f} | {s['control_effort']:.4e} | "
                     f"{100*s['saturation_fraction']:.2f}% | "
                     f"{s['peak_wheel_torque']:.3e} ({100*s['torque_margin']:.0f}%) | "
                     f"{s['max_wheel_speed_rpm']:.0f} | "
                     f"{s.get('mean_eigenaxis_deviation_deg', float('nan')):.2f} |")
        L.append("")
        L.append("Drop-in config block:\n")
        L.append("```yaml")
        L.append(_yaml_block(decoded, ctype,
                             J=configs[0].satellite.inertia_tensor))
        L.append("```\n")

    L.append("## Is the fast endpoint gene-bounded or actuator-bounded?\n")
    gname = diag.get("gain_under_test", "Kp")
    L.append(f"**{diag['verdict']}**\n")
    L.append(f"- {gname} position inside its log-uniform search range (0 = lower "
             f"bound, 1 = upper): "
             f"{np.round(diag['kp_normalized_position'], 4).tolist()}")
    L.append(f"- Pinned at the upper bound (>= {diag['pin_threshold']}): "
             f"{diag['kp_pinned_at_upper_bound']}")
    L.append(f"- Worst-case torque saturation over the scenarios: "
             f"{100*diag['saturation_fraction']:.2f}% of timesteps "
             f"(actuator limit considered active above "
             f"{100*SATURATION_ACTIVE:.0f}%)")
    L.append(f"- Baseline ITAE: {diag['baseline_itae']:.4e}")
    L.append("")
    L.append(f"| {gname} scaled by | ITAE | vs baseline |")
    L.append("|---|---|---|")
    for t in diag["scaled_kp_trials"]:
        L.append(f"| {t['kp_scale']:g}x | {t['itae']:.4e} | {t['itae_ratio']:.3f}x |")
    L.append("")
    bname = "K_SCALE_BOUNDS" if ctype == "WIE" else "KP_BOUNDS"
    L.append(f"The scaling trial is the decisive test. Pushing {gname} past the top "
             "of its search range and re-simulating answers directly whether "
             "the bound was binding: if ITAE improves, the search wanted more "
             f"gain and the bound refused it, so the endpoint is an artefact of "
             f"{bname}. If ITAE does not improve while the wheels are "
             "saturating, the 4e-3 N m torque limit binds first and widening "
             "the bound would change nothing.\n")
    L.append("Note that gains pinning at a bound is by itself the *expected* "
             "signature of a Pareto extreme, not evidence of a badly chosen "
             "decode range — the minimum-effort endpoint should sit at minimum "
             f"{gname}, and the fastest endpoint should push {gname} up until "
             "something stops it. The question is only what stops it.\n")

    if ctype == "WIE":
        L.append("### Did the eigenaxis property survive?\n")
        L.append("Wie Case 1 is eigenaxis *by construction* — but only for an "
                 "exact inertia tensor, unsaturated wheels and no external "
                 "torque. This run has a gravity-gradient bias enabled and a "
                 "4e-3 N m per-wheel limit, so the measured deviation of ω "
                 "from the initial error eigenaxis is the cost of running the "
                 "ideal law on real hardware. It is reported per pick in the "
                 "re-simulation tables above, and per individual per "
                 "generation in `generations_front.csv` "
                 "(`mean_eigenaxis_deviation_deg`).\n")
        L.append("Read it against the saturation column: a fast endpoint with "
                 "a large deviation bought its speed by clipping torque and "
                 "leaving the shortest-path trajectory, which is exactly the "
                 "trade the eigenaxis law was chosen to avoid. A knee "
                 "solution holding deviation near zero is the one that "
                 "actually delivers the property.\n")

    path = out_dir / "results.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path
