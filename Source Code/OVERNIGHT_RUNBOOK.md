# Overnight GA runbook

## Tonight: the eigenaxis run

**Terminal 1 — start the run** (from the `deimos/` directory):

```
python overnight_tune.py --type WIE --hours 6
```

That tunes the **Wie Case 1 eigenaxis regulator** rather than a PID. Three
genes instead of nine — `k`, `d` and `ki`, giving `K = k*J`, `D = d*J`,
`mu = 1` — because the eigenaxis property is a consequence of `K` being a
scalar multiple of `J`, not of the gain values. Six per-axis gains would
search a strictly larger space that no longer contains an eigenaxis
controller. Same objectives as the PID search (ITAE, steady-state error,
control effort, saturation as a constraint), on purpose, so the two fronts
can be overlaid.

Warm-started from `configs/controllers/wie_eigenaxis.yaml` (zeta = 1.0,
t_s = 20 s). NSGA-II is elitist, so the returned front cannot be dominated by
that textbook design — the search can only improve on the sizing rule.

### One thing that changed in the plant-facing code, and why it mattered

The paper assumes an ideal torque actuator: `J*omega_dot = u - omega x J*omega`,
which `mu = 1` cancels outright. DEIMoS is a momentum-exchange system, so the
wheels' own stored momentum sits inside that cross product too:

```
J*omega_dot = u - omega x (J*omega + h_w)
```

Cancelling only the body part left `-omega x h_w` — a torque that is not along
the error eigenaxis, and that grows through the slew because `h_w` is exactly
what the wheels accumulate to perform it. Measured on `slew_40_30_25`:

| | mean eigenaxis deviation | settling time |
|---|---|---|
| body-only decoupling (before) | **3.65 deg** | 14.8 s |
| total-momentum decoupling (now) | **0.005 deg** | 13.3 s |

Three orders of magnitude on the property that defines the control law, and
almost nothing on the number anyone normally checks. `WieRegulator` now takes
`decouple_wheel_momentum`, `sim/runner.py` feeds it `h_w`, and the GA forces it
on for every candidate. It defaults **off** everywhere else so `wie_case3.yaml`
and every result already recorded with it are untouched.
`tests/test_wie_eigenaxis.py` guards the difference in both directions.

### Everything else still works the same way

```
python overnight_tune.py                    # 8 hours, 3 seeds, PID, 2 scenarios
```

That is the whole command. Defaults are already what you asked for: PID (all
nine genes — Kp, Kd, Ki), 8 hours, seeds 0/1/2 plus up to two extra if the
budget outlasts them, population 60, worst-case over `slew_40_30_25` and
`slew_55_65_15`, gravity-gradient bias on, all cores but one.

Before it commits the night it runs a **preflight**: one candidate evaluated
on the real configs, timed. If a config is broken you find out in thirty
seconds, not at 7 a.m. It then prints how many generations the budget
actually buys.

**Terminal 2 — watch it:**

```
python -m deimos.tuning.monitor_ga runs/overnight --controller-type WIE
```

(`--controller-type` must match what you launched, or the gene panels decode
three scalars as if they were nine per-axis gains.)

Fifteen live panels: hypervolume with the stall rule that will stop the run,
front size, a status block (generation, evaluations, s/gen, deadline, best
metrics so far), best vs population median for each objective, **settling
time / steady-state error / control effort per generation**, the three Pareto
projections, the front in gain space as parallel coordinates, **each gain's
trajectory across generations**, and a cross-seed hypervolume overlay. It is
read-only — closing the window does not touch the run.

Leave a PNG updating instead of a window (checkable from your phone):

```
python -m deimos.tuning.monitor_ga runs/overnight --save runs/overnight/live.png --no-window
```

### The numbers as text, any time

```
python -m deimos.tuning.logbook runs/overnight/seed0/checkpoint.pkl --last 30
```

prints a per-generation table — generation, front size, how many individuals
settled, hypervolume, and the best settling time / steady-state error /
control effort / ITAE / saturation reached. Add `--front -1` to dump the
latest generation's whole Pareto front with each solution's Kp, Kd, Ki and its
metrics, or `--picks -1` for just the compact fastest/cheapest/knee summary.
Add `--csv somedir` to write the CSVs anywhere. Safe to run against a
checkpoint while the GA is still going.

### A third terminal, updating live as generations land

```
python -m deimos.tuning.logbook runs/overnight/seed0/checkpoint.pkl --controller-type WIE --watch
```

Prints a new block the moment each generation finishes — no reprinting, just
new output appended, like a log file growing. This is the compact
`[best_tracking] Kp=... Kd=... Ki=... (settle=...s sse=... effort=... sat=...)`
view, one line per representative solution. Add `--watch-mode front` for the
whole Pareto front every generation instead of just the four picks, or
`--interval 5` to poll faster than the default 10s. Ctrl+C stops the watcher
only — the GA keeps running.

## In the morning

Read `runs/overnight/results.md` first. It contains:

- per-seed table: generations, evaluations, wall time, final hypervolume, why each seed stopped
- final hypervolume **mean ± std across seeds** — the reproducibility number
- the merged front across all seeds, and how much each seed contributed to it
- the four picks (best_tracking / most_accurate / cheapest_effort / knee) with
  re-simulated settling time, overshoot, saturation and peak wheel torque on
  **both** scenarios, plus a copy-pasteable YAML block for each
- the **gene-bounded vs actuator-bounded verdict** for the fast endpoint
  (phrased in terms of `k` for a WIE run, and the scaling trial scales `k` and
  `ki` together so the trial candidate stays a valid Case 1 controller)
- for a WIE run only, a section titled **"Did the eigenaxis property
  survive?"** — Case 1 is eigenaxis by construction only for exact `J`,
  unsaturated wheels and no external torque, and this run has a
  gravity-gradient bias and a 4e-3 N m wheel limit. Read
  `mean_eigenaxis_deviation_deg` against the saturation column: in the smoke
  run the fast endpoint hit **26 deg** of deviation at 6% saturation while an
  unsaturated knee solution held **0.05 deg**. That is the real trade this
  search surfaces, and it is the number that says whether a given point on
  the front actually delivers the property you implemented the law for

Also written, per seed:

| file | one row per | what it answers |
|---|---|---|
| `generations_summary.csv` | generation | best and median settling time, steady-state error, control effort, ITAE, overshoot, saturation, peak wheel torque, wheel speed — plus hypervolume, how many individuals settled, and wall time. This is the file to plot from. |
| `generations_front.csv` | Pareto-front individual, per generation | "what were the gains at generation 40, and what did each of them actually do" — absolute Kp/Kd/Ki with that solution's metrics |
| `generations_full.csv` | every individual, per generation | the same for the whole population including everything rejected — the only record of the shape of the search space, not just its optimum |
| `pareto_front.csv` | final front solution | gains + objectives + all diagnostics |
| `figures/ga_*.png` | — | 8 figures, including `ga_diagnostic_history.png` (settling time, steady-state error, effort, overshoot, saturation, peak torque per generation) and `ga_gain_history.png` (each gain's front min–max band and median vs generation, against its search bounds) |

And at the run level: `results.md`, `pareto_front_merged.csv`, `picks.json`,
`figures/seed_comparison.png`, `run_summary.json`.

Gains in every CSV are **absolute** — the Kp/Kd/Ki the controller uses, not the
internal `[0,1]` gene or the Ki/Kp ratio the search operates on — so a row can
be pasted into a config with no conversion.

Everything is written **after every seed**, not at the end. If the laptop dies
at 4 a.m., the seeds that finished are complete.

## What changed, and why

**ITAE replaces settling time as the first objective.** A settling time capped
at the run duration is a sentinel: every non-settling individual scores
*exactly* the cap, so they all tie and selection among them is random —
precisely when the search most needs a gradient. `ITAE = ∫ t·|θ_err| dt` is
finite and strictly ordered for every candidate. The `t` weight makes late
error expensive and early error nearly free, so it is a settling measure, not
just an error measure, and non-settling runs score badly by construction with
no branch and no tuned penalty constant. Settling time is still computed and
reported; it just no longer steers the search.

**Hypervolume replaces per-objective bests as the convergence signal.**
Per-objective bests only see the *endpoints* of the front. The front can be
reshaping substantially in its interior — knee moving, gaps filling — while
all three endpoints sit flat, so stopping on flat endpoints stops the run
while it is still doing the work you care about. Hypervolume is monotone with
Pareto dominance, so it responds to convergence, spread and cardinality at
once. The reference point is **fixed** (the worst-case objective tuple, which
depends only on controller type and duration), which is what makes the number
comparable across generations, seeds and reruns; a reference derived from the
current population would make it meaningless.

**Multiple seeds.** One stochastic trajectory is not a result. Three
independent runs give a hypervolume mean ± std and a merged front, and the
per-seed contribution to that merged front is a direct measure of whether the
trade surface is reproducible or an artefact of one lucky run.

**Worst-case over two scenarios.** Each candidate is simulated on both slews
and the objectives are aggregated elementwise **max**. A gain set only scores
well if it scores well on both, which is the direct answer to "are these gains
overfit to one maneuver?". Mean aggregation is deliberately not offered — it
lets a candidate buy a good average by being unacceptable on the hard case.

**Parallel evaluation, a real deadline, and plateau stopping.** Evaluations
within a generation are independent, so the pool gives close to linear
speedup. Each seed has a hard wall-clock deadline and stops at the next
generation boundary; unused budget from a seed that plateaued early is
re-divided among the seeds still to come. A seed that raises is logged and
skipped rather than killing the night.

Deliberately **not** changed, because they were misdiagnoses:

- *Gains pinning at bounds.* That is the expected signature of a Pareto
  extreme, not a decode-range error. The minimum-effort endpoint *should* sit
  at minimum Kp. The real question is what stops the fast endpoint, which is
  now answered explicitly — see below.
- *Front saturating at pop_size.* Normal. Once the whole population is
  mutually non-dominated, crowding distance taking over is the designed
  behaviour. Larger population improves front *resolution*, not correctness,
  and random re-injection fights NSGA-II's elitism.
- *eta_c / eta_m.* Left at 15 / 20.

## The viva question, answered automatically

`results.md` ends with a section titled *"Is the fast endpoint gene-bounded or
actuator-bounded?"*. It gathers two independent pieces of evidence:

1. **Static** — where Kp sits inside its log-uniform search range (0 = lower
   bound, 1 = upper), and the worst-case torque saturation over the scenarios.
2. **Active, and decisive** — Kp is scaled 2× and 5× *past* the top of the
   search range and re-simulated. If ITAE improves, the search wanted more
   gain and the bound refused it, so the endpoint is an artefact of
   `KP_BOUNDS`. If ITAE does not improve while the wheels are saturating, the
   4×10⁻³ N·m limit binds first and widening the bound changes nothing.

   (For PID, Ki is scaled with Kp so the Ki/Kp ratio the search operates on
   stays fixed and the trial isolates Kp.)

You get a one-sentence verdict you can quote directly.

## Useful flags

```
--hours 6                  shorter budget
--seeds 0 1                just two seeds
--type WIE                 3 genes (k, d, ki) — Wie Case 1 eigenaxis
--type PD                  6 genes instead of 9
--pop-size 80              denser front, fewer generations
--duration 90              longer evaluation window
--workers 4                cooler/quieter laptop
--scenarios A.yaml B.yaml C.yaml     worst-case over three
--no-gravity-gradient      (don't, for PID — Ki would trend to its lower bound)
--stall-patience 40        more conservative plateau stopping
--skip-preflight
```

## If something looks wrong at 2 a.m.

- **Every candidate at worst-case.** The preflight warns about this explicitly.
  Usually a scenario/controller mismatch.
- **Hypervolume flat from generation 0.** The warm start was already on the
  front and the population has not diversified. Check `ga_objective_spread.png`
  — if the population median never moves, the search is riding a few
  individuals.
- **A seed failed.** Look for `seed<N>_FAILED.txt`. The other seeds continued.
- **Want to stop early.** Ctrl+C. Everything up to the last completed
  generation is already on disk in `seed<N>/checkpoint.pkl`, and the seeds
  that finished have full outputs.
