# DEIMoS

First-principles physics and control simulation for a reaction-wheel 3U
CubeSat ADCS, built for **IITISoC '26 — "Attitude Control of a Satellite
Using Reaction Wheels."** No external attitude-control libraries: the
quaternion algebra, rigid-body dynamics, RK4 propagator, control laws,
wheel allocation and magnetorquer desaturation are all hand-derived and
tested (155 pytest cases).

```bash
pip install -e ".[dev]"
pytest
```

## Competition deliverables, mapped to commands

| Brief requirement | How to reproduce it |
|---|---|
| **Scenario A** — large-angle slew + hold | `deimos run --scenario configs/scenarios/slew_40_30_25.yaml --controller configs/controllers/wie_eigenaxis.yaml` |
| **Scenario B** — attitude hold under constant disturbance | `deimos run --scenario configs/scenarios/hold_under_disturbance.yaml --controller configs/controllers/pid_example.yaml` |
| Attitude vs time / attitude error vs time | `attitude_error`, `quaternion_components`, `error_quaternion` figures |
| Wheel speeds vs time / per-wheel torque vs time | `wheel_panel` figure (both stacked), `saturation` for the limit timeline |
| Saturation analysis | `saturation`, `torque_envelope`, `wheel_momentum` figures + the summary's saturation percentages — speed saturation is **enforced plant physics** (a pinned wheel accepts no accelerating torque), not just a flag |
| **Bonus:** singularity-free attitude representation | quaternions throughout (`src/deimos/math/quaternion.py`), scalar-first, renormalised each RK4 step |
| **Bonus:** recover from wheel saturation with a second actuator | magnetorquer momentum dumping: `deimos run --scenario configs/scenarios/desat_recovery.yaml --controller configs/controllers/wie_eigenaxis.yaml --plots desat` |
| **Bonus:** realistic orbital disturbances | gravity-gradient bound + residual-dipole envelope (`dynamics/disturbances.py`, Scenario B header) and a dipole geomagnetic field model (`dynamics/environment.py`) |
| **Bonus:** 3D animation of the manoeuvre | `deimos animate --scenario configs/scenarios/slew_40_30_25.yaml --controller configs/controllers/wie_eigenaxis.yaml --save slew.mp4` (GIF fallback without ffmpeg; `--stl your_cad.stl` renders the CAD mesh — `pip install -e ".[viz3d]"`) |
| **Bonus:** optimal-control comparison vs baseline | `deimos tune` (NSGA-II Pareto search, below) + `deimos compare` across presets |

Every `run`/`compare`/`tune` writes a timestamped directory under `runs/`
with the exact composed config, a git-SHA/versions manifest, metrics JSON
and figures — each report number is regenerable from one command.

## Single source of truth

**Everything we are not tuning lives in [`src/deimos/constants.py`](src/deimos/constants.py)**:
the CAD inertia tensor, wheel and magnetorquer hardware limits, orbit,
solar-array geometry, the power budget, sim defaults. Config composition
(`sim/config.py`) derives its base layer from that module, so a YAML file
only ever carries what an experiment *changes* — a scenario (initial/target
attitude, duration, disturbance, magnetorquers) or a controller (gains/case).
There used to be a `configs/_base.yaml` mirroring constants by hand; it
drifted (two different inertia tensors in circulation) and was deleted.
`tests/test_single_source.py` guards the contract.

Values not yet measured are flagged `ASSUMPTION` in constants.py — currently:
spacecraft mass, magnetorquer dipole/power (iMTQ-class placeholder), orbit
inclination (SSO), battery capacity/bus voltage. Replace them there and
every simulation, plot and card follows.

## Layout

```
src/deimos/
    constants.py     THE mission parameters file -- single source of truth
    math/             quaternion algebra, kinematics
    dynamics/         Euler's equation, RK4 propagator, disturbances,
                      geomagnetic dipole field (environment.py)
    actuators/        reaction wheel array (min-norm allocation, torque +
                      enforced speed saturation), magnetorquers (cross-
                      product desaturation law with hysteresis)
    control/          PD / PID / Wie regulator + a registry (name -> class)
    sim/              config composition, the simulate() entry point, results
    analysis/         metrics, stability proofs, design card, power card
    viz/              single-run + comparison figures, 3D attitude
                      animation (attitude3d.py, STL-capable)
    tuning/           NSGA-II multi-objective gain search
    cli.py            deimos run / compare / tune / animate

configs/
    scenarios/*.yaml       what happens: initial/target, duration,
                           disturbance, magnetorquers
    controllers/*.yaml     who flies it: one controller's gains/case

studies/                jupyter notebooks (manual tuning, GA studies, report figures)
tests/                  pytest -- 155 cases, physics properties included
runs/                   command output, gitignored
legacy/                 pre-package scripts, see legacy/README.md
```

## The control laws

| Preset | Law | Why it's here |
|---|---|---|
| `wie_eigenaxis` | Wie/Weiss/Arapostathis (1989) Case 1, mu=1, K=kJ | shortest-path (eigenaxis) slew, globally stable for any k,d>0; includes wheel-momentum decoupling (measured: 3.65° → 0.005° mean eigenaxis deviation) |
| `wie_case3` | Case 3, mu=0, K⁻¹=αJ+βI | robust to inertia error, near-minimum path |
| `pd_baseline` / `pd_aggressive` | quaternion PD | baseline + GA warm start |
| `pid_example` | quaternion PID with anti-windup | disturbance rejection (Scenario B) |

Gains proportional to J were rescaled per-axis when the plant moved to the
3U CAD tensor (2026-07, see preset headers); the Wie presets derive K, D
from `constants.INERTIA_TENSOR` at build time and needed nothing.

## Magnetorquer desaturation (bonus extension)

The wheels store every disturbance impulse they absorb; at 12 000 rpm a
wheel pins and that axis is lost. `configs/scenarios/desat_recovery.yaml`
starts the array at 72 % speed and lets three orthogonal torquer rods bleed
~8e-3 N m s of momentum back out through the geomagnetic field over ~40 min,
while the wheels keep the body pointed (attitude error stays < 1° — proven in
`tests/test_magnetorquers.py`). Law, sign conventions and engagement
hysteresis: `src/deimos/actuators/magnetorquers.py`.

## Power (EPS cross-check)

`deimos run` prints a **power card** after every simulation: wheel + torquer
electrical energy for the manoeuvre against the 2.0 W ADCS allocation, the
8.5 W orbit-average generation (0.1024 m² GaAs array, team EPS sizing) and
battery depth-of-discharge. Budget numbers live in `constants.py`'s POWER
section; the model is `analysis/power.py` (conservative: no regenerative
braking credit, resistive torquer rods).

## Gain tuning (NSGA-II)

`deimos tune` searches per-axis gains with a multi-objective GA and returns a
Pareto front rather than one answer, so the performance/cost trade is explicit
instead of hidden in a threshold choice.

| Law | Genes | Objectives (all minimised) |
|---|---|---|
| `--type PD` | 6 — `Kp`, `Kd` | settling time, control effort, saturation fraction |
| `--type PID` | 9 — `Kp`, `Kd`, `Ki/Kp` | IAE, steady-state error, control effort |
| `--type WIE` | 3 — `k`, `d`, `ki/k` | ITAE, steady-state error, control effort |

Three design points that are easy to get wrong, and are the reason the rows
differ:

- **`Ki` is searched as a ratio of `Kp`, not as an absolute gain.**
  `control/pid.py`'s design rule is "keep `Ki` small relative to `Kp`", which
  an absolute bound cannot express when `Kp` itself spans three decades in the
  same search — an absolute `Ki` of 1e-3 is negligible against `Kp`=1e-1 and
  catastrophic against `Kp`=1e-4.
- **PID does not use settling time.** A settling time capped at the run
  duration is a *constant* for every candidate that never settles, which
  silently collapses the 3-objective search into a 2-objective one and makes
  `argmin` over that column meaningless. IAE is finite and strictly ordered for
  every candidate. Settling time is still reported, just not searched on.
- **Anti-windup is required for the PID search to mean anything.** Unbounded,
  the integral accumulator integrates the entire slew transient and at the top
  of the `Ki/Kp` range demands several times the wheel torque limit; every
  candidate limit-cycles and the GA returns gains worse than an untuned PD.
  `deimos tune --type PID` therefore caps the integral term's contribution at
  25% of the per-wheel limit (`controller.integral_limit`, opt-in elsewhere and
  off by default so existing presets are unchanged).

Integral action only earns its keep against a persistent bias, so `tune`
applies a gravity-gradient disturbance by default for PID
(`--no-gravity-gradient` to disable); on a disturbance-free slew the correct
answer is `Ki`→0, i.e. a PD.

**The search is warm-started** from the gains already in `--controller`.
NSGA-II is elitist, so a seeded individual can only be displaced by something
that dominates it — meaning the returned front is never worse than the preset
you started from. Budget is `pop_size x (generations + 1)` simulations — the
defaults (24 x 21 = 504) are the practical minimum for 9-D. For the serious
multi-seed, multi-scenario overnight run see `OVERNIGHT_RUNBOOK.md`
(`python overnight_tune.py`). **Note:** GA fronts recorded before the 3U
tensor swap (2026-07) were tuned against the old plant — re-run to refresh.

Figures land in `runs/<stamp>_tune_*/figures/`: the Pareto front (3D and
pairwise projections), NSGA-II convergence, a parallel-coordinates view of
where each gain sits within its search bounds, and the re-simulated time
response of every pick against the untuned baseline. `pareto_front.csv` carries
the whole front with absolute gains so a different pick can be made later
without re-running.

## Migration notes (from the pre-package `ADCS src` + `dashboard` tree)

This package was assembled from two previously separate, un-versioned
directories (`ADCS src`, flat physics/control modules; `dashboard`,
sim/config/plotting/tuning) plus a partially-started GitHub skeleton. A few
things were reconciled in the process, worth knowing before trusting an old
number you find quoted elsewhere:

- **Inertia tensor.** Three different `Iyy` values were in circulation
  during the migration; the confirmed value then (`1.4130e-2`) was itself
  superseded 2026-07 by the current 3U CAD export
  (`diag ≈ [3.605e-2, 3.450e-2, 1.016e-2]`, Fusion 360 "3U_Cubesat_Master").
  `constants.py` carries the one live tensor; nothing else does.
- **Wheel hardware.** The old `ADCS src/ReactionWheelArray.py` (4-wheel
  pyramid, `wheel_inertia=2.07e-5`, `max_speed=1570.8`) and the further-along
  GitHub-repo version (3-wheel orthogonal cone, `wheel_inertia=5.8e-6`,
  `max_speed=1256.6`, CAD/datasheet-sourced) disagreed. The cone/GitHub
  numbers were taken as current (`actuators/reaction_wheels.py`,
  `constants.py`); the pyramid geometry is kept as an option
  (`wheels.config: pyramid`) but is no longer the default.
- **GA tuning had two copies.** `dashboard/core/run_ga_tuning.py` and
  `tune_pd_ga.py` were near-duplicates; `tune_pd_ga.py`'s own comments
  document that it read a nonexistent `results.wheel_torque` field and
  silently scored every candidate as 100% saturated. `run_ga_tuning.py`'s
  fixed version became `tuning/objectives.py`; `tune_pd_ga.py` is parked in
  `legacy/`.
- **`control/registry.py`** replaces `simulate.py`'s old `if/elif` controller
  dispatch. Every controller now implements a common
  `compute_torque(q, omega, q_target, dt=None)` (see `control/base.py`).
- **The 3D visualizer** (`AttitudeVisualizer.py`/`STLLoader.py`) was ported
  2026-07 into `viz/attitude3d.py` + `viz/stl_mesh.py`, driven by SimResults
  (`deimos animate`).

Anything not listed above (math, kinematics, rigid-body dynamics, the RK4
propagator, the allocation/saturation logic, the Wie regulator's four design
cases, every figure) is a direct port with only import paths changed.
