"""
config.py
=========

Reads YAML into a typed `SimConfig` object. This is the only file that knows
about YAML -- everything downstream (sim/runner.py, viz/) works with plain
Python/numpy via the dataclass fields, never with raw dicts.

Design choice: dataclasses instead of a plain dict. A typo like `Kp` vs `kp`
or a missing field raises immediately at load time with a clear message,
rather than silently propagating into a wrong simulation result you might
not notice until you're staring at a weird plot.

THE BASE LAYER COMES FROM deimos.constants, NOT FROM A YAML FILE. There used
to be a configs/_base.yaml mirroring constants.py by hand; the two drifted
apart (they ended up carrying different inertia tensors -- exactly the
copy-paste failure constants.py exists to prevent), so _base.yaml was
deleted in the 2026-07 restructure and `_base_raw()` below derives the base
layer from constants directly. Change the spacecraft in constants.py and
every config, simulation and report follows.

Two entry points:
  load_config(path)                     a single YAML file carrying everything
                                          the base layer doesn't (initial/
                                          target/controller/sim.duration); it
                                          may also override any base key
  compose_config(scenario, controller)  the configs/ layout: a scenario
                                          (initial/target/duration/disturbance/
                                          magnetorquers) and a controller
                                          block, merged over the base
Both funnel into the same `_parse_raw`, so a composed config and a
self-contained one behave identically once loaded.

A scenario may still override `satellite:`/`wheels:` explicitly (the
inertia-mismatch study in studies/explore.ipynb does) -- the merge just no
longer *requires* every file to restate the spacecraft.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml

from deimos import constants


# --- sub-sections, one dataclass per YAML top-level key -----------------

@dataclass
class SatelliteConfig:
    inertia_tensor: np.ndarray  # (3,3)

@dataclass
class InitialConfig:
    attitude_rpy_deg: np.ndarray  # (3,)
    omega: np.ndarray             # (3,)
    # Optional pre-loaded wheel speeds [rad/s], length = number of wheels.
    # None -> wheels start at rest. The desaturation-recovery scenario uses
    # this to start the array near its momentum capacity.
    wheel_speeds: np.ndarray = None

@dataclass
class TargetConfig:
    attitude_rpy_deg: np.ndarray  # (3,)

@dataclass
class ControllerConfig:
    type: str

    # --- PD / PID fields (required when type == "PD"/"PID", None otherwise) ---
    Kp: np.ndarray = None  # (3,3) diagonal
    Kd: np.ndarray = None  # (3,3) diagonal

    # --- Wie fields (required when type == "wie", None otherwise) ---
    # K and D are not set in YAML: they are *derived* from J by
    # WieRegulator.design(), because the paper's global-stability proof for
    # mu=0 only holds when K is matched to J. Letting a preset hand-set K
    # would silently break that guarantee.
    case: str = None              # eigenaxis | robust | mortensen | near_eigenaxis
    zeta: float = None            # damping ratio
    settling_time_s: float = None # large-angle settling time, t_s = 8/(zeta*omega_n)
    # Alternative to (zeta, settling_time_s), eigenaxis case only: give the
    # scale factors directly, K = k_scale*J and D = d_scale*J with mu = 1.
    # This is the form the NSGA-II search writes (tuning/objectives.py,
    # controller_type="WIE") -- it needs to reach (K, D) pairs that the
    # t_s = 8/(zeta*omega_n) sizing rule cannot express. Exactly one of the
    # two forms may be present; _parse_raw rejects a config carrying both.
    k_scale: float = None         # K = k_scale * J
    d_scale: float = None         # D = d_scale * J
    # Include the wheels' stored momentum in the mu-decoupling term. Required
    # for the eigenaxis case to hold on this momentum-exchange plant -- see
    # control/wie.py. Defaults False so existing mu != 0 presets keep the
    # results they were recorded with.
    decouple_wheel_momentum: bool = False
    Ki: np.ndarray = None         # (3,) optional, opt-in integral action
    u_max: float = None           # [N m] optional, opt-in per-axis saturation (any type)
    integral_limit: float = None  # [N m] optional anti-windup cap on |Ki*z| (PD/PID)

@dataclass
class WheelsConfig:
    config: str
    tilt_deg: float
    wheel_inertia: float
    max_torque: float
    max_speed: float

@dataclass
class MagnetorquersConfig:
    """Desaturation (momentum-dumping) magnetorquers -- see
    actuators/magnetorquers.py for the control law. The hardware ceiling
    (max_dipole) defaults from constants; enabled/k_desat/threshold_frac are
    per-scenario decisions, which is why this block lives at scenario level
    rather than in constants."""
    enabled: bool
    max_dipole: float       # [A m^2] per-axis dipole limit
    k_desat: float          # [1/s] momentum-dump rate gain
    threshold_frac: float   # desat engages when |h_w| exceeds this fraction
                            # of a single wheel's momentum capacity

@dataclass
class DisturbanceConfig:
    enabled: bool
    constant_torque: np.ndarray  # (3,)

@dataclass
class SimSettings:
    dt: float
    duration: float


@dataclass
class SimConfig:
    name: str  # derived from the filename (or passed explicitly), used to label plots/outputs
    satellite: SatelliteConfig
    initial: InitialConfig
    target: TargetConfig
    controller: ControllerConfig
    wheels: WheelsConfig
    magnetorquers: MagnetorquersConfig
    disturbance: DisturbanceConfig
    sim: SimSettings


# --- the base layer, from constants -------------------------------------

def _base_raw() -> dict:
    """The dict every config starts from, derived from deimos.constants --
    the single source of truth for everything we are not tuning. This is
    what configs/_base.yaml used to be, minus the hand-copied drift."""
    return {
        "satellite": {
            "inertia_tensor": constants.INERTIA_TENSOR.ravel().tolist(),
        },
        "wheels": {
            "config": constants.WHEEL_CONFIG,
            "tilt_deg": constants.WHEEL_TILT_DEG,
            "wheel_inertia": constants.WHEEL_INERTIA,
            "max_torque": constants.WHEEL_MAX_TORQUE,
            "max_speed": constants.WHEEL_MAX_SPEED,
        },
        "magnetorquers": {
            "enabled": False,
            "max_dipole": constants.MTQ_MAX_DIPOLE,
            "k_desat": 5.0e-3,       # [1/s] ~200 s dump time constant before
                                     # dipole clipping stretches it
            "threshold_frac": 0.3,
        },
        "disturbance": {"enabled": False, "constant_torque": [0.0, 0.0, 0.0]},
        "sim": {"dt": constants.SIM_DT_DEFAULT},
    }


# --- helpers --------------------------------------------------------------

def _build_inertia_tensor(values) -> np.ndarray:
    """Accepts either 3 diagonal entries or 9 row-major entries (full tensor)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 3:
        return np.diag(arr)
    if arr.size == 9:
        return arr.reshape(3, 3)
    raise ValueError(
        f"satellite.inertia_tensor must have 3 (diagonal) or 9 (full matrix) "
        f"entries, got {arr.size}"
    )


def _build_diag_gain(values, name: str) -> np.ndarray:
    """Controller gains are specified as 3 diagonal entries in YAML; this
    builds the actual 3x3 matrix PDController/PIDController expect."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"controller.{name} must have exactly 3 entries, got {arr.size}")
    return np.diag(arr)


_WIE_CASES = ("eigenaxis", "robust", "mortensen", "near_eigenaxis")


def _build_ki(values) -> np.ndarray:
    """The integral gain multiplies the accumulator elementwise
    (`self.Ki * self._z`), so unlike Kp/Kd it stays a length-3 vector, not a
    3x3 matrix. Building a matrix here would broadcast wrongly and silently
    produce a different control law."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"controller.Ki must have exactly 3 entries, got {arr.size}")
    return arr.reshape(3)


def _require(d: dict, key: str, section: str):
    if key not in d:
        raise KeyError(f"Missing required field '{key}' in '{section}' section of config")
    return d[key]


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge `override` onto `base`, recursing into nested dicts. Lists/
    scalars in `override` replace `base` outright (no list-concatenation
    surprises)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_raw(raw: dict, name: str) -> SimConfig:
    try:
        sat_raw = _require(raw, "satellite", "root")
        satellite = SatelliteConfig(
            inertia_tensor=_build_inertia_tensor(_require(sat_raw, "inertia_tensor", "satellite"))
        )

        init_raw = _require(raw, "initial", "root")
        initial = InitialConfig(
            attitude_rpy_deg=np.asarray(_require(init_raw, "attitude_rpy_deg", "initial"), dtype=np.float64),
            omega=np.asarray(_require(init_raw, "omega", "initial"), dtype=np.float64),
            wheel_speeds=(np.asarray(init_raw["wheel_speeds"], dtype=np.float64)
                          if init_raw.get("wheel_speeds") is not None else None),
        )

        tgt_raw = _require(raw, "target", "root")
        target = TargetConfig(
            attitude_rpy_deg=np.asarray(_require(tgt_raw, "attitude_rpy_deg", "target"), dtype=np.float64),
        )

        ctrl_raw = _require(raw, "controller", "root")
        ctype = str(_require(ctrl_raw, "type", "controller"))
        u_max = float(ctrl_raw["u_max"]) if "u_max" in ctrl_raw else None
        integral_limit = (float(ctrl_raw["integral_limit"])
                          if "integral_limit" in ctrl_raw else None)

        # Each controller type validates only its own fields, so a PD preset
        # is never asked for `case` and a Wie preset is never asked for Kp.
        if ctype.strip().lower() == "wie":
            # Two mutually exclusive ways to specify (K, D) -- see
            # control/wie.py's build_wie_from_config. Catching the clash here
            # rather than at build time means a config that would have had
            # its zeta silently ignored fails at load, next to the file that
            # caused it.
            has_direct = ("k_scale" in ctrl_raw) or ("d_scale" in ctrl_raw)
            has_rule = ("zeta" in ctrl_raw) or ("settling_time_s" in ctrl_raw)
            if has_direct and has_rule:
                raise ValueError(
                    "controller gives both k_scale/d_scale and "
                    "zeta/settling_time_s. Both set the same (K, D) and would "
                    "disagree silently -- keep one.")

            case = str(ctrl_raw.get("case", "eigenaxis" if has_direct else
                                    _require(ctrl_raw, "case", "controller")))
            if case not in _WIE_CASES:
                raise ValueError(
                    f"controller.case '{case}' is not one of {_WIE_CASES}"
                )

            if has_direct:
                if case != "eigenaxis":
                    raise ValueError(
                        f"k_scale/d_scale mean K = k*J, D = d*J, mu = 1, which "
                        f"is the 'eigenaxis' case; case='{case}' cannot be "
                        f"written this way (its K is not a scalar multiple "
                        f"of J). Use zeta/settling_time_s instead.")
                controller = ControllerConfig(
                    type=ctype,
                    case=case,
                    k_scale=float(_require(ctrl_raw, "k_scale", "controller")),
                    d_scale=float(_require(ctrl_raw, "d_scale", "controller")),
                    Ki=_build_ki(ctrl_raw["Ki"]) if "Ki" in ctrl_raw else None,
                    u_max=u_max,
                    integral_limit=integral_limit,
                    decouple_wheel_momentum=bool(
                        ctrl_raw.get("decouple_wheel_momentum", False)),
                )
            else:
                controller = ControllerConfig(
                    type=ctype,
                    case=case,
                    zeta=float(_require(ctrl_raw, "zeta", "controller")),
                    settling_time_s=float(_require(ctrl_raw, "settling_time_s", "controller")),
                    Ki=_build_ki(ctrl_raw["Ki"]) if "Ki" in ctrl_raw else None,
                    u_max=u_max,
                    integral_limit=integral_limit,
                    decouple_wheel_momentum=bool(
                        ctrl_raw.get("decouple_wheel_momentum", False)),
                )
        elif ctype.strip().lower() == "pid":
            # PID requires Ki -- that's what distinguishes choosing "PID"
            # from choosing "PD" with Ki left unset. A PID preset that
            # forgot Ki would otherwise silently behave like a plain PD run;
            # PIDController.__init__ raises on Ki=None too, but catching it
            # here gives a config-level error at load time instead of at
            # simulate() time.
            controller = ControllerConfig(
                type=ctype,
                Kp=_build_diag_gain(_require(ctrl_raw, "Kp", "controller"), "Kp"),
                Kd=_build_diag_gain(_require(ctrl_raw, "Kd", "controller"), "Kd"),
                Ki=_build_ki(_require(ctrl_raw, "Ki", "controller")),
                u_max=u_max,
                integral_limit=integral_limit,
            )
        else:
            controller = ControllerConfig(
                type=ctype,
                Kp=_build_diag_gain(_require(ctrl_raw, "Kp", "controller"), "Kp"),
                Kd=_build_diag_gain(_require(ctrl_raw, "Kd", "controller"), "Kd"),
                # Opt-in Ki, same field Wie's config uses -- reusing it here
                # rather than adding a second field, since it means the same
                # thing (length-3 integral gain vector) for both types.
                Ki=_build_ki(ctrl_raw["Ki"]) if "Ki" in ctrl_raw else None,
                u_max=u_max,
                integral_limit=integral_limit,
            )

        wh_raw = _require(raw, "wheels", "root")
        wheels = WheelsConfig(
            config=str(_require(wh_raw, "config", "wheels")),
            tilt_deg=float(_require(wh_raw, "tilt_deg", "wheels")),
            wheel_inertia=float(_require(wh_raw, "wheel_inertia", "wheels")),
            max_torque=float(_require(wh_raw, "max_torque", "wheels")),
            max_speed=float(_require(wh_raw, "max_speed", "wheels")),
        )

        # Defaults restated here (as well as in _base_raw) on purpose: a raw
        # dict handed straight to _parse_raw -- tests do this -- still parses
        # without the base layer underneath it.
        mtq_raw = raw.get("magnetorquers", {})
        magnetorquers = MagnetorquersConfig(
            enabled=bool(mtq_raw.get("enabled", False)),
            max_dipole=float(mtq_raw.get("max_dipole", constants.MTQ_MAX_DIPOLE)),
            k_desat=float(mtq_raw.get("k_desat", 5.0e-3)),
            threshold_frac=float(mtq_raw.get("threshold_frac", 0.3)),
        )

        dist_raw = raw.get("disturbance", {"enabled": False, "constant_torque": [0.0, 0.0, 0.0]})
        disturbance = DisturbanceConfig(
            enabled=bool(dist_raw.get("enabled", False)),
            constant_torque=np.asarray(dist_raw.get("constant_torque", [0.0, 0.0, 0.0]), dtype=np.float64),
        )

        sim_raw = _require(raw, "sim", "root")
        sim = SimSettings(
            dt=float(_require(sim_raw, "dt", "sim")),
            duration=float(_require(sim_raw, "duration", "sim")),
        )

    except KeyError as e:
        raise KeyError(f"Invalid config '{name}': {e}") from e

    return SimConfig(
        name=name,
        satellite=satellite,
        initial=initial,
        target=target,
        controller=controller,
        wheels=wheels,
        magnetorquers=magnetorquers,
        disturbance=disturbance,
        sim=sim,
    )


def _read_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --- main entry points -------------------------------------------------

def load_config(path: str | Path) -> SimConfig:
    """A single YAML file, merged over the constants-derived base layer.
    The file only needs what the base doesn't provide (initial/target/
    controller/sim.duration); anything it does restate wins the merge."""
    path = Path(path)
    raw = _deep_merge(_base_raw(), _read_yaml(path))
    return _parse_raw(raw, name=path.stem)


def compose_config(scenario: str | Path, controller: str | Path,
                    name: str | None = None) -> SimConfig:
    """The configs/ layout: the constants-derived base + a scenario
    (initial/target/sim.duration/disturbance/magnetorquers) + a controller
    block, merged base -> scenario -> controller (later keys win)."""
    if name is not None and str(name).endswith((".yaml", ".yml")):
        raise TypeError(
            "compose_config() no longer takes a base YAML -- the base layer "
            "comes from deimos.constants now (configs/_base.yaml is gone). "
            "Call compose_config(scenario, controller) with just two paths.")
    scenario = Path(scenario)
    raw = _deep_merge(_base_raw(), _read_yaml(scenario))
    raw = _deep_merge(raw, _read_yaml(controller))
    return _parse_raw(raw, name=name or scenario.stem)
