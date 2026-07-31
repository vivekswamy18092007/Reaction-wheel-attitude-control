"""
controller.type (a string in YAML) -> factory that builds a control.base.
Controller from a sim.config.ControllerConfig + the plant's inertia tensor.

This is the single place sim/runner.py asks "what controller do I build for
this config"; adding a new type (LQR going from stub to real, or a fifth
control law) means writing the class + a build_x() here, not touching
runner.py's dispatch.
"""

from __future__ import annotations

from deimos.control.pd import PDController
from deimos.control.pid import PIDController
from deimos.control.wie import WieRegulator, build_wie_from_config
from deimos.control.lqr import LQRController


def _build_pd(config, J):
    return PDController(Kp=config.Kp, Kd=config.Kd, Ki=config.Ki,
                         u_max=config.u_max, integral_limit=config.integral_limit)


def _build_pid(config, J):
    return PIDController(Kp=config.Kp, Kd=config.Kd, Ki=config.Ki,
                          u_max=config.u_max, integral_limit=config.integral_limit)


def _build_wie(config, J):
    # K, D and mu are derived from the *same* inertia tensor the plant runs
    # on, never a hardcoded one -- gains sized for the wrong spacecraft
    # saturate the wheels forever.
    wie_raw = {"case": config.case}
    # Exactly one of the two forms is populated -- sim/config.py rejects a
    # config carrying both, so this cannot pass both through and let
    # build_wie_from_config pick silently.
    if config.k_scale is not None or config.d_scale is not None:
        wie_raw["k_scale"] = config.k_scale
        wie_raw["d_scale"] = config.d_scale
    else:
        wie_raw["zeta"] = config.zeta
        wie_raw["settling_time_s"] = config.settling_time_s
    if config.Ki is not None:
        wie_raw["Ki"] = config.Ki
    if config.u_max is not None:
        wie_raw["u_max"] = config.u_max
    if config.integral_limit is not None:
        wie_raw["integral_limit"] = config.integral_limit
    wie_raw["decouple_wheel_momentum"] = config.decouple_wheel_momentum
    return build_wie_from_config(wie_raw, J)


def _build_lqr(config, J):
    return LQRController()


REGISTRY = {
    "pd": _build_pd,
    "pid": _build_pid,
    "wie": _build_wie,
    "lqr": _build_lqr,
}


def build(config, J):
    """config: a sim.config.ControllerConfig. J: the plant's inertia tensor."""
    ctype = config.type.strip().lower()
    if ctype not in REGISTRY:
        raise ValueError(
            f"Unknown controller.type '{config.type}' "
            f"(expected one of: {', '.join(sorted(REGISTRY))})"
        )
    return REGISTRY[ctype](config, J)
