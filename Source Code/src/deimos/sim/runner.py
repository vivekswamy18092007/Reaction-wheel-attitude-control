"""
runner.py
=========

`simulate(config)` builds a controller (via control.registry), the wheel
array and the RK4 propagator from a SimConfig, and runs them. This file adds
zero new physics or control logic -- it is purely "read config field -> pass
as constructor argument".

`_build_controller`, `_build_wheels`, `_make_u_func` and
`_rpy_deg_to_quaternion` are used directly by studies/explore.ipynb (e.g. for
the inertia-mismatch experiment in §7.3, which needs the controller and plant
to diverge -- something the public `simulate(config)` entry point can't do
since SimConfig carries only one inertia_tensor).
"""

from __future__ import annotations

import numpy as np

from deimos.math.quaternion import Quaternion, euler_to_quaternion
from deimos.dynamics.propagator import AttitudeSimulator
from deimos.actuators.reaction_wheels import ReactionWheelArray
from deimos.control import registry

from deimos.sim.config import SimConfig
from deimos.sim.results import SimResults


def _rpy_deg_to_quaternion(rpy_deg: np.ndarray) -> Quaternion:
    roll, pitch, yaw = np.radians(rpy_deg)
    return euler_to_quaternion(roll, pitch, yaw).normalize()


def _build_controller(config: SimConfig):
    """Looks up config.controller.type in control.registry and builds the
    matching controller. Add a new type by writing a class + a build_x() in
    control/registry.py -- nothing here needs to change."""
    return registry.build(config.controller, config.satellite.inertia_tensor)


def _make_u_func(config: SimConfig, controller, q_target: Quaternion,
                  wheels=None):
    """Wraps a controller in the `u_func(t, q, omega, Omega) -> body torque`
    signature AttitudeSimulator.run() expects. Every controller in the
    registry shares control.base.Controller's compute_torque/reset shape, so
    there is nothing type-specific left to branch on here -- saturation
    (u_max) is handled inside each controller's compute_torque, and the
    wheel allocation happens downstream in AttitudeSimulator.step()
    regardless of which law produced the commanded torque.

    ONE capability is negotiated rather than assumed: a controller that
    performs gyroscopic decoupling against the TOTAL angular momentum needs
    the wheels' stored momentum h_w, which only this wrapper has access to
    (it is a function of Omega, which the controller never sees). Such a
    controller advertises `needs_wheel_momentum = True` and is handed h_w;
    everything else keeps the original four-argument call untouched. Feature
    detection rather than an isinstance check, so a future controller can opt
    in without this file learning its type -- consistent with how u_max and
    Ki are handled.
    """
    # Fresh instance per run comes from _build_controller; reset() here too
    # so re-running one config object can't carry a frozen sign or a stale
    # integral accumulator across runs.
    controller.reset()

    # dt must be passed only when Ki is set -- and it must match the
    # propagator's dt, or the integral term silently does nothing.
    dt = config.sim.dt if controller.Ki is not None else None
    q_target_arr = q_target.q

    if getattr(controller, "needs_wheel_momentum", False):
        if wheels is None:
            raise ValueError(
                f"{type(controller).__name__} needs the wheel momentum h_w "
                "for its decoupling term, but _make_u_func was called without "
                "a wheel array. Pass wheels= -- studies/explore.ipynb calls "
                "this helper directly and must too.")

        def u_func(t, q_array, omega, Omega):
            return controller.compute_torque(q_array, omega, q_target_arr,
                                             dt=dt, h_w=wheels.momentum(Omega))
        return u_func

    def u_func(t, q_array, omega, Omega):
        return controller.compute_torque(q_array, omega, q_target_arr, dt=dt)

    return u_func


def _build_wheels(config: SimConfig) -> ReactionWheelArray:
    wheels = ReactionWheelArray(
        config=config.wheels.config,
        tilt_deg=config.wheels.tilt_deg,
        wheel_inertia=config.wheels.wheel_inertia,
        max_torque=config.wheels.max_torque,
        max_speed=config.wheels.max_speed,
        Omega0=config.initial.wheel_speeds,
    )
    if (config.initial.wheel_speeds is not None
            and len(config.initial.wheel_speeds) != wheels.N):
        raise ValueError(
            f"initial.wheel_speeds has {len(config.initial.wheel_speeds)} "
            f"entries but the '{config.wheels.config}' geometry has "
            f"{wheels.N} wheels")
    return wheels


def _build_mtq(config: SimConfig):
    """(MagnetorquerArray, B_func) when desaturation is enabled, else
    (None, None). The threshold is specified as a fraction of a single
    wheel's momentum capacity so it means the same thing under any future
    wheel spec -- see configs/scenarios/desat_recovery.yaml."""
    if not config.magnetorquers.enabled:
        return None, None
    from deimos.actuators.magnetorquers import MagnetorquerArray
    from deimos.dynamics.environment import magnetic_field_body

    capacity = config.wheels.wheel_inertia * config.wheels.max_speed
    mtq = MagnetorquerArray(
        max_dipole=config.magnetorquers.max_dipole,
        k_desat=config.magnetorquers.k_desat,
        threshold=config.magnetorquers.threshold_frac * capacity,
    )
    return mtq, magnetic_field_body


def simulate(config: SimConfig) -> SimResults:
    # --- initial / target attitude, from roll/pitch/yaw degrees in config ---
    q0 = _rpy_deg_to_quaternion(config.initial.attitude_rpy_deg)
    q_target = _rpy_deg_to_quaternion(config.target.attitude_rpy_deg)
    omega0 = np.asarray(config.initial.omega, dtype=np.float64)

    # --- build controller + wheels + magnetorquers + simulator ---
    controller = _build_controller(config)
    wheels = _build_wheels(config)
    mtq, B_func = _build_mtq(config)
    sim = AttitudeSimulator(
        inertia_tensor=config.satellite.inertia_tensor,
        q0=q0.q,
        omega0=omega0,
        wheel_array=wheels,
        dt=config.sim.dt,
        mtq=mtq,
        B_func=B_func,
    )

    # --- control law closure ---
    u_func = _make_u_func(config, controller, q_target, wheels=wheels)

    # --- disturbance torque, only nonzero if config.disturbance.enabled ---
    if config.disturbance.enabled:
        tau_d = np.asarray(config.disturbance.constant_torque, dtype=np.float64)
        def tau_ext_func(t, q_array, omega):
            return tau_d
    else:
        tau_ext_func = None  # AttitudeSimulator.run() defaults this to zeros

    # --- run ---
    sim.run(duration=config.sim.duration, u_func=u_func, tau_ext_func=tau_ext_func)

    # --- package into SimResults ---
    return SimResults.from_history(
        name=config.name,
        history=sim.history,
        q_target=q_target.q,
        inertia_tensor=config.satellite.inertia_tensor,
        wheels=wheels,
    )
