"""
Free-function view of sim.results.SimResults's per-run metrics. Each of
these is a one-line passthrough to the identically-named SimResults method --
this module exists so code that wants "the metrics" doesn't need to know
they're implemented as dataclass methods, and so a future result type that
isn't a SimResults has somewhere to plug in the same names.
"""

from __future__ import annotations

import numpy as np

from deimos.sim.results import SimResults


def settling_time(results: SimResults, threshold_deg: float = 1.0) -> float | None:
    return results.settling_time(threshold_deg)


def max_wheel_speed(results: SimResults) -> float:
    return results.max_wheel_speed()


def final_attitude_error_deg(results: SimResults) -> float:
    return results.final_attitude_error_deg()


def saturation_fraction(results: SimResults) -> float:
    return results.saturation_fraction()


def per_wheel_saturation_fraction(results: SimResults) -> np.ndarray:
    return results.per_wheel_saturation_fraction()


def speed_saturation_fraction(results: SimResults) -> float:
    return results.speed_saturation_fraction()


def peak_control_torque(results: SimResults) -> float:
    return results.peak_control_torque()


def peak_wheel_torque(results: SimResults) -> float:
    return results.peak_wheel_torque()


def torque_margin(results: SimResults) -> float:
    return results.torque_margin()


def control_effort(results: SimResults) -> float:
    return results.control_effort()


def electrical_energy(results: SimResults) -> float:
    return results.electrical_energy()


def overshoot_deg(results: SimResults) -> float:
    return results.overshoot_deg()


def mean_eigenaxis_deviation_deg(results: SimResults) -> float:
    return results.mean_eigenaxis_deviation_deg()


def summary(results: SimResults) -> str:
    return results.summary()
