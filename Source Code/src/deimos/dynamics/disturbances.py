"""
Environmental disturbance torque models.

Only the gravity-gradient torque is modelled so far -- it is the dominant
secular disturbance for a LEO CubeSat and, unlike drag or SRP, it depends
only on the inertia tensor and the orbit radius, both of which the project
already fixes in constants.py.
"""

from __future__ import annotations

import numpy as np

from deimos import constants


def orbital_rate_squared(altitude_m: float = constants.ORBIT_ALTITUDE_M) -> float:
    """n^2 = mu / R^3 for a circular orbit [rad^2/s^2]."""
    R = constants.EARTH_RADIUS_M + altitude_m
    return constants.MU_EARTH / R**3


def gravity_gradient_torque(J, altitude_m: float = constants.ORBIT_ALTITUDE_M) -> float:
    """
    Standard worst-case gravity-gradient torque magnitude [N m]:

        tau_gg = (3/2) * n^2 * |J_max - J_min|

    This is the textbook bound on the secular torque an inertia asymmetry
    picks up at orbital rate, not a time-resolved model -- it is the right
    magnitude to hold constant in the body frame when you want a steady bias
    for the integral term to reject, which is exactly what tuning a PID
    needs. Returns a scalar; apply it per-axis via np.full(3, tau_gg) to
    match `disturbance.constant_torque`'s shape.
    """
    Jd = np.diag(np.asarray(J, dtype=float))
    return 1.5 * orbital_rate_squared(altitude_m) * (Jd.max() - Jd.min())
