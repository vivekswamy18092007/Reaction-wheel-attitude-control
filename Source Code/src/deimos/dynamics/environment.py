"""
Orbital environment models -- currently the geomagnetic field, which is what
the magnetorquer desaturation law pushes against.

Model: NON-TILTED DIPOLE along a circular orbit (Wertz, Spacecraft Attitude
Determination and Control, Appendix H; also Psiaki, "Magnetic Torquer
Attitude Control via Asymptotic Periodic Linear Quadratic Regulation", 2001,
eq. 1). For a circular orbit of inclination i and argument of latitude
u = n*t (measured from the ascending node), the field in the ORBIT frame is

    B_orbit(t) = B0 * (R_E / r)^3 * [ cos(u) sin(i),
                                      -cos(i),
                                      2 sin(u) sin(i) ]

with B0 the mean equatorial surface field (constants.B0_EQUATOR_T). The
field rotates at twice orbit rate in the orbit plane -- which is exactly why
cross-product desaturation works: any wheel-momentum component parallel to B
at one moment is perpendicular to it a quarter-orbit later.

FRAME REGISTRATION ASSUMPTION (document in the report): the simulation
propagates attitude relative to an arbitrary inertial frame and does not
model translation, so we identify that inertial frame with the orbit frame
at t = 0 and evaluate B_orbit directly in it. For desaturation this is
inconsequential -- the law only needs a realistically rotating field of
realistic magnitude, not centimetre-level orbit geometry. Replace with an
IGRF model + real orbit propagation if the project ever needs field-accurate
ground-track work.
"""

from __future__ import annotations

import numpy as np

from deimos import constants
from deimos.math.quaternion import Quaternion
from deimos.dynamics.disturbances import orbital_rate_squared


def magnetic_field_inertial(t: float,
                            altitude_m: float = constants.ORBIT_ALTITUDE_M,
                            inclination_deg: float = constants.ORBIT_INCLINATION_DEG,
                            ) -> np.ndarray:
    """Geomagnetic field [T] at time t along the orbit, in the inertial
    (= t=0 orbit) frame. Magnitude at 500 km is ~25-45 uT depending on
    orbit position, matching the dipole model's known range."""
    n = np.sqrt(orbital_rate_squared(altitude_m))
    u = n * t
    i = np.radians(inclination_deg)
    r = constants.EARTH_RADIUS_M + altitude_m
    B_mag = constants.B0_EQUATOR_T * (constants.EARTH_RADIUS_M / r) ** 3
    return B_mag * np.array([
        np.cos(u) * np.sin(i),
        -np.cos(i),
        2.0 * np.sin(u) * np.sin(i),
    ])


def magnetic_field_body(t: float, q: np.ndarray) -> np.ndarray:
    """Geomagnetic field [T] in the BODY frame.

    q: scalar-first attitude quaternion (body -> inertial, the convention
    used throughout deimos). to_rotation_matrix() maps body to inertial,
    so its transpose brings the inertial-frame field into the body frame.
    """
    R = Quaternion(*q).to_rotation_matrix()
    return R.T @ magnetic_field_inertial(t)
