"""
power.py
========

Holds a simulated maneuver against the spacecraft's power budget -- the
EPS-facing counterpart of analysis/design_card.py. Nothing here runs a
simulation; it reads a SimResults plus the budget numbers in
deimos/constants.py (the single source for panel areas, generation and the
per-subsystem load allocation) and reports whether the ADCS stayed inside
its slice.

Two deliberate conservatisms, so a PASS here is a real PASS:
  * Wheel electrical energy is integral(sum |g_i * Omega_i|) -- no
    regenerative recovery credited while braking (results.electrical_energy).
  * Magnetorquer power is quadratic in the dipole command (resistive rods:
    P = P_rod * (m/m_max)^2 per axis), integrated over the run.
"""

from __future__ import annotations

import numpy as np

from deimos import constants
from deimos.sim.results import SimResults


def mtq_energy(results: SimResults) -> float:
    """Magnetorquer electrical energy [J]: quadratic (resistive) power model
    per axis, P_i = MTQ_POWER_W * (m_i / MTQ_MAX_DIPOLE)^2, integrated."""
    if results.m is None or not np.any(results.m):
        return 0.0
    frac2 = (results.m / constants.MTQ_MAX_DIPOLE) ** 2
    power = constants.MTQ_POWER_W * frac2.sum(axis=1)
    return float(np.trapezoid(power, results.t))


def power_card(results: SimResults) -> str:
    """Text power report for one run, in the design-card style."""
    duration = float(results.t[-1] - results.t[0])
    wheel_J = results.electrical_energy()
    mtq_J = mtq_energy(results)
    total_J = wheel_J + mtq_J

    avg_W = total_J / duration if duration > 0 else 0.0
    peak_W = float(np.abs(results.wheel_power).sum(axis=1).max())
    adcs_alloc = constants.POWER_BUDGET_W["ADCS"]

    # Battery depth of discharge if this maneuver ran entirely on battery
    # (eclipse worst case) -- 1 Wh = 3600 J.
    dod = total_J / (constants.BATTERY_CAPACITY_WH * 3600.0)

    verdict_avg = "OK" if avg_W <= adcs_alloc else "EXCEEDS ALLOCATION"
    verdict_peak = ("OK" if peak_W <= adcs_alloc else
                    "above allocation -- acceptable transiently, flag if sustained")

    L = []
    add = L.append
    add("=" * 74)
    add(f"POWER CARD - {results.name}")
    add("=" * 74)

    add("\nGENERATION (constants.py, team EPS sizing)")
    add(f"  solar cell area       : {constants.SOLAR_AREA_TOTAL_M2 * 1e4:.0f} cm^2 "
        f"({constants.SOLAR_AREA_WINGS_M2 * 1e4:.0f} wings + "
        f"{constants.SOLAR_AREA_BODY_M2 * 1e4:.0f} body), "
        f"GaAs @ {constants.SOLAR_CELL_EFFICIENCY:.0%}")
    add(f"  peak solar power      : {constants.SOLAR_PEAK_POWER_W:.1f} W")
    add(f"  orbit-average power   : {constants.SOLAR_AVG_POWER_W:.1f} W "
        f"(sunlit {constants.SUNLIT_FRACTION:.0%}, "
        f"illumination factor {constants.ILLUMINATION_FACTOR:.0%})")
    add(f"  total load w/ margin  : {constants.POWER_LOAD_WITH_MARGIN_W:.1f} W "
        f"(raw {constants.POWER_LOAD_TOTAL_W:.1f} W + "
        f"{constants.POWER_MARGIN_FRACTION:.0%})")

    add("\nTHIS MANEUVER")
    add(f"  duration              : {duration:.1f} s")
    add(f"  wheel electrical      : {wheel_J:.3e} J (no-regen upper bound)")
    if mtq_J > 0:
        add(f"  magnetorquer          : {mtq_J:.3e} J (resistive model)")
    add(f"  average ADCS power    : {avg_W:.3f} W vs {adcs_alloc:.1f} W "
        f"allocation -- {verdict_avg}")
    add(f"  peak ADCS power       : {peak_W:.3f} W -- {verdict_peak}")
    add(f"  battery DoD (eclipse) : {100 * dod:.3f}% of "
        f"{constants.BATTERY_CAPACITY_WH:.0f} Wh (ASSUMPTION -- see constants.py)")

    return "\n".join(L)
