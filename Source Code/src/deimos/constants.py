"""


Every number that describes the spacecraft, its hardware, its orbit or its
power system. Anything we are not tuning lives in this module and
nowhere else. Config composition (sim/config.py) builds its base layer from
here, so every simulation, notebook, plot and report traces back to these
values by construction. 

  * Tunable quantities (controller gains, target attitudes, scenario
    durations) do not belong here, They live in configs/scenarios/ and
    configs/controllers/, to make them tunable.
"""

import numpy as np



INERTIA_TENSOR = np.array([
    [3.6050e-2, -1.3068876e-5, -2.0749156e-5],
    [-1.3068876e-5, 3.4500e-2, 5.8844152e-5],
    [-2.0749156e-5, 5.8844152e-5, 1.0160e-2],
])


SPACECRAFT_MASS_KG = 3.9       
                                #
SPACECRAFT_DIMS_M = (0.10, 0.10, 0.34)  
                                        


# REACTION WHEELS - Maxon ECX Flat 22L, 3-wheel orthogonal cone
# Current design: 3-wheel orthogonal cone (see actuators/reaction_wheels.py).
# Numbers are the CAD/datasheet values, not placeholders.
WHEEL_INERTIA = 5.8e-6      # [kg m^2] per wheel, CAD-exported (Al 7075 hub + rim)
WHEEL_MAX_TORQUE = 4.0e-3   # [N m] 
WHEEL_MAX_SPEED = 1256.6    # [rad/s] 12000 rpm, ECX Flat 22L datasheet
WHEEL_TILT_DEG = 32.0       # only used by the legacy "pyramid" config; "cone"
                            # fixes its cant angle internally and ignores this
WHEEL_CONFIG = "cone"

# Momentum a single wheel can store before it pins at max speed [N m s].
# The magnetorquer desaturation threshold is expressed as a fraction of this.
WHEEL_MOMENTUM_CAPACITY = WHEEL_INERTIA * WHEEL_MAX_SPEED   # ~7.29e-3

# MAGNETORQUERS- 3 orthogonal rods, desaturation (momentum dumping) only

MTQ_MAX_DIPOLE = 0.2        # [A m^2] per-axis dipole moment limit
MTQ_POWER_W = 0.2           # [W] per rod at full drive for the power budget


# ORBIT - circular LEO for disturbance + geomagnetic-field models

MU_EARTH = 3.986004418e14   # [m^3/s^2]
EARTH_RADIUS_M = 6378137.0  # [m]
ORBIT_ALTITUDE_M = 500e3    # [m] representative CubeSat LEO altitude
ORBIT_INCLINATION_DEG = 97.4  # ASSUMPTION -- sun-synchronous at 500 km, the
                              # default rideshare orbit; consistent with the
                              # POWER section's 62% sunlit fraction
B0_EQUATOR_T = 3.12e-5      # [T] mean equatorial geomagnetic field at the
                            # surface 

# POWER - solar generation and load budget 

# Solar cells: triple-junction GaAs, body-mounted panels alone are
# insufficient so 2 deployable wings.
#   wings: 2 wings x 7 cells, 80 x 40 mm  ---> 14 cells, 448 cm^2
#  body : 4 side faces x 12 cells, 30 x 40 mm ---> 48 cells, 576 cm^2

SOLAR_AREA_WINGS_M2 = 14 * (0.080 * 0.040)   # 0.0448
SOLAR_AREA_BODY_M2 = 48 * (0.030 * 0.040)    # 0.0576
SOLAR_AREA_TOTAL_M2 = SOLAR_AREA_WINGS_M2 + SOLAR_AREA_BODY_M2  # 0.1024

SOLAR_CELL_EFFICIENCY = 0.28    
SUNLIT_FRACTION = 0.62          # 38% eclipse per orbit
ILLUMINATION_FACTOR = 0.35      # effective incidence-angle + partial-face
                                # illumination factor (team estimate)
SOLAR_CONSTANT_W_M2 = 1361.0    # [W/m^2] AM0

# Peak: 1361 * 0.1024 * 0.28 = 39.0 W. Orbit-average: 39 * 0.62 * 0.35 = 8.5 W.
SOLAR_PEAK_POWER_W = SOLAR_CONSTANT_W_M2 * SOLAR_AREA_TOTAL_M2 * SOLAR_CELL_EFFICIENCY
SOLAR_AVG_POWER_W = SOLAR_PEAK_POWER_W * SUNLIT_FRACTION * ILLUMINATION_FACTOR

# Load budget [W] -
POWER_BUDGET_W = {
    "OBC": 0.5,
    "ADCS": 2.0,
    "payload_imager": 0.7,
    "communications": 0.5,
    "EPS_losses": 0.6,
}
POWER_LOAD_TOTAL_W = sum(POWER_BUDGET_W.values())          # 4.3
POWER_MARGIN_FRACTION = 0.30
POWER_LOAD_WITH_MARGIN_W = POWER_LOAD_TOTAL_W * (1 + POWER_MARGIN_FRACTION)  # 5.6

BATTERY_CAPACITY_WH = 40.0   # ASSUMPTION 
BATTERY_BUS_VOLTAGE = 7.4    # ASSUMPTION 

# SIMULATION DEFAULTS
SIM_DT_DEFAULT = 0.01        # [s] RK4 step
