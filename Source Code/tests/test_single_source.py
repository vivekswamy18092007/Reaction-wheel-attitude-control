"""
The single-source-of-truth contract: every composed config carries
constants.py's numbers unless a YAML explicitly overrides them, and the old
three-file compose form fails loudly instead of silently misbinding.
"""

from pathlib import Path

import numpy as np
import pytest

from deimos import constants
from deimos.sim.config import compose_config
from deimos.analysis.power import power_card
from deimos.sim.runner import simulate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIOS = CONFIGS / "scenarios"
CONTROLLERS = CONFIGS / "controllers"


def test_base_yaml_is_gone():
    """The file whose hand-copied drift motivated the restructure must not
    quietly come back."""
    assert not (CONFIGS / "_base.yaml").exists()


def test_composed_config_carries_the_constants_tensor():
    cfg = compose_config(SCENARIOS / "slew_40_30_25.yaml",
                         CONTROLLERS / "wie_eigenaxis.yaml")
    assert np.allclose(cfg.satellite.inertia_tensor, constants.INERTIA_TENSOR)
    assert cfg.wheels.max_torque == constants.WHEEL_MAX_TORQUE
    assert cfg.wheels.max_speed == constants.WHEEL_MAX_SPEED
    assert cfg.sim.dt == constants.SIM_DT_DEFAULT


def test_scenario_can_still_override_the_base():
    """The inertia-mismatch study depends on this: an explicit satellite
    block in a scenario wins the merge."""
    import yaml, tempfile, os
    override = {"satellite": {"inertia_tensor": [1e-2, 1e-2, 1e-2]},
                "initial": {"attitude_rpy_deg": [10, 0, 0], "omega": [0, 0, 0]},
                "target": {"attitude_rpy_deg": [0, 0, 0]},
                "sim": {"duration": 5.0}}
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(override, f)
        cfg = compose_config(path, CONTROLLERS / "pd_baseline.yaml")
        assert np.allclose(cfg.satellite.inertia_tensor, np.diag([1e-2, 1e-2, 1e-2]))
    finally:
        os.unlink(path)


def test_old_three_path_call_fails_loudly():
    with pytest.raises(TypeError, match="no longer takes a base YAML"):
        compose_config(SCENARIOS / "slew_40_30_25.yaml",
                       CONTROLLERS / "wie_eigenaxis.yaml",
                       CONTROLLERS / "pd_baseline.yaml")


def test_magnetorquer_defaults_off_with_constants_ceiling():
    cfg = compose_config(SCENARIOS / "slew_40_30_25.yaml",
                         CONTROLLERS / "wie_eigenaxis.yaml")
    assert cfg.magnetorquers.enabled is False
    assert cfg.magnetorquers.max_dipole == constants.MTQ_MAX_DIPOLE


def test_desat_scenario_parses_wheel_speeds():
    cfg = compose_config(SCENARIOS / "desat_recovery.yaml",
                         CONTROLLERS / "wie_eigenaxis.yaml")
    assert cfg.magnetorquers.enabled is True
    assert cfg.initial.wheel_speeds is not None
    assert cfg.initial.wheel_speeds.shape == (3,)
    assert cfg.sim.dt == pytest.approx(0.1)


def test_power_card_reads_constants_budget():
    cfg = compose_config(SCENARIOS / "slew_15_15_15_with_rate.yaml",
                         CONTROLLERS / "pd_aggressive.yaml")
    cfg.sim.duration = 5.0
    card = power_card(simulate(cfg))
    assert "POWER CARD" in card
    assert f"{constants.POWER_BUDGET_W['ADCS']:.1f} W" in card
