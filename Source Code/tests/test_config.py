from pathlib import Path

import numpy as np
import pytest

from deimos.sim.config import compose_config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIOS = CONFIGS / "scenarios"
CONTROLLERS = CONFIGS / "controllers"


def test_compose_pd_baseline_on_slew_55_65_15():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pd_baseline.yaml")
    assert cfg.controller.type == "PD"
    assert cfg.satellite.inertia_tensor.shape == (3, 3)
    assert np.allclose(cfg.initial.attitude_rpy_deg, [55.0, 65.0, 15.0])
    assert cfg.sim.duration == pytest.approx(60.0)
    assert cfg.sim.dt == pytest.approx(0.01)
    assert cfg.wheels.config == "cone"


def test_compose_wie_case3_on_slew_40_30_25():
    cfg = compose_config(SCENARIOS / "slew_40_30_25.yaml", CONTROLLERS / "wie_case3.yaml")
    assert cfg.controller.type == "wie"
    assert cfg.controller.case == "near_eigenaxis"
    assert cfg.controller.settling_time_s == pytest.approx(8.0)


def test_compose_pid_requires_ki_field():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pid_example.yaml")
    assert cfg.controller.type == "PID"
    assert cfg.controller.Ki is not None
    assert cfg.controller.Ki.shape == (3,)


def test_integral_limit_parses_and_reaches_the_controller():
    from deimos.control import registry
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pid_example.yaml")
    assert cfg.controller.integral_limit == pytest.approx(1.0e-3)
    controller = registry.build(cfg.controller, cfg.satellite.inertia_tensor)
    assert controller.integral_limit == pytest.approx(1.0e-3)


def test_integral_limit_defaults_to_none_when_absent():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pd_baseline.yaml")
    assert cfg.controller.integral_limit is None


def test_scenario_overrides_controller_does_not_leak_into_base():
    # compose_config merges base -> scenario -> controller; the controller
    # file must not accidentally carry satellite/wheels keys that would
    # silently override the shared base for one preset only.
    cfg_a = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pd_baseline.yaml")
    cfg_b = compose_config(SCENARIOS / "slew_40_30_25.yaml", CONTROLLERS / "wie_case3.yaml")
    assert np.allclose(cfg_a.satellite.inertia_tensor, cfg_b.satellite.inertia_tensor)
    assert cfg_a.wheels.max_torque == cfg_b.wheels.max_torque


def test_missing_required_field_raises_keyerror(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("satellite:\n  inertia_tensor: [1, 2, 3]\n")
    with pytest.raises(KeyError):
        load_config(bad)
