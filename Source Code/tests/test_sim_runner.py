from pathlib import Path

import numpy as np
import pytest

from deimos.sim.config import compose_config
from deimos.sim.runner import simulate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIOS = CONFIGS / "scenarios"
CONTROLLERS = CONFIGS / "controllers"


def _short(cfg, duration=5.0):
    cfg.sim.duration = duration
    return cfg


def test_pd_run_converges_attitude_error_downward():
    cfg = compose_config(SCENARIOS / "slew_15_15_15_with_rate.yaml", CONTROLLERS / "pd_baseline.yaml")
    cfg = _short(cfg, duration=20.0)
    results = simulate(cfg)
    assert results.attitude_error_deg[-1] < results.attitude_error_deg[0]


def test_wie_run_converges_attitude_error_downward():
    cfg = compose_config(SCENARIOS / "slew_40_30_25.yaml", CONTROLLERS / "wie_case3.yaml")
    cfg = _short(cfg, duration=15.0)
    results = simulate(cfg)
    assert results.attitude_error_deg[-1] < results.attitude_error_deg[0]


def test_momentum_conservation_with_no_control_and_no_disturbance():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pd_baseline.yaml")
    cfg = _short(cfg, duration=2.0)
    cfg.controller.Kp = np.zeros((3, 3))
    cfg.controller.Kd = np.zeros((3, 3))
    results = simulate(cfg)
    drift = results.total_momentum_norm.max() - results.total_momentum_norm[0]
    assert abs(drift) < 1e-9


def test_quaternion_stays_normalized():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pd_baseline.yaml")
    cfg = _short(cfg, duration=5.0)
    results = simulate(cfg)
    assert np.abs(results.quaternion_norm_error).max() < 1e-8


def test_results_array_shapes_are_consistent():
    cfg = compose_config(SCENARIOS / "slew_15_15_15_with_rate.yaml", CONTROLLERS / "pd_baseline.yaml")
    cfg = _short(cfg, duration=1.0)
    results = simulate(cfg)
    T = len(results.t)
    assert results.q.shape == (T, 4)
    assert results.omega.shape == (T, 3)
    assert results.Omega.shape[0] == T
    assert results.u.shape == (T, 3)
    assert results.g.shape == results.Omega.shape


def test_saturation_fraction_is_between_zero_and_one():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml", CONTROLLERS / "pd_aggressive.yaml")
    cfg = _short(cfg, duration=5.0)
    results = simulate(cfg)
    assert 0.0 <= results.saturation_fraction() <= 1.0
