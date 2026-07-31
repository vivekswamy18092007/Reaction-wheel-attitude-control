"""Smoke tests for the GA figures -- these assert the plots build without
error and carry the right structure, not that they look right."""

import matplotlib
matplotlib.use("Agg")   # headless: no display in CI / test runs

import numpy as np
import pytest

from deimos.tuning.objectives import make_decode, pareto_picks
from deimos.viz import tuning as vt


@pytest.fixture
def fake_front():
    rng = np.random.default_rng(0)
    pop = rng.random((12, 9))
    obj = rng.random((12, 3))
    history = [
        {"generation": g, "front_size": 3 + g,
         "best_per_objective": np.array([10.0 - g, 5.0 - 0.5 * g, 1.0]),
         "front_objectives": rng.random((3, 3))}
        for g in range(4)
    ]
    return pop, obj, make_decode("PID"), history


def test_pareto_projections_has_three_panels(fake_front):
    _, obj, _, _ = fake_front
    fig = vt.plot_pareto_projections(obj, "PID")
    assert len(fig.axes) == 3


def test_pareto_3d_builds(fake_front):
    _, obj, _, _ = fake_front
    fig = vt.plot_pareto_3d(obj, "PID")
    assert fig.axes[0].name == "3d"


def test_convergence_has_one_panel_per_objective_plus_front_size(fake_front):
    _, _, _, history = fake_front
    fig = vt.plot_convergence(history, "PID")
    assert len(fig.axes) == 4   # 3 objectives + front size


def test_parallel_coordinates_has_one_tick_per_gene(fake_front):
    pop, obj, decode, _ = fake_front
    fig = vt.plot_gain_parallel_coordinates(pop, decode, "PID", obj=obj)
    assert len(fig.axes[0].get_xticks()) == 9


def test_parallel_coordinates_normalizes_into_unit_range(fake_front):
    pop, obj, decode, _ = fake_front
    fig = vt.plot_gain_parallel_coordinates(pop, decode, "PID", obj=obj)
    # Genes are uniform in [0,1] and decoding is log-uniform within the
    # bounds, so every normalized coordinate must land back in [0,1].
    ydata = np.concatenate([ln.get_ydata() for ln in fig.axes[0].lines
                            if len(ln.get_ydata()) == 9])
    assert ydata.min() >= -1e-9 and ydata.max() <= 1 + 1e-9


def test_plot_tuning_saves_every_figure(tmp_path, fake_front):
    pop, obj, decode, history = fake_front
    figs = vt.plot_tuning(pop, obj, decode, history, controller_type="PID",
                           save_dir=tmp_path, show=False)
    assert set(figs) == set(vt.REGISTRY)
    written = {p.name for p in tmp_path.glob("*.png")}
    assert written == {f"ga_{name}.png" for name in vt.REGISTRY}


def test_pd_front_uses_pd_objective_labels(fake_front):
    _, obj, _, _ = fake_front
    fig = vt.plot_pareto_projections(obj, "PD")
    assert "saturation fraction" in fig.axes[1].get_ylabel()


def test_picks_are_highlighted_on_projections(fake_front):
    _, obj, _, _ = fake_front
    picks = pareto_picks(obj, "PID")
    fig = vt.plot_pareto_projections(obj, "PID", picks=picks)
    # one cloud scatter + one per pick
    assert len(fig.axes[0].collections) == 1 + len(picks)
