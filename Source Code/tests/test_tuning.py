from pathlib import Path

import numpy as np
import pytest

from deimos.dynamics.disturbances import gravity_gradient_torque, orbital_rate_squared
from deimos.sim.config import compose_config
from deimos.tuning import objectives as O
from deimos.tuning.nsga2 import NSGA2

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIOS = CONFIGS / "scenarios"
CONTROLLERS = CONFIGS / "controllers"

J = np.diag([1.4010e-2, 1.4130e-2, 2.9380e-3])


def _base_config(disturbed=True):
    cfg = compose_config(SCENARIOS / "slew_15_15_15_with_rate.yaml",
                          CONTROLLERS / "pd_baseline.yaml")
    if disturbed:
        cfg.disturbance.enabled = True
        cfg.disturbance.constant_torque = np.full(3, gravity_gradient_torque(J))
    return cfg


# --- disturbance model ---------------------------------------------------

def test_gravity_gradient_is_positive_and_small():
    tau = gravity_gradient_torque(J)
    # ~1e-8 N m for this J at 500 km: real, but four orders below the 4 mN m
    # wheel limit -- a bias to reject, not a torque that threatens authority.
    assert 0 < tau < 1e-6


def test_gravity_gradient_zero_for_spherical_inertia():
    assert gravity_gradient_torque(np.eye(3) * 0.01) == pytest.approx(0.0)


def test_gravity_gradient_scales_with_asymmetry():
    small = gravity_gradient_torque(np.diag([0.01, 0.011, 0.012]))
    large = gravity_gradient_torque(np.diag([0.01, 0.011, 0.10]))
    assert large > small


def test_orbital_rate_matches_hand_calculation():
    from deimos import constants
    R = constants.EARTH_RADIUS_M + constants.ORBIT_ALTITUDE_M
    assert orbital_rate_squared() == pytest.approx(constants.MU_EARTH / R**3)


# --- search-space plumbing ------------------------------------------------

def test_gene_counts_per_controller_type():
    assert O.n_genes("PD") == 6
    assert O.n_genes("PID") == 9
    assert len(O.gene_labels("PID")) == 9
    assert len(O.gene_bounds("PID")) == 9


def test_gene_labels_last_block_is_the_ki_ratio():
    # Ki is searched relative to Kp, so the gene is a ratio -- the label has
    # to say so or the parallel-coordinates axis is silently mislabelled.
    assert O.gene_labels("PID") == ["Kp_x", "Kp_y", "Kp_z", "Kd_x", "Kd_y",
                                     "Kd_z", "Ki/Kp_x", "Ki/Kp_y", "Ki/Kp_z"]


def test_unknown_controller_type_raises():
    with pytest.raises(ValueError):
        O.n_genes("LQR")


def test_decode_respects_bounds_at_gene_extremes():
    decode = O.make_decode("PID")
    kp, kd, ki = decode(np.zeros(9))
    assert np.allclose(kp, O.KP_BOUNDS[0])
    assert np.allclose(ki, O.KI_RATIO_BOUNDS[0] * O.KP_BOUNDS[0])
    kp, kd, ki = decode(np.ones(9))
    assert np.allclose(kp, O.KP_BOUNDS[1])
    assert np.allclose(ki, O.KI_RATIO_BOUNDS[1] * O.KP_BOUNDS[1])


def test_decoded_ki_is_always_small_relative_to_kp():
    # The whole point of the ratio parameterization: no reachable gene
    # combination can produce a Ki that rivals its own Kp.
    decode = O.make_decode("PID")
    rng = np.random.default_rng(0)
    for genes in rng.random((50, 9)):
        kp, _, ki = decode(genes)
        assert np.all(ki / kp <= O.KI_RATIO_BOUNDS[1] + 1e-12)


def test_decode_midpoint_is_log_centered():
    decode = O.make_decode("PD")
    kp, _ = decode(np.full(6, 0.5))
    expected = 10 ** (0.5 * (np.log10(O.KP_BOUNDS[0]) + np.log10(O.KP_BOUNDS[1])))
    assert np.allclose(kp, expected)


def test_decoded_to_flat_round_trips_back_to_ratio_space():
    # flat form must land back in gene/bounds space so the parallel-coords
    # normalization is valid; for PID that means Ki -> Ki/Kp.
    decode = O.make_decode("PID")
    genes = np.full(9, 0.3)
    flat = O.decoded_to_flat(decode(genes), "PID")
    assert flat.shape == (9,)
    expected_ratio = O._log_uniform(0.3, *O.KI_RATIO_BOUNDS)
    assert np.allclose(flat[6:], expected_ratio)


# --- applying gains to a config -------------------------------------------

def test_apply_gains_pid_sets_vector_ki():
    cfg = _base_config()
    decode = O.make_decode("PID")
    O.apply_gains(cfg, decode(np.full(9, 0.5)), "PID")
    assert cfg.controller.type == "PID"
    assert cfg.controller.Kp.shape == (3, 3)
    # Ki multiplies the accumulator elementwise, so it must stay a vector.
    assert cfg.controller.Ki.shape == (3,)


def test_apply_gains_pd_clears_stale_ki():
    # Seeding from a PID preset must not leave integral action silently on
    # while the run is reported as a PD.
    cfg = _base_config()
    cfg.controller.Ki = np.array([1e-4, 1e-4, 1e-4])
    O.apply_gains(cfg, O.make_decode("PD")(np.full(6, 0.5)), "PD")
    assert cfg.controller.type == "PD"
    assert cfg.controller.Ki is None


# --- evaluator -------------------------------------------------------------

def test_pd_evaluator_returns_three_objectives():
    ev = O.make_evaluator(_base_config(), controller_type="PD", duration=3.0)
    out = ev(O.make_decode("PD")(np.full(6, 0.5)))
    assert len(out) == 3
    assert all(np.isfinite(v) for v in out)


def test_pid_evaluator_returns_three_objectives():
    ev = O.make_evaluator(_base_config(), controller_type="PID", duration=3.0)
    out = ev(O.make_decode("PID")(np.full(9, 0.5)))
    assert len(out) == 3
    assert all(np.isfinite(v) for v in out)


def test_pid_evaluator_rejects_saturating_candidate():
    # A negative limit makes any saturation infeasible, which must produce
    # the worst-case tuple rather than a competitive score.
    ev = O.make_evaluator(_base_config(), controller_type="PID", duration=3.0,
                           saturation_limit=-1.0)
    out = ev(O.make_decode("PID")(np.full(9, 0.9)))
    assert out == O.worst_case("PID", 3.0)


def test_worst_case_itae_is_the_true_upper_bound():
    # 180 deg held for the whole window: integral of t*180 dt = 90*T^2. This
    # has to be a genuine bound, not a magic number, or the hypervolume
    # reference is not a reference.
    assert O.worst_case("PD", 10.0)[0] == pytest.approx(90.0 * 100.0)
    assert O.worst_case("PID", 60.0)[0] == pytest.approx(90.0 * 3600.0)


def test_itae_matches_a_hand_computed_integral():
    t = np.linspace(0.0, 2.0, 2001)
    e = np.full_like(t, 3.0)                     # constant 3 deg error
    assert O.itae(e, t) == pytest.approx(3.0 * 0.5 * 2.0**2, rel=1e-6)


def test_itae_penalizes_late_error_more_than_early_error():
    # The whole reason for the t weight: the same total error area scores
    # worse when it happens late, which is what makes ITAE a *settling*
    # objective rather than just an error objective.
    t = np.linspace(0.0, 10.0, 1001)
    early = np.where(t < 2.0, 5.0, 0.0)
    late = np.where(t > 8.0, 5.0, 0.0)
    assert O.itae(late, t) > O.itae(early, t)


def test_itae_objective_discriminates_non_settling_candidates():
    # The regression that motivated replacing thresholded settling time:
    # when no candidate settles, a capped settle time is CONSTANT and the
    # first objective carries no information at all. ITAE must still order
    # them, for BOTH controller types.
    for ctype, n in (("PD", 6), ("PID", 9)):
        ev = O.make_evaluator(_base_config(), controller_type=ctype, duration=6.0)
        decode = O.make_decode(ctype)
        weak = ev(decode(np.zeros(n)))                    # tiny Kp/Kd
        strong = ev(decode(np.full(n, 0.55)))
        assert weak[0] != strong[0]
        # A near-zero-gain controller barely moves, so it must score worse.
        assert weak[0] > strong[0]


def test_evaluator_worst_case_aggregates_across_scenarios():
    # Multi-scenario evaluation must return the elementwise MAX, so a gain
    # set cannot buy a good score by being excellent on the easy maneuver.
    easy = compose_config(SCENARIOS / "slew_15_15_15_with_rate.yaml",
                           CONTROLLERS / "pd_baseline.yaml")
    hard = compose_config(SCENARIOS / "slew_55_65_15.yaml",
                           CONTROLLERS / "pd_baseline.yaml")
    decoded = O.make_decode("PD")(np.full(6, 0.5))

    ev = O.Evaluator([easy, hard], controller_type="PD", duration=5.0)
    combined = ev(decoded)
    singles = ev.per_scenario(decoded)

    assert len(singles) == 2
    assert combined == pytest.approx(np.max(np.array(singles), axis=0))
    assert all(combined[i] >= s[i] for s in singles for i in range(3))


def test_normalized_gene_position_is_zero_and_one_at_the_bounds():
    decode = O.make_decode("PID")
    assert np.allclose(O.normalized_gene_position(decode(np.zeros(9)), "PID"), 0.0)
    assert np.allclose(O.normalized_gene_position(decode(np.ones(9)), "PID"), 1.0)


def test_decoder_is_picklable():
    # Parallel evaluation ships the evaluator (and its decode) to worker
    # processes; on Windows those are spawned, not forked, so anything that
    # crosses must pickle. A closure would not.
    import pickle
    decode = O.make_decode("PID")
    restored = pickle.loads(pickle.dumps(decode))
    assert np.allclose(np.concatenate(restored(np.full(9, 0.3))),
                       np.concatenate(decode(np.full(9, 0.3))))


def test_evaluator_is_picklable():
    import pickle
    ev = O.Evaluator(_base_config(), controller_type="PID", duration=2.0)
    restored = pickle.loads(pickle.dumps(ev))
    assert restored.worst == ev.worst
    assert len(restored.base_configs) == 1


# --- Pareto utilities ------------------------------------------------------

def test_pareto_picks_indices_are_in_range_and_optimal():
    obj = np.array([[1.0, 9.0, 5.0],
                    [9.0, 1.0, 5.0],
                    [5.0, 5.0, 1.0],
                    [4.0, 4.0, 4.0]])
    picks = O.pareto_picks(obj, "PID")
    assert set(picks) == {"best_tracking", "most_accurate", "cheapest_effort", "knee"}
    assert all(0 <= i < len(obj) for i in picks.values())
    assert picks["best_tracking"] == 0
    assert picks["most_accurate"] == 1
    assert picks["cheapest_effort"] == 2


def test_pareto_picks_labels_differ_by_controller_type():
    obj = np.random.default_rng(0).random((6, 3))
    assert "least_saturated" in O.pareto_picks(obj, "PD")
    assert "most_accurate" in O.pareto_picks(obj, "PID")


def test_to_runs_dict_carries_ki_for_pid_only():
    rng = np.random.default_rng(0)
    pop_pid, obj = rng.random((5, 9)), rng.random((5, 3))
    picks = O.to_runs_dict(pop_pid, obj, O.make_decode("PID"), controller_type="PID")
    assert all(p["Ki"] is not None and p["Ki"].shape == (3,) for p in picks.values())

    pop_pd = rng.random((5, 6))
    picks_pd = O.to_runs_dict(pop_pd, obj, O.make_decode("PD"), controller_type="PD")
    assert all(p["Ki"] is None for p in picks_pd.values())


# --- NSGA-II history -------------------------------------------------------

def test_encode_decode_round_trips_exactly():
    cfg = compose_config(SCENARIOS / "slew_55_65_15.yaml",
                          CONTROLLERS / "pid_example.yaml")
    # Ki is overridden to a ratio safely inside KI_RATIO_BOUNDS (1e-4..1e-1)
    # rather than trusting whatever pid_example.yaml currently tunes to --
    # this test is about the encode/decode contract round-tripping exactly
    # for an in-bounds gain set, not about that preset's specific numbers
    # (which legitimately drift with hand-tuning). Out-of-bounds clipping has
    # its own test right below.
    cfg.controller.Ki = np.diag(cfg.controller.Kp) * 1e-2
    kp, kd, ki = O.make_decode("PID")(O.encode_config_gains(cfg, "PID"))
    assert np.allclose(kp, np.diag(cfg.controller.Kp))
    assert np.allclose(kd, np.diag(cfg.controller.Kd))
    assert np.allclose(ki, cfg.controller.Ki)


def test_encode_clips_out_of_bounds_gains_into_the_search_space():
    cfg = _base_config()
    cfg.controller.Kp = np.diag([10.0, 10.0, 10.0])   # far above KP_BOUNDS
    genes = O.encode_config_gains(cfg, "PD")
    assert np.all((genes >= 0.0) & (genes <= 1.0))


def test_encode_returns_none_for_a_preset_without_pd_gains():
    # A Wie preset derives K/D from J and stores no Kp/Kd, so there is
    # nothing to warm-start from -- that must be reported, not crash.
    cfg = compose_config(SCENARIOS / "slew_40_30_25.yaml",
                          CONTROLLERS / "wie_case3.yaml")
    assert O.encode_config_gains(cfg, "PD") is None


def test_encode_pd_preset_for_pid_search_seeds_a_midrange_ratio():
    cfg = _base_config()          # pd_baseline.yaml, no Ki
    assert cfg.controller.Ki is None
    _, _, ki = O.make_decode("PID")(O.encode_config_gains(cfg, "PID"))
    kp = np.diag(cfg.controller.Kp)
    ratio = ki / kp
    assert np.all(ratio > O.KI_RATIO_BOUNDS[0])
    assert np.all(ratio < O.KI_RATIO_BOUNDS[1])


def test_warm_start_puts_the_preset_gains_in_the_initial_population():
    cfg = _base_config()
    pop, obj, decode, ga = O.tune(cfg, controller_type="PD", pop_size=4,
                                   n_generations=0, seed=0, verbose=False,
                                   duration=1.0, warm_start=True)
    seeded = O.encode_config_gains(cfg, "PD")
    # generation 0's front is drawn from the initial population, which must
    # contain the seed verbatim.
    assert np.any([np.allclose(ind, seeded) for ind in pop]) or len(pop) < 4


def test_nsga2_seed_individuals_land_in_the_initial_population():
    captured = {}

    def evaluate(d):
        captured.setdefault("seen", []).append(np.array(d))
        return (float(np.sum(d)), float(np.prod(d)))

    ga = NSGA2(n_genes=3, evaluate=evaluate, decode=lambda g: g,
               pop_size=5, seed=0, verbose_eval=False)
    seed = np.array([[0.11, 0.22, 0.33]])
    ga.run(0, verbose=False, seed_individuals=seed)
    assert any(np.allclose(s, seed[0]) for s in captured["seen"])


def test_nsga2_rejects_seed_with_wrong_gene_count():
    ga = NSGA2(n_genes=3, evaluate=lambda d: (0.0, 0.0), decode=lambda g: g,
               pop_size=4, seed=0, verbose_eval=False)
    with pytest.raises(ValueError, match="3 genes"):
        ga.run(0, verbose=False, seed_individuals=np.zeros((1, 5)))


def test_nsga2_records_history_including_generation_zero():
    ga = NSGA2(n_genes=3, evaluate=lambda d: (float(np.sum(d)), float(np.prod(d))),
               decode=lambda g: g, pop_size=6, seed=0, verbose_eval=False)
    ga.run(3, verbose=False)
    assert [h["generation"] for h in ga.history] == [0, 1, 2, 3]
    assert all(h["best_per_objective"].shape == (2,) for h in ga.history)
    assert all(h["front_size"] >= 1 for h in ga.history)


def test_nsga2_best_objective_never_worsens():
    ga = NSGA2(n_genes=3, evaluate=lambda d: (float(np.sum(d)), float(np.prod(d))),
               decode=lambda g: g, pop_size=8, seed=1, verbose_eval=False)
    ga.run(4, verbose=False)
    best = np.array([h["best_per_objective"] for h in ga.history])
    # NSGA-II is elitist (parents compete with offspring), so the best value
    # of each objective is monotonically non-increasing.
    assert np.all(np.diff(best, axis=0) <= 1e-12)


# --- end to end ------------------------------------------------------------

@pytest.mark.parametrize("ctype,expected_genes", [("PD", 6), ("PID", 9)])
def test_tune_end_to_end_tiny(ctype, expected_genes):
    pop, obj, decode, ga = O.tune(_base_config(), controller_type=ctype,
                                   pop_size=4, n_generations=1, seed=0,
                                   verbose=False, duration=2.0)
    assert pop.shape[1] == expected_genes
    assert obj.shape[1] == 3
    assert len(pop) == len(obj)
    assert len(ga.history) == 2
    assert len(O.decoded_to_flat(decode(pop[0]), ctype)) == expected_genes


def test_tune_warns_when_pid_has_no_disturbance():
    with pytest.warns(RuntimeWarning, match="disturbance-free"):
        O.tune(_base_config(disturbed=False), controller_type="PID",
               pop_size=4, n_generations=0, seed=0, verbose=False, duration=1.0)


def test_tune_does_not_warn_for_pd():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        O.tune(_base_config(disturbed=False), controller_type="PD",
               pop_size=4, n_generations=0, seed=0, verbose=False, duration=1.0)
