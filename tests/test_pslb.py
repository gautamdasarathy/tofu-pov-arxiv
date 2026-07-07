import dataclasses

import numpy as np

from tofu_pov import MaskedPSLB, PSLB, PSLBConfig, RandomPolicy, SyntheticLowRankBanditEnv, run_bandit
from tofu_pov.oful import OFULModel


def _config(warmup_rounds=1, use_intersection=True):
    return PSLBConfig(
        d=5,
        m=2,
        K=3,
        T=20,
        lambda_reg=1.0,
        delta=0.05,
        L=4.0,
        S=1.0,
        R=0.01,
        warmup_rounds=warmup_rounds,
        use_intersection=use_intersection,
        random_seed=0,
    )


def _round(value=1.0):
    full = np.array(
        [
            [value, 0.0, 0.0, 0.0, 0.0],
            [0.0, value, 0.0, 0.0, 0.0],
            [value, value, 0.0, 0.0, 0.0],
        ]
    )
    masks = np.ones_like(full, dtype=bool)
    return full.copy(), masks, full


def test_pslb_warmup_collects_decision_sets_and_updates_ambient_model():
    policy = PSLB(_config(warmup_rounds=1))
    masked, masks, full = _round()

    action = policy.observe(masked, masks, full)
    policy.update(1.0)
    state = policy.state_dict()

    assert 0 <= action < 3
    assert state["history_size"] == 3
    assert state["n_selected"] == 1
    assert state["U_hat"] is None
    assert state["warmup_policy"] == "random"
    assert state["warmup_updates"] == 0


def test_pslb_zero_oful_warmup_updates_warmup_model_and_scores():
    policy = PSLB(dataclasses.replace(_config(warmup_rounds=1), warmup_policy="zero_oful"))
    masked, masks, full = _round()

    action = policy.observe(masked, masks, full)
    assert 0 <= action < 3
    assert policy.last_scores is not None
    policy.update(1.0)
    state = policy.state_dict()

    assert state["warmup_policy"] == "zero_oful"
    assert state["warmup_updates"] == 1
    assert state["n_selected"] == 1


def test_pslb_produces_finite_scores_after_warmup():
    policy = PSLB(_config(warmup_rounds=1))

    policy.observe(*_round(1.0))
    policy.update(1.0)
    action = policy.observe(*_round(2.0))
    state = policy.state_dict()

    assert 0 <= action < 3
    assert state["U_hat"].shape == (5, 2)
    assert state["last_scores"].shape == (3,)
    assert state["last_projected_scores"].shape == (3,)
    assert state["last_ambient_scores"].shape == (3,)
    assert np.all(np.isfinite(state["last_scores"]))
    assert state["intersection_method"] == "projected_sampled"
    assert state["last_intersection_sample_count"] > 0
    assert state["last_intersection_feasible_count"] >= 0


def test_pslb_freezes_subspace_after_first_estimate_by_default():
    policy = PSLB(_config(warmup_rounds=0))

    policy.observe(*_round(1.0))
    policy.update(1.0)
    policy.observe(*_round(2.0))
    state = policy.state_dict()

    assert state["subspace_update"] == "once"
    assert len(state["rank_history"]) == 1


def test_pslb_every_round_subspace_update_preserves_legacy_behavior():
    policy = PSLB(dataclasses.replace(_config(warmup_rounds=0), subspace_update="every_round"))

    policy.observe(*_round(1.0))
    policy.update(1.0)
    policy.observe(*_round(2.0))
    state = policy.state_dict()

    assert state["subspace_update"] == "every_round"
    assert len(state["rank_history"]) == 2


def test_pslb_projected_only_mode_runs_without_ambient_intersection():
    policy = PSLB(_config(warmup_rounds=0, use_intersection=False))
    action = policy.observe(*_round())

    assert 0 <= action < 3
    assert policy.last_ambient_scores is not None
    assert np.allclose(policy.last_scores, policy.last_projected_scores)
    assert policy.state_dict()["last_intersection_sample_count"] == 0


def test_pslb_min_ucb_mode_preserves_legacy_intersection_surrogate():
    policy = PSLB(dataclasses.replace(_config(warmup_rounds=0), intersection_method="min_ucb"))
    policy.observe(*_round())

    expected = np.minimum(policy.last_projected_scores, policy.last_ambient_scores)
    np.testing.assert_allclose(policy.last_scores, expected)
    assert policy.state_dict()["intersection_method"] == "min_ucb"
    assert policy.state_dict()["last_intersection_sample_count"] == 0


def test_masked_pslb_ignores_full_arms():
    config = _config(warmup_rounds=0)
    masked_policy = MaskedPSLB(config)
    explicit_policy = PSLB(dataclasses.replace(config, arm_source="masked"))
    masked, masks, full = _round()
    misleading_full = full + 100.0

    masked_action = masked_policy.observe(masked, masks, misleading_full)
    explicit_action = explicit_policy.observe(masked, masks, full * -100.0)

    assert masked_action == explicit_action
    np.testing.assert_allclose(masked_policy.last_scores, explicit_policy.last_scores)


def test_full_pslb_uses_full_arms_when_provided():
    config = _config(warmup_rounds=0)
    full_policy = PSLB(config)
    masked_policy = PSLB(dataclasses.replace(config, arm_source="masked"))
    masked, masks, full = _round()
    altered_full = np.array(
        [
            [0.0, 0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 10.0],
        ]
    )

    full_policy.observe(masked, masks, altered_full)
    masked_policy.observe(masked, masks, full)

    assert not np.allclose(full_policy.last_scores, masked_policy.last_scores)


def test_masked_pslb_threshold_rank_sets_active_dimension():
    config = dataclasses.replace(
        _config(warmup_rounds=0),
        d=5,
        m=5,
        rank_selection="threshold",
        min_rank=1,
        max_rank=5,
        rank_threshold_constant=1.0,
        covariance_radius_schedule=lambda t, n_history: 0.01,
    )
    policy = MaskedPSLB(config)

    policy.observe(*_round())
    state = policy.state_dict()

    assert state["arm_source"] == "masked"
    assert state["rank_selection"] == "threshold"
    assert state["active_m"] == 2
    assert state["U_hat"].shape == (5, 2)
    assert state["rank_history"] == [2]


def test_masked_pslb_preserves_projected_sampled_intersection_config():
    config = dataclasses.replace(
        _config(warmup_rounds=0),
        intersection_method="projected_sampled",
        intersection_sample_count=17,
        intersection_fallback="min_ucb",
        warmup_policy="zero_oful",
    )
    policy = MaskedPSLB(config)

    assert policy.config.arm_source == "masked"
    assert policy.config.intersection_method == "projected_sampled"
    assert policy.config.intersection_sample_count == 17
    assert policy.config.intersection_fallback == "min_ucb"
    assert policy.config.warmup_policy == "zero_oful"


def test_sampled_alias_maps_to_projected_sampled():
    config = dataclasses.replace(_config(warmup_rounds=0), intersection_method="sampled")

    assert config.intersection_method == "projected_sampled"


def test_projected_sampled_intersection_does_not_add_ambient_residual_reward():
    policy = PSLB(
        PSLBConfig(
            d=2,
            m=1,
            K=2,
            T=10,
            lambda_reg=1.0,
            delta=0.05,
            L=2.0,
            S=1.0,
            R=0.01,
            warmup_rounds=0,
            random_seed=0,
            intersection_sample_count=0,
        )
    )
    policy.U_hat = np.array([[1.0], [0.0]])
    policy.ambient_model = OFULModel(2, 1.0)
    policy.ambient_model.y = np.array([0.0, 10.0])
    projected_model = OFULModel(1, 1.0)
    arms = np.array([[1.0, 0.0], [0.0, 1.0]])
    projected_scores = np.array([1.0, 0.0])
    ambient_scores = np.array([1.0, 11.0])

    scores = policy._sampled_intersection_scores(
        arms=arms,
        projected_model=projected_model,
        projected_beta=1.0,
        projected_scores=projected_scores,
        ambient_beta=20.0,
        ambient_scores=ambient_scores,
    )

    assert scores[1] == 0.0
    assert policy.last_intersection_feasible_count > 0


def test_ambient_sampled_diagnostic_keeps_residual_lift_behavior():
    policy = PSLB(
        PSLBConfig(
            d=2,
            m=1,
            K=2,
            T=10,
            lambda_reg=1.0,
            delta=0.05,
            L=2.0,
            S=1.0,
            R=0.01,
            warmup_rounds=0,
            random_seed=0,
            intersection_method="ambient_sampled",
            intersection_sample_count=0,
        )
    )
    policy.U_hat = np.array([[1.0], [0.0]])
    policy.ambient_model = OFULModel(2, 1.0)
    policy.ambient_model.y = np.array([0.0, 10.0])
    projected_model = OFULModel(1, 1.0)
    arms = np.array([[1.0, 0.0], [0.0, 1.0]])
    projected_scores = np.array([1.0, 0.0])
    ambient_scores = np.array([1.0, 11.0])

    scores = policy._ambient_residual_intersection_scores(
        arms=arms,
        projected_model=projected_model,
        projected_beta=1.0,
        projected_scores=projected_scores,
        ambient_beta=20.0,
        ambient_scores=ambient_scores,
    )

    assert scores[1] > 0.0
    assert policy.last_intersection_feasible_count > 0


def test_projected_sampled_intersection_falls_back_when_no_feasible_candidate_exists():
    policy = PSLB(
        PSLBConfig(
            d=2,
            m=1,
            K=2,
            T=10,
            lambda_reg=1.0,
            delta=0.05,
            L=2.0,
            S=1.0,
            R=0.01,
            warmup_rounds=0,
            random_seed=0,
            intersection_sample_count=0,
            intersection_fallback="ambient_ucb",
        )
    )
    policy.U_hat = np.array([[1.0], [0.0]])
    policy.ambient_model = OFULModel(2, 1.0)
    policy.ambient_model.y = np.array([10.0, 0.0])
    projected_model = OFULModel(1, 1.0)
    arms = np.array([[1.0, 0.0], [0.0, 1.0]])
    projected_scores = np.array([1.0, 2.0])
    ambient_scores = np.array([3.0, 4.0])

    scores = policy._sampled_intersection_scores(
        arms=arms,
        projected_model=projected_model,
        projected_beta=0.0,
        projected_scores=projected_scores,
        ambient_beta=0.0,
        ambient_scores=ambient_scores,
    )

    np.testing.assert_allclose(scores, ambient_scores)
    assert policy.last_intersection_feasible_count == 0
    assert policy.last_intersection_fallback == "ambient_ucb"


def test_pslb_end_to_end_synthetic_regret_is_finite():
    T = 50
    env = SyntheticLowRankBanditEnv(
        d=6,
        m=2,
        K=4,
        p=0.8,
        T=T,
        noise_std=0.0,
        perturbation_std=0.01,
        seed=12,
    )
    policy = PSLB(
        PSLBConfig(
            d=6,
            m=2,
            K=4,
            T=T,
            lambda_reg=1.0,
            delta=0.05,
            L=4.0,
            S=1.0,
            R=0.01,
            warmup_rounds=3,
            random_seed=5,
        )
    )

    result = run_bandit(policy, env, seed=12, policy_seed=5)

    assert result.actions.shape == (T,)
    assert np.all(np.isfinite(result.cumulative_regret))


def test_pslb_is_competitive_with_random_on_easy_low_rank_instance():
    T = 80

    def make_env(seed):
        return SyntheticLowRankBanditEnv(d=6, m=2, K=4, p=1.0, T=T, noise_std=0.0, seed=seed)

    pslb = run_bandit(
        PSLB(
            PSLBConfig(
                d=6,
                m=2,
                K=4,
                T=T,
                lambda_reg=1.0,
                delta=0.05,
                L=4.0,
                S=1.0,
                R=0.01,
                warmup_rounds=3,
                use_intersection=False,
                random_seed=7,
            )
        ),
        make_env(13),
        seed=13,
        policy_seed=7,
    )
    random = run_bandit(RandomPolicy(K=4, seed=7), make_env(13), seed=13, policy_seed=7)

    assert pslb.cumulative_regret[-1] < random.cumulative_regret[-1]
