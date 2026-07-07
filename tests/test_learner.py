import numpy as np

from tofu_pov import TOFUPOV, TOFUPOVConfig
from tofu_pov.oful import OFULModel


def _config(t_b=2, T=8):
    return TOFUPOVConfig(
        d=4,
        m=2,
        K=3,
        p=1.0,
        lambda_reg=1.0,
        t_b=t_b,
        T=T,
        delta=0.05,
        L=2.0,
        S=1.0,
        R=0.01,
        lambda_1=1.0,
        lambda_m=0.5,
        M=1.0,
        random_seed=0,
    )


def _config_with(**overrides):
    values = {
        "d": 4,
        "m": 2,
        "K": 3,
        "p": 1.0,
        "lambda_reg": 1.0,
        "t_b": 2,
        "T": 8,
        "delta": 0.05,
        "L": 2.0,
        "S": 1.0,
        "R": 0.01,
        "lambda_1": 1.0,
        "lambda_m": 0.5,
        "M": 1.0,
        "random_seed": 0,
    }
    values.update(overrides)
    return TOFUPOVConfig(**values)


def _round(value=1.0):
    X = np.array(
        [
            [value, 0.0, 0.0, 0.0],
            [0.0, value, 0.0, 0.0],
            [value, value, 0.0, 0.0],
        ]
    )
    return X, np.ones_like(X, dtype=bool)


def test_burn_in_collects_history_without_oful_updates():
    learner = TOFUPOV(_config(t_b=2, T=8))

    for _ in range(2):
        action = learner.observe(*_round())
        assert 0 <= action < 3
        learner.update(0.0)

    state = learner.state_dict()
    assert state["history_size"] == 6
    assert state["epoch_start"] is None
    assert state["V"] is None
    assert state["burnin_updates"] == 2


def test_epoch_starts_and_oful_resets_on_doubling_schedule():
    learner = TOFUPOV(_config(t_b=1, T=8))

    learner.observe(*_round(1.0))
    learner.update(0.0)

    learner.observe(*_round(2.0))
    learner.update(1.0)
    assert learner.state_dict()["epoch_starts"] == [2]
    assert learner.state_dict()["V"] is not None

    learner.observe(*_round(3.0))
    learner.update(1.0)
    v_before_reset = learner.state_dict()["V"].copy()

    learner.observe(*_round(4.0))
    v_after_reset_before_update = learner.state_dict()["V"].copy()
    assert learner.state_dict()["epoch_starts"] == [2, 4]
    assert not np.allclose(v_before_reset, v_after_reset_before_update)
    learner.update(1.0)


def test_action_scores_are_finite_and_action_is_valid():
    learner = TOFUPOV(_config(t_b=1, T=5))
    learner.observe(*_round())
    learner.update(0.0)

    action = learner.observe(*_round())

    assert 0 <= action < 3
    assert learner.last_scores.shape == (3,)
    assert np.all(np.isfinite(learner.last_scores))


def test_oful_tie_breaking_uses_first_max_index():
    model = OFULModel(dimension=2, lambda_reg=1.0)
    features = np.zeros((4, 2))

    action, scores = model.select(features, beta=1.0)

    assert action == 0
    np.testing.assert_allclose(scores, np.zeros(4))


def test_zero_oful_burnin_updates_ambient_burnin_model():
    learner = TOFUPOV(_config_with(burnin_policy="zero_oful"))

    learner.observe(*_round())
    learner.update(1.0)
    state = learner.state_dict()

    assert state["burnin_policy"] == "zero_oful"
    assert state["burnin_updates"] == 1
    assert state["burnin_oful_updates"] == 1


def test_warm_start_reuses_burnin_rewards_in_first_epoch():
    learner = TOFUPOV(
        _config_with(
            t_b=2,
            T=8,
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
        )
    )

    for _ in range(2):
        learner.observe(*_round())
        learner.update(1.0)

    learner.observe(*_round())
    state = learner.state_dict()

    assert state["epoch_start"] == 3
    assert state["warm_start_from_burnin"] is True
    assert state["burnin_updates"] == 2
    assert learner.oful.n_updates == 2
    learner.update(1.0)
    assert learner.oful.n_updates == 3


def test_first_epoch_warm_start_does_not_replay_at_later_epochs():
    learner = TOFUPOV(
        _config_with(
            t_b=1,
            T=8,
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
            warm_start_replay="first_epoch",
        )
    )

    learner.observe(*_round(1.0))
    learner.update(1.0)

    learner.observe(*_round(2.0))
    assert learner.state_dict()["epoch_starts"] == [2]
    assert learner.oful.n_updates == 1
    learner.update(1.0)

    learner.observe(*_round(3.0))
    learner.update(1.0)

    learner.observe(*_round(4.0))
    assert learner.state_dict()["epoch_starts"] == [2, 4]
    assert learner.state_dict()["warm_started_epoch_count"] == 1
    assert learner.oful.n_updates == 0


def test_every_epoch_warm_start_replays_burnin_at_later_epochs():
    learner = TOFUPOV(
        _config_with(
            t_b=1,
            T=8,
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
            warm_start_replay="every_epoch",
        )
    )

    learner.observe(*_round(1.0))
    learner.update(1.0)

    learner.observe(*_round(2.0))
    assert learner.oful.n_updates == 1
    learner.update(1.0)

    learner.observe(*_round(3.0))
    learner.update(1.0)

    learner.observe(*_round(4.0))
    assert learner.state_dict()["epoch_starts"] == [2, 4]
    assert learner.state_dict()["warm_started_epoch_count"] == 2
    assert learner.oful.n_updates == 1


def test_full_history_warm_start_reprojects_all_selected_rewards_each_epoch():
    learner = TOFUPOV(
        _config_with(
            t_b=1,
            T=8,
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
            warm_start_replay="full_history_every_epoch",
        )
    )

    learner.observe(*_round(1.0))
    learner.update(1.0)

    learner.observe(*_round(2.0))
    assert learner.state_dict()["epoch_starts"] == [2]
    assert learner.oful.n_updates == 1
    learner.update(1.0)

    learner.observe(*_round(3.0))
    learner.update(1.0)

    learner.observe(*_round(4.0))
    state = learner.state_dict()
    assert state["epoch_starts"] == [2, 4]
    assert state["warm_start_replay"] == "full_history_every_epoch"
    assert state["warm_started_epoch_count"] == 2
    assert state["full_history_replayed_epoch_count"] == 2
    assert state["selected_updates"] == 3
    assert learner.oful.n_updates == 3
    learner.update(1.0)
    assert learner.oful.n_updates == 4


def test_threshold_rank_selection_sets_active_epoch_dimension():
    learner = TOFUPOV(
        _config_with(
            d=4,
            m=4,
            p=1.0,
            t_b=2,
            T=8,
            rank_selection="threshold",
            min_rank=1,
            max_rank=4,
            rank_threshold_constant=1.0,
            covariance_radius_schedule=lambda tau_e, n_history: 0.01,
        )
    )

    for _ in range(2):
        learner.observe(*_round())
        learner.update(1.0)

    learner.observe(*_round())
    state = learner.state_dict()

    assert state["rank_selection"] == "threshold"
    assert state["active_m"] == 2
    assert state["rank_history"] == [2]
    assert state["U_hat"].shape == (4, 2)
    assert state["V"].shape == (2, 2)
