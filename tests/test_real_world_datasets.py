from __future__ import annotations

import numpy as np

from tofu_pov import (
    FullInformationOFUL,
    make_classification_bandit_dataset,
    run_bandit,
)


def test_classification_bandit_builder_shapes_rewards_and_masks_are_consistent():
    X = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [0.5, 0.5, 1.0],
            [-1.0, 0.2, 0.0],
        ]
    )
    y = np.array([0, 1, 2, 1])

    data = make_classification_bandit_dataset(
        X,
        y,
        name="toy_unit",
        p=0.5,
        seed=3,
        T=8,
        d=7,
        class_names=("a", "b", "c"),
    )

    assert data.full_arms.shape == (8, 3, 7)
    assert data.masked_arms.shape == data.full_arms.shape
    assert data.masks.shape == data.full_arms.shape
    assert data.rewards.shape == (8, 3)
    assert np.allclose(data.masked_arms, np.where(data.masks, data.full_arms, 0.0))
    assert np.allclose(data.rewards.sum(axis=1), 1.0)
    assert np.all(data.rewards[np.arange(data.T), data.labels] == 1.0)
    assert data.metadata["mask_rate"] > 0.0


def test_classification_bandit_builder_is_deterministic_and_p_one_is_full_observation():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10) % 2

    first = make_classification_bandit_dataset(X, y, name="det", p=1.0, seed=4, T=6, d=5)
    second = make_classification_bandit_dataset(X, y, name="det", p=1.0, seed=4, T=6, d=5)

    assert np.all(first.masks)
    assert np.allclose(first.masked_arms, first.full_arms)
    assert np.allclose(first.full_arms, second.full_arms)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.rewards, second.rewards)


def test_full_information_oful_uses_full_arms_with_array_env():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    y = np.array([0, 1, 0, 1])
    data = make_classification_bandit_dataset(X, y, name="full_info", p=0.25, seed=1, T=8, d=6)

    result = run_bandit(
        FullInformationOFUL(d=data.d, lambda_reg=1.0, S=1.0, R=0.5, delta=0.05),
        data.as_env(seed=1),
        seed=1,
    )

    assert result.actions.shape == (8,)
    assert result.cumulative_regret.shape == (8,)
    assert np.all(np.isfinite(result.cumulative_regret))
