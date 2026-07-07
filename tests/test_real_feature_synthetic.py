from __future__ import annotations

import numpy as np

from tofu_pov import make_real_feature_synthetic_dataset


def test_real_feature_synthetic_builder_creates_masked_low_rank_instance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))

    data = make_real_feature_synthetic_dataset(
        X,
        source_name="unit",
        p=0.5,
        seed=7,
        T=12,
        K=4,
        d=10,
        m=3,
    )

    assert data.full_arms.shape == (12, 4, 10)
    assert data.masked_arms.shape == data.full_arms.shape
    assert data.rewards.shape == (12, 4)
    assert data.U.shape == (10, 3)
    assert data.theta_star.shape == (10,)
    assert np.allclose(data.masked_arms, np.where(data.masks, data.full_arms, 0.0))
    assert np.allclose(data.U.T @ data.U, np.eye(3))
    assert np.linalg.matrix_rank(data.full_arms.reshape(-1, 10), tol=1e-8) <= 3


def test_real_feature_synthetic_builder_is_deterministic_and_p_one_is_full_observation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 5))

    first = make_real_feature_synthetic_dataset(
        X,
        source_name="unit",
        p=1.0,
        seed=4,
        T=8,
        K=3,
        d=8,
        m=2,
    )
    second = make_real_feature_synthetic_dataset(
        X,
        source_name="unit",
        p=1.0,
        seed=4,
        T=8,
        K=3,
        d=8,
        m=2,
    )

    assert np.all(first.masks)
    assert np.allclose(first.masked_arms, first.full_arms)
    assert np.allclose(first.full_arms, second.full_arms)
    assert np.allclose(first.rewards, second.rewards)
    assert np.allclose(first.theta_star, second.theta_star)
