import numpy as np

from tofu_pov.subspace import (
    corrected_covariance,
    estimate_subspace,
    subspace_distance,
    threshold_rank,
)


def test_corrected_covariance_matches_full_covariance_when_p_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))

    corrected = corrected_covariance(X, p=1.0)
    expected = X.T @ X / X.shape[0]

    np.testing.assert_allclose(corrected, expected, atol=1e-12)


def test_estimate_subspace_matches_pca_when_p_is_one():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(6, 2))
    U, _ = np.linalg.qr(raw, mode="reduced")
    Z = rng.normal(size=(500, 2))
    X = Z @ U.T

    U_hat, _ = estimate_subspace(X, p=1.0, m=2)

    assert subspace_distance(U, U_hat) < 1e-10


def test_corrected_covariance_approaches_full_covariance_under_masking():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30000, 5))
    p = 0.7
    masks = rng.random(size=X.shape) < p
    masked = np.where(masks, X, 0.0)

    corrected = corrected_covariance(masked, p=p)
    full = X.T @ X / X.shape[0]

    np.testing.assert_allclose(corrected, full, atol=0.04)


def test_threshold_rank_counts_large_eigenvalues_and_clamps():
    eigenvalues = np.array([5.0, 2.0, 0.2, 0.01])

    assert threshold_rank(eigenvalues, threshold=1.0, min_rank=1, max_rank=4) == 2
    assert threshold_rank(eigenvalues, threshold=10.0, min_rank=1, max_rank=4) == 1
    assert threshold_rank(eigenvalues, threshold=0.0, min_rank=1, max_rank=3) == 3
