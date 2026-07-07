import numpy as np
import pytest

from tofu_pov.imputation import ImputationError, impute_arm


def test_imputation_with_full_observation_returns_original_arm():
    rng = np.random.default_rng(3)
    U, _ = np.linalg.qr(rng.normal(size=(5, 2)), mode="reduced")
    x = rng.normal(size=5)
    mask = np.ones(5, dtype=bool)

    x_hat, _ = impute_arm(x, mask, U)

    np.testing.assert_allclose(x_hat, x)


def test_imputation_recovers_noiseless_low_rank_arm_with_true_subspace():
    rng = np.random.default_rng(4)
    U, _ = np.linalg.qr(rng.normal(size=(6, 2)), mode="reduced")
    coeff = np.array([0.7, -1.3])
    x = U @ coeff
    mask = np.array([True, True, True, True, True, False])
    masked = np.where(mask, x, 0.0)

    x_hat, coeff_hat = impute_arm(masked, mask, U)

    np.testing.assert_allclose(coeff_hat, coeff, atol=1e-10)
    np.testing.assert_allclose(x_hat, x, atol=1e-10)


def test_imputation_raises_clear_error_when_system_is_singular():
    U = np.eye(3, 2)
    x = np.array([1.0, 0.0, 0.0])
    mask = np.array([True, False, False])

    with pytest.raises(ImputationError, match="impute_ridge"):
        impute_arm(x, mask, U)
