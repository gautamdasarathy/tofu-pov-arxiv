from __future__ import annotations

import numpy as np
import pytest

from tofu_pov import FullInformationOFUL, OracleSubspaceOFUL, ZeroImputedOFUL


def test_zero_imputed_oful_beta_scale_preserves_default_and_changes_scores():
    arms = np.array([[1.0, 0.0], [0.0, 2.0]])
    masks = np.ones_like(arms, dtype=bool)

    default_policy = ZeroImputedOFUL(d=2, lambda_reg=1.0, S=1.0, R=0.1, delta=0.05)
    explicit_policy = ZeroImputedOFUL(d=2, lambda_reg=1.0, S=1.0, R=0.1, delta=0.05, beta_scale=1.0)
    small_policy = ZeroImputedOFUL(d=2, lambda_reg=1.0, S=1.0, R=0.1, delta=0.05, beta_scale=0.5)
    large_policy = ZeroImputedOFUL(d=2, lambda_reg=1.0, S=1.0, R=0.1, delta=0.05, beta_scale=2.0)

    default_policy.observe(arms, masks)
    explicit_policy.observe(arms, masks)
    small_policy.observe(arms, masks)
    large_policy.observe(arms, masks)

    np.testing.assert_allclose(default_policy.last_scores, explicit_policy.last_scores)
    assert np.all(large_policy.last_scores > small_policy.last_scores)


@pytest.mark.parametrize(
    "policy",
    [
        ZeroImputedOFUL(d=2, beta_scale=0.5),
        FullInformationOFUL(d=2, beta_scale=0.5),
        OracleSubspaceOFUL(np.eye(2), beta_scale=0.5),
    ],
)
def test_oful_variants_reject_nonpositive_beta_scale(policy):
    del policy
    with pytest.raises(ValueError, match="beta_scale"):
        ZeroImputedOFUL(d=2, beta_scale=0.0)
    with pytest.raises(ValueError, match="beta_scale"):
        FullInformationOFUL(d=2, beta_scale=-1.0)
    with pytest.raises(ValueError, match="beta_scale"):
        OracleSubspaceOFUL(np.eye(2), beta_scale=0.0)


def test_full_information_and_oracle_beta_scale_changes_scores():
    arms = np.array([[1.0, 0.0], [0.0, 2.0]])
    masks = np.ones_like(arms, dtype=bool)

    full_small = FullInformationOFUL(d=2, beta_scale=0.5)
    full_large = FullInformationOFUL(d=2, beta_scale=2.0)
    full_small.observe(arms * 0.0, masks, arms)
    full_large.observe(arms * 0.0, masks, arms)
    assert np.all(full_large.last_scores > full_small.last_scores)

    oracle_small = OracleSubspaceOFUL(np.eye(2), beta_scale=0.5)
    oracle_large = OracleSubspaceOFUL(np.eye(2), beta_scale=2.0)
    oracle_small.observe(arms * 0.0, masks, arms)
    oracle_large.observe(arms * 0.0, masks, arms)
    assert np.all(oracle_large.last_scores > oracle_small.last_scores)
