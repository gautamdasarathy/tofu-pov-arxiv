from __future__ import annotations

import numpy as np

from tofu_pov.oful import OFULModel


def test_oful_inverse_state_matches_direct_solve_after_updates():
    rng = np.random.default_rng(123)
    model = OFULModel(dimension=5, lambda_reg=0.7)

    for _ in range(20):
        feature = rng.normal(size=5)
        reward = float(rng.normal())
        model.update(feature, reward)

    np.testing.assert_allclose(model.V_inv, np.linalg.inv(model.V), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.theta_hat, np.linalg.solve(model.V, model.y), rtol=1e-10, atol=1e-10)

    features = rng.normal(size=(7, 5))
    solved = np.linalg.solve(model.V, features.T).T
    expected_uncertainty = np.sqrt(np.maximum(np.einsum("ij,ij->i", features, solved), 0.0))
    np.testing.assert_allclose(model.uncertainty(features), expected_uncertainty, rtol=1e-10, atol=1e-10)
