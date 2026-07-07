"""Low-rank imputation for partially observed arms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ImputationError(ValueError):
    """Raised when an arm cannot be imputed under the supplied subspace."""


def impute_arm(
    masked_arm: NDArray[np.float64],
    mask: NDArray[np.bool_],
    U: NDArray[np.float64],
    impute_ridge: float = 0.0,
    rank_tol: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Impute a single arm and return `(x_hat, latent_coefficients)`.

    Observed coordinates are kept exactly equal to `masked_arm[mask]`; missing
    coordinates are filled with the low-rank reconstruction `U_missing @ a_hat`.
    """

    x = np.asarray(masked_arm, dtype=float)
    observed = np.asarray(mask, dtype=bool)
    basis = np.asarray(U, dtype=float)

    if x.ndim != 1:
        raise ValueError("masked_arm must have shape (d,).")
    if observed.shape != x.shape:
        raise ValueError("mask must have shape (d,).")
    if basis.ndim != 2 or basis.shape[0] != x.shape[0]:
        raise ValueError("U must have shape (d, m).")
    if impute_ridge < 0.0:
        raise ValueError("impute_ridge must be nonnegative.")

    m = basis.shape[1]
    U_obs = basis[observed, :]
    x_obs = x[observed]
    gram = U_obs.T @ U_obs
    if impute_ridge > 0.0:
        gram = gram + impute_ridge * np.eye(m)
    else:
        rank = np.linalg.matrix_rank(gram, tol=rank_tol)
        if rank < m:
            raise ImputationError(
                "The observed coordinates do not identify the latent coefficients. "
                "Increase burn-in/observation probability or set impute_ridge > 0."
            )

    try:
        coefficients = np.linalg.solve(gram, U_obs.T @ x_obs)
    except np.linalg.LinAlgError as exc:
        raise ImputationError(
            "Failed to solve the imputation system. Increase burn-in or set "
            "impute_ridge > 0."
        ) from exc

    x_hat = x.copy()
    missing = ~observed
    x_hat[missing] = basis[missing, :] @ coefficients
    return x_hat, coefficients


def impute_arms(
    masked_arms: NDArray[np.float64],
    masks: NDArray[np.bool_],
    U: NDArray[np.float64],
    impute_ridge: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Impute a batch of arms with shapes `(K, d)` and `(K, d)`."""

    X = np.asarray(masked_arms, dtype=float)
    M = np.asarray(masks, dtype=bool)
    if X.ndim != 2:
        raise ValueError("masked_arms must have shape (K, d).")
    if M.shape != X.shape:
        raise ValueError("masks must have shape (K, d).")

    imputed = []
    coefficients = []
    for arm, mask in zip(X, M):
        x_hat, a_hat = impute_arm(arm, mask, U, impute_ridge=impute_ridge)
        imputed.append(x_hat)
        coefficients.append(a_hat)
    return np.vstack(imputed), np.vstack(coefficients)
