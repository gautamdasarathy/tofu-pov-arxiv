"""Subspace estimation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def corrected_covariance(masked_arms: NDArray[np.float64], p: float) -> NDArray[np.float64]:
    """Estimate the full-arm covariance from Bernoulli-masked observations.

    `masked_arms` is an `(n, d)` matrix with zeros in missing entries. The
    estimator is unbiased when each coordinate is independently observed with
    probability `p`.
    """

    X = np.asarray(masked_arms, dtype=float)
    if X.ndim != 2:
        raise ValueError("masked_arms must have shape (n, d).")
    if X.shape[0] == 0:
        raise ValueError("At least one masked arm is required.")
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1].")

    n, d = X.shape
    sigma = (X.T @ X) / (n * p * p)
    diagonal_second_moment = np.mean(X * X, axis=0)
    diagonal_correction = (1.0 / p - 1.0 / (p * p)) * diagonal_second_moment
    sigma[np.diag_indices(d)] += diagonal_correction
    return (sigma + sigma.T) / 2.0


def estimate_subspace(
    masked_arms: NDArray[np.float64],
    p: float,
    m: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the top-`m` eigenvectors and eigenvalues of corrected covariance."""

    sigma = corrected_covariance(masked_arms, p)
    return estimate_subspace_from_covariance(sigma, m)


def sorted_eigendecomposition(
    covariance: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return eigenvalues/eigenvectors sorted from largest to smallest."""

    sigma = np.asarray(covariance, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must have shape (d, d).")
    sigma = (sigma + sigma.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def estimate_subspace_from_covariance(
    covariance: NDArray[np.float64],
    m: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the top-`m` eigenvectors/eigenvalues of a covariance matrix."""

    sigma = np.asarray(covariance, dtype=float)
    d = sigma.shape[0]
    if m <= 0 or m > d:
        raise ValueError("m must be in [1, d].")
    eigenvalues, eigenvectors = sorted_eigendecomposition(sigma)
    return eigenvectors[:, :m], eigenvalues[:m]


def threshold_rank(
    eigenvalues: NDArray[np.float64],
    threshold: float,
    min_rank: int = 1,
    max_rank: int | None = None,
) -> int:
    """Select rank by counting eigenvalues above a threshold and clamping."""

    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("eigenvalues must have shape (d,).")
    if threshold < 0.0:
        raise ValueError("threshold must be nonnegative.")
    if min_rank <= 0:
        raise ValueError("min_rank must be positive.")
    cap = values.shape[0] if max_rank is None else int(max_rank)
    if cap <= 0 or cap > values.shape[0]:
        raise ValueError("max_rank must be in [1, len(eigenvalues)].")
    if min_rank > cap:
        raise ValueError("min_rank must be no larger than max_rank.")

    selected = int(np.count_nonzero(values[:cap] >= threshold))
    return max(min_rank, min(selected, cap))


def projection_matrix(U: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the orthogonal projector onto the columns of `U`."""

    basis = np.asarray(U, dtype=float)
    if basis.ndim != 2:
        raise ValueError("U must have shape (d, m).")
    return basis @ basis.T


def subspace_distance(U: NDArray[np.float64], V: NDArray[np.float64]) -> float:
    """Spectral distance between two column spaces via projector difference."""

    PU = projection_matrix(U)
    PV = projection_matrix(V)
    return float(np.linalg.norm(PU - PV, ord=2))
