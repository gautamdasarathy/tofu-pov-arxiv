"""OFUL primitives shared by TOFU-POV and baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class OFULModel:
    """Small linear OFUL state container."""

    dimension: int
    lambda_reg: float
    V: NDArray[np.float64] = field(init=False)
    V_inv: NDArray[np.float64] = field(init=False)
    y: NDArray[np.float64] = field(init=False)
    n_updates: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if self.lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive.")
        self.reset()

    def reset(self) -> None:
        self.V = self.lambda_reg * np.eye(self.dimension)
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.dimension)
        self.y = np.zeros(self.dimension)
        self.n_updates = 0

    @property
    def theta_hat(self) -> NDArray[np.float64]:
        return self.V_inv @ self.y

    def uncertainty(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        Z = np.asarray(features, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != self.dimension:
            raise ValueError("features must have shape (K, dimension).")
        solved = Z @ self.V_inv
        values = np.einsum("ij,ij->i", Z, solved)
        return np.sqrt(np.maximum(values, 0.0))

    def scores(
        self,
        features: NDArray[np.float64],
        beta: float,
    ) -> NDArray[np.float64]:
        Z = np.asarray(features, dtype=float)
        scores = Z @ self.theta_hat + beta * self.uncertainty(Z)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("OFUL produced non-finite UCB scores.")
        return scores

    def select(
        self,
        features: NDArray[np.float64],
        beta: float,
    ) -> tuple[int, NDArray[np.float64]]:
        scores = self.scores(features, beta)
        return int(np.argmax(scores)), scores

    def update(self, feature: NDArray[np.float64], reward: float) -> None:
        z = np.asarray(feature, dtype=float)
        if z.shape != (self.dimension,):
            raise ValueError("feature must have shape (dimension,).")
        self.V = self.V + np.outer(z, z)
        solved = self.V_inv @ z
        denominator = 1.0 + float(z @ solved)
        if denominator <= 0.0:
            raise FloatingPointError("OFUL inverse update encountered a non-positive denominator.")
        self.V_inv = self.V_inv - np.outer(solved, solved) / denominator
        self.V_inv = 0.5 * (self.V_inv + self.V_inv.T)
        self.y = self.y + z * float(reward)
        self.n_updates += 1

    def logdet_ratio(self) -> float:
        sign, logdet = np.linalg.slogdet(self.V)
        if sign <= 0:
            raise FloatingPointError("OFUL design matrix is not positive definite.")
        return float(logdet - self.dimension * np.log(self.lambda_reg))


def oful_confidence_radius(
    model: OFULModel,
    S: float,
    R: float,
    delta: float,
    bias: float = 0.0,
    bias_rounds: int = 0,
) -> float:
    """Compute a standard OFUL radius plus an optional TOFU-POV bias term."""

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")
    if S < 0.0 or R < 0.0 or bias < 0.0:
        raise ValueError("S, R, and bias must be nonnegative.")

    log_term = 0.5 * model.logdet_ratio() - np.log(delta)
    stochastic = R * np.sqrt(2.0 * max(log_term, 0.0))
    regularization = np.sqrt(model.lambda_reg) * S
    approximation = bias * np.sqrt(max(int(bias_rounds), 0))
    return float(regularization + stochastic + approximation)
