"""Bandit environments for TOFU-POV experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class BanditEnv(Protocol):
    """Minimal online bandit environment protocol."""

    def reset(self, seed: int | None = None) -> None:
        ...

    def get_round(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
        ...

    def step(self, action_idx: int) -> float:
        ...

    def optimal_reward(self, full_arms: NDArray[np.float64]) -> float:
        ...


@dataclass
class SyntheticLowRankBanditEnv:
    """Synthetic low-rank contextual bandit with Bernoulli feature masks."""

    d: int
    m: int
    K: int
    p: float
    T: int
    noise_std: float = 0.0
    perturbation_std: float = 0.0
    theta_norm: float = 1.0
    loading_min: float = 0.75
    loading_max: float = 1.25
    feature_scale: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.d <= 0 or self.m <= 0 or self.K <= 0 or self.T <= 0:
            raise ValueError("d, m, K, and T must be positive.")
        if self.m > self.d:
            raise ValueError("m must be no larger than d.")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("p must be in (0, 1].")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be nonnegative.")
        if self.perturbation_std < 0.0:
            raise ValueError("perturbation_std must be nonnegative.")
        if self.loading_min <= 0.0 or self.loading_max < self.loading_min:
            raise ValueError("Require 0 < loading_min <= loading_max.")
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        raw = self.rng.normal(size=(self.d, self.m))
        self.U, _ = np.linalg.qr(raw, mode="reduced")
        self.loading = self.rng.uniform(self.loading_min, self.loading_max, size=self.m)

        theta_low = self.rng.normal(size=self.m)
        norm = np.linalg.norm(theta_low)
        if norm == 0.0:
            theta_low[0] = 1.0
            norm = 1.0
        self.theta_low = theta_low / norm * self.theta_norm
        self.theta_star = self.U @ self.theta_low
        self.round_index = 0
        self._current_full_arms: NDArray[np.float64] | None = None
        self._current_masked_arms: NDArray[np.float64] | None = None
        self._current_masks: NDArray[np.bool_] | None = None

    def get_round(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
        if self.round_index >= self.T:
            raise StopIteration("Synthetic environment has no rounds left.")

        latent = self.rng.normal(scale=self.feature_scale, size=(self.K, self.m))
        low_rank_arms = (latent * self.loading) @ self.U.T
        perturbation = self.rng.normal(scale=self.perturbation_std, size=(self.K, self.d))
        full_arms = low_rank_arms + perturbation
        masks = self.rng.random(size=(self.K, self.d)) < self.p
        masked_arms = np.where(masks, full_arms, 0.0)

        self._current_full_arms = full_arms
        self._current_masked_arms = masked_arms
        self._current_masks = masks
        return masked_arms.copy(), masks.copy(), full_arms.copy()

    def reward_mean(
        self,
        full_arms: NDArray[np.float64],
        action_idx: int | None = None,
    ) -> float | NDArray[np.float64]:
        means = np.asarray(full_arms, dtype=float) @ self.theta_star
        if action_idx is None:
            return means
        return float(means[int(action_idx)])

    def step(self, action_idx: int) -> float:
        if self._current_full_arms is None:
            raise RuntimeError("Call get_round() before step(action_idx).")
        action = int(action_idx)
        if action < 0 or action >= self.K:
            raise ValueError("action_idx is out of range.")

        mean = float(self._current_full_arms[action] @ self.theta_star)
        reward = mean + float(self.rng.normal(scale=self.noise_std))
        self.round_index += 1
        return reward

    def optimal_reward(self, full_arms: NDArray[np.float64]) -> float:
        return float(np.max(np.asarray(full_arms, dtype=float) @ self.theta_star))
