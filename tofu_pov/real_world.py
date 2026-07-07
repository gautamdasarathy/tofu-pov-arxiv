"""Array-backed environments for real-world-style experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ArrayBanditEnv:
    """Bandit environment backed by arrays of candidate arms and rewards.

    This is useful for real-world datasets that can be converted into
    `(T, K, d)` candidate features and a `(T, K)` reward matrix, such as
    classification-to-bandit reductions or recommendation data with known
    ratings for the sampled candidate set.
    """

    masked_arms: NDArray[np.float64]
    masks: NDArray[np.bool_]
    rewards: NDArray[np.float64]
    full_arms: NDArray[np.float64] | None = None
    reward_noise_std: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self.masked_arms = np.asarray(self.masked_arms, dtype=float)
        self.masks = np.asarray(self.masks, dtype=bool)
        self.rewards = np.asarray(self.rewards, dtype=float)
        if self.masked_arms.ndim != 3:
            raise ValueError("masked_arms must have shape (T, K, d).")
        if self.masks.shape != self.masked_arms.shape:
            raise ValueError("masks must have shape (T, K, d).")
        if self.rewards.shape != self.masked_arms.shape[:2]:
            raise ValueError("rewards must have shape (T, K).")
        if self.full_arms is None:
            self.full_arms = self.masked_arms.copy()
        else:
            self.full_arms = np.asarray(self.full_arms, dtype=float)
            if self.full_arms.shape != self.masked_arms.shape:
                raise ValueError("full_arms must have shape (T, K, d).")
        if self.reward_noise_std < 0.0:
            raise ValueError("reward_noise_std must be nonnegative.")

        self.T, self.K, self.d = self.masked_arms.shape
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)
        self.round_index = 0
        self._current_index: int | None = None

    def get_round(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
        if self.round_index >= self.T:
            raise StopIteration("ArrayBanditEnv has no rounds left.")
        idx = self.round_index
        self._current_index = idx
        return (
            self.masked_arms[idx].copy(),
            self.masks[idx].copy(),
            self.full_arms[idx].copy(),
        )

    def reward_mean(
        self,
        full_arms: NDArray[np.float64],
        action_idx: int | None = None,
    ) -> float | NDArray[np.float64]:
        del full_arms
        if self._current_index is None:
            raise RuntimeError("Call get_round() before reward_mean().")
        means = self.rewards[self._current_index]
        if action_idx is None:
            return means.copy()
        return float(means[int(action_idx)])

    def step(self, action_idx: int) -> float:
        if self._current_index is None:
            raise RuntimeError("Call get_round() before step(action_idx).")
        action = int(action_idx)
        if action < 0 or action >= self.K:
            raise ValueError("action_idx is out of range.")
        reward = float(self.rewards[self._current_index, action])
        if self.reward_noise_std > 0.0:
            reward += float(self.rng.normal(scale=self.reward_noise_std))
        self.round_index += 1
        return reward

    def optimal_reward(self, full_arms: NDArray[np.float64]) -> float:
        del full_arms
        if self._current_index is None:
            raise RuntimeError("Call get_round() before optimal_reward().")
        return float(np.max(self.rewards[self._current_index]))
