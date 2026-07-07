"""Real-feature, synthetic-reward bandit builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.real_world import ArrayBanditEnv
from tofu_pov.real_world_datasets import load_real_feature_matrix


@dataclass(frozen=True)
class RealFeatureSyntheticDataset:
    """Low-rank linear bandit instance derived from real covariates."""

    name: str
    masked_arms: NDArray[np.float64]
    masks: NDArray[np.bool_]
    full_arms: NDArray[np.float64]
    rewards: NDArray[np.float64]
    U: NDArray[np.float64]
    theta_star: NDArray[np.float64]
    latent_arms: NDArray[np.float64]
    metadata: dict[str, Any]

    @property
    def T(self) -> int:
        return int(self.full_arms.shape[0])

    @property
    def K(self) -> int:
        return int(self.full_arms.shape[1])

    @property
    def d(self) -> int:
        return int(self.full_arms.shape[2])

    @property
    def m(self) -> int:
        return int(self.U.shape[1])

    def as_env(self, reward_noise_std: float = 0.0, seed: int | None = None) -> ArrayBanditEnv:
        return ArrayBanditEnv(
            masked_arms=self.masked_arms,
            masks=self.masks,
            rewards=self.rewards,
            full_arms=self.full_arms,
            reward_noise_std=reward_noise_std,
            seed=seed,
        )


def load_real_feature_synthetic_dataset(
    dataset_name: str,
    *,
    p: float,
    seed: int,
    T: int,
    K: int,
    d: int,
    m: int,
    perturbation_std: float = 0.0,
    allow_downloads: bool = False,
    base_feature_dim: int | None = None,
) -> RealFeatureSyntheticDataset:
    """Build a low-rank linear bandit from a real feature matrix."""

    feature_data = load_real_feature_matrix(
        dataset_name,
        seed=seed,
        base_feature_dim=base_feature_dim,
        allow_downloads=allow_downloads,
    )
    return make_real_feature_synthetic_dataset(
        feature_data.X,
        source_name=dataset_name,
        p=p,
        seed=seed,
        T=T,
        K=K,
        d=d,
        m=m,
        perturbation_std=perturbation_std,
    )


def make_real_feature_synthetic_dataset(
    X: NDArray[np.float64],
    *,
    source_name: str,
    p: float,
    seed: int,
    T: int,
    K: int,
    d: int,
    m: int,
    perturbation_std: float = 0.0,
) -> RealFeatureSyntheticDataset:
    """Use real feature factors as latent arms, then embed them in `R^d`.

    The resulting full arms are exactly rank-`m` when `perturbation_std=0`.
    This isolates the missing-feature and ambient-dimension effects while
    preserving a non-Gaussian latent distribution from real covariates.
    """

    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1].")
    if T <= 0 or K <= 0 or d <= 0 or m <= 0:
        raise ValueError("T, K, d, and m must be positive.")
    if m > d:
        raise ValueError("m must be no larger than d.")
    if perturbation_std < 0.0:
        raise ValueError("perturbation_std must be nonnegative.")

    rng = np.random.default_rng(seed)
    features = _standardize_rows(np.asarray(X, dtype=float))
    latent_pool = _real_feature_latents(features, m=m)
    n = latent_pool.shape[0]
    sampled = rng.choice(n, size=T * K, replace=True)
    latent_arms = latent_pool[sampled].reshape(T, K, m)

    loading = np.linspace(1.25, 0.75, m)
    latent_arms = latent_arms * loading.reshape(1, 1, m)
    U = _random_orthonormal_basis(d=d, m=m, rng=rng)
    full_arms = np.einsum("tkm,dm->tkd", latent_arms, U, optimize=True)
    if perturbation_std > 0.0:
        full_arms = full_arms + rng.normal(scale=perturbation_std, size=full_arms.shape)

    max_norm = float(max(np.max(np.linalg.norm(full_arms, axis=2)), 1e-12))
    full_arms = full_arms / max_norm
    latent_arms = latent_arms / max_norm

    theta_low = rng.normal(size=m)
    theta_low = theta_low / max(np.linalg.norm(theta_low), 1e-12)
    theta_star = U @ theta_low
    rewards = np.einsum("tkd,d->tk", full_arms, theta_star, optimize=True)

    masks = np.ones_like(full_arms, dtype=bool) if p == 1.0 else rng.random(size=full_arms.shape) < p
    masked_arms = np.where(masks, full_arms, 0.0)

    metadata: dict[str, Any] = {
        "source_name": source_name,
        "seed": seed,
        "p": p,
        "T": T,
        "K": K,
        "d": d,
        "m": m,
        "perturbation_std": perturbation_std,
        "mask_rate": float(np.mean(masks)),
        "max_arm_norm": float(np.max(np.linalg.norm(full_arms, axis=2))),
        "mean_reward_gap": float(np.mean(np.max(rewards, axis=1) - np.mean(rewards, axis=1))),
    }
    name = f"{source_name}_lr_d{d}_m{m}"
    if perturbation_std > 0.0:
        name += f"_eps{perturbation_std:g}"
    return RealFeatureSyntheticDataset(
        name=name,
        masked_arms=masked_arms,
        masks=masks,
        full_arms=full_arms,
        rewards=rewards,
        U=U,
        theta_star=theta_star,
        latent_arms=latent_arms,
        metadata=metadata,
    )


def _standardize_rows(X: NDArray[np.float64]) -> NDArray[np.float64]:
    if X.ndim != 2:
        raise ValueError("X must have shape (n_examples, n_features).")
    Z = np.nan_to_num(X, copy=True)
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    scale = np.std(Z, axis=0, keepdims=True)
    Z = Z / np.where(scale > 1e-12, scale, 1.0)
    row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(row_norms > 1e-12, row_norms, 1.0)


def _real_feature_latents(features: NDArray[np.float64], *, m: int) -> NDArray[np.float64]:
    if m > min(features.shape):
        raise ValueError("m must be no larger than min(n_examples, n_features).")
    _, _, vt = np.linalg.svd(features, full_matrices=False)
    latents = features @ vt[:m].T
    latents = latents - np.mean(latents, axis=0, keepdims=True)
    scale = np.std(latents, axis=0, keepdims=True)
    latents = latents / np.where(scale > 1e-12, scale, 1.0)
    row_norms = np.linalg.norm(latents, axis=1, keepdims=True)
    return latents / np.sqrt(float(m)) / np.where(row_norms > 1e-12, row_norms, 1.0)


def _random_orthonormal_basis(d: int, m: int, rng: np.random.Generator) -> NDArray[np.float64]:
    raw = rng.normal(size=(d, m))
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return basis[:, :m]
