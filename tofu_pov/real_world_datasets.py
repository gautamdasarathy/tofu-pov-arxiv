"""Dataset builders for real-world-style contextual bandit experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.real_world import ArrayBanditEnv


class DatasetUnavailableError(RuntimeError):
    """Raised when an optional real-world dataset cannot be loaded."""


@dataclass(frozen=True)
class RealWorldBanditDataset:
    """Array-backed classification-to-bandit dataset."""

    name: str
    masked_arms: NDArray[np.float64]
    masks: NDArray[np.bool_]
    full_arms: NDArray[np.float64]
    rewards: NDArray[np.float64]
    labels: NDArray[np.int64]
    class_names: tuple[str, ...]
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

    def as_env(self, reward_noise_std: float = 0.0, seed: int | None = None) -> ArrayBanditEnv:
        return ArrayBanditEnv(
            masked_arms=self.masked_arms,
            masks=self.masks,
            rewards=self.rewards,
            full_arms=self.full_arms,
            reward_noise_std=reward_noise_std,
            seed=seed,
        )


@dataclass(frozen=True)
class RealFeatureMatrix:
    """Loaded real covariates before conversion into bandit arms."""

    name: str
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    class_names: tuple[str, ...]
    metadata: dict[str, Any]


def load_real_feature_matrix(
    name: str,
    *,
    seed: int = 0,
    base_feature_dim: int | None = None,
    allow_downloads: bool = False,
) -> RealFeatureMatrix:
    """Load real covariates for downstream bandit reductions."""

    if name == "toy_classification":
        X, y, class_names = _load_toy_classification(T=2000, seed=seed)
    elif name == "digits_sklearn":
        X, y, class_names = _load_digits_sklearn()
    elif name == "20newsgroups":
        X, y, class_names = _load_20newsgroups(
            seed=seed,
            base_feature_dim=base_feature_dim or 40,
            allow_downloads=allow_downloads,
        )
    elif name == "covertype":
        X, y, class_names = _load_covertype(allow_downloads=allow_downloads)
    else:
        raise ValueError(f"Unknown real feature dataset: {name}")

    return RealFeatureMatrix(
        name=name,
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=np.int64),
        class_names=class_names,
        metadata={
            "name": name,
            "n_examples": int(np.asarray(X).shape[0]),
            "n_features": int(np.asarray(X).shape[1]),
            "n_classes": len(class_names),
        },
    )


def load_real_world_bandit_dataset(
    name: str,
    *,
    p: float,
    seed: int,
    T: int,
    d: int,
    base_feature_dim: int | None = None,
    allow_downloads: bool = False,
) -> RealWorldBanditDataset:
    """Load a named dataset and convert it into a masked classification bandit."""

    if name == "toy_classification":
        X, y, class_names = _load_toy_classification(T=max(T, 240), seed=seed)
    elif name == "digits_sklearn":
        X, y, class_names = _load_digits_sklearn()
    elif name == "20newsgroups":
        X, y, class_names = _load_20newsgroups(
            seed=seed,
            base_feature_dim=base_feature_dim or 40,
            allow_downloads=allow_downloads,
        )
    elif name == "covertype":
        X, y, class_names = _load_covertype(allow_downloads=allow_downloads)
    else:
        raise ValueError(f"Unknown real-world dataset: {name}")

    return make_classification_bandit_dataset(
        X,
        y,
        name=name,
        p=p,
        seed=seed,
        T=T,
        d=d,
        class_names=class_names,
    )


def make_classification_bandit_dataset(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    name: str,
    p: float,
    seed: int,
    T: int,
    d: int,
    class_names: tuple[str, ...] | None = None,
) -> RealWorldBanditDataset:
    """Convert supervised classification data into finite-arm bandit arrays."""

    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1].")
    if T <= 0 or d <= 0:
        raise ValueError("T and d must be positive.")

    rng = np.random.default_rng(seed)
    features = _prepare_features(np.asarray(X, dtype=float))
    labels, inferred_names = _encode_labels(np.asarray(y))
    n = features.shape[0]
    if n == 0:
        raise ValueError("Cannot build a bandit dataset from zero examples.")
    indices = rng.choice(n, size=T, replace=T > n)
    features = features[indices]
    labels = labels[indices]

    names = class_names if class_names is not None else inferred_names
    K = len(names)
    full_arms = _class_block_arms(features, K=K, d=d, seed=seed + 17)
    raw_norms = np.linalg.norm(full_arms, axis=2)
    max_norm = float(max(np.max(raw_norms), 1e-12))
    full_arms = full_arms / max_norm

    rewards = np.zeros((T, K), dtype=float)
    rewards[np.arange(T), labels] = 1.0

    if p == 1.0:
        masks = np.ones_like(full_arms, dtype=bool)
    else:
        masks = rng.random(size=full_arms.shape) < p
    masked_arms = np.where(masks, full_arms, 0.0)

    metadata: dict[str, Any] = {
        "name": name,
        "seed": seed,
        "p": p,
        "T": T,
        "K": K,
        "d": d,
        "base_feature_dim": int(features.shape[1]),
        "raw_max_arm_norm": max_norm,
        "max_arm_norm": float(np.max(np.linalg.norm(full_arms, axis=2))),
        "mask_rate": float(np.mean(masks)),
    }
    return RealWorldBanditDataset(
        name=name,
        masked_arms=masked_arms,
        masks=masks,
        full_arms=full_arms,
        rewards=rewards,
        labels=labels.astype(np.int64),
        class_names=tuple(str(item) for item in names),
        metadata=metadata,
    )


def _prepare_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
    if X.ndim != 2:
        raise ValueError("X must have shape (n_examples, n_features).")
    Z = np.asarray(X, dtype=float)
    Z = np.nan_to_num(Z, copy=False)
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    scale = np.std(Z, axis=0, keepdims=True)
    Z = Z / np.where(scale > 1e-12, scale, 1.0)
    row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(row_norms > 1e-12, row_norms, 1.0)


def _encode_labels(y: NDArray[Any]) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    values, encoded = np.unique(y, return_inverse=True)
    names = tuple(str(value) for value in values)
    return encoded.astype(np.int64), names


def _class_block_arms(
    features: NDArray[np.float64],
    *,
    K: int,
    d: int,
    seed: int,
) -> NDArray[np.float64]:
    T, q = features.shape
    uncompressed_d = K * q
    if d == uncompressed_d:
        arms = np.zeros((T, K, d), dtype=float)
        for action in range(K):
            start = action * q
            arms[:, action, start : start + q] = features
        return arms

    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(K, q, d)) / np.sqrt(float(d))
    return np.einsum("tq,kqd->tkd", features, projection, optimize=True)


def _load_toy_classification(
    *,
    T: int,
    seed: int,
    n_classes: int = 4,
    q: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, q))
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    y = rng.integers(n_classes, size=T)
    X = centers[y] + 0.35 * rng.normal(size=(T, q))
    class_names = tuple(f"class_{idx}" for idx in range(n_classes))
    return X, y.astype(np.int64), class_names


def _load_digits_sklearn() -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError(
            "digits_sklearn requires the optional real_world dependency scikit-learn."
        ) from exc

    data = load_digits()
    class_names = tuple(str(target) for target in data.target_names)
    return np.asarray(data.data, dtype=float), np.asarray(data.target, dtype=np.int64), class_names


def _load_20newsgroups(
    *,
    seed: int,
    base_feature_dim: int,
    allow_downloads: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    try:
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError(
            "20newsgroups requires the optional real_world dependency scikit-learn."
        ) from exc

    try:
        dataset = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            download_if_missing=allow_downloads,
        )
    except OSError as exc:
        raise DatasetUnavailableError(
            "20newsgroups is not cached locally; rerun with --allow-downloads to fetch it."
        ) from exc

    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, stop_words="english")
    tfidf = vectorizer.fit_transform(dataset.data)
    n_components = min(base_feature_dim, max(1, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    X = normalize(svd.fit_transform(tfidf))
    class_names = tuple(str(name) for name in dataset.target_names)
    return np.asarray(X, dtype=float), np.asarray(dataset.target, dtype=np.int64), class_names


def _load_covertype(
    *,
    allow_downloads: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    try:
        from sklearn.datasets import fetch_covtype
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError(
            "covertype requires the optional real_world dependency scikit-learn."
        ) from exc

    try:
        dataset = fetch_covtype(download_if_missing=allow_downloads)
    except OSError as exc:
        raise DatasetUnavailableError(
            "covertype is not cached locally; rerun with --allow-downloads to fetch it."
        ) from exc

    y = np.asarray(dataset.target, dtype=np.int64) - 1
    class_names = tuple(f"cover_{idx + 1}" for idx in range(int(np.max(y)) + 1))
    return np.asarray(dataset.data, dtype=float), y, class_names
