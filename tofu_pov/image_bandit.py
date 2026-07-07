"""Image-classification-to-bandit reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.real_world import ArrayBanditEnv
from tofu_pov.real_world_datasets import DatasetUnavailableError


@dataclass(frozen=True)
class ImageClassificationFullDataset:
    """Unmasked image-classification decision sets."""

    name: str
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


@dataclass(frozen=True)
class ImageClassificationBanditDataset:
    """Masked image-classification decision sets."""

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


def load_image_classification_full_dataset(
    source: str,
    *,
    representation_dim: int,
    seed: int,
    T: int,
    ambient_dim: int | None = None,
    allow_downloads: bool = False,
    train_fraction: float = 0.8,
    max_train_examples: int = 12_000,
) -> ImageClassificationFullDataset:
    """Train a lightweight image classifier and emit class-action features."""

    X, y, class_names = _load_image_source(source, allow_downloads=allow_downloads)
    return make_image_classification_full_dataset(
        X,
        y,
        source=source,
        representation_dim=representation_dim,
        seed=seed,
        T=T,
        ambient_dim=ambient_dim,
        class_names=class_names,
        train_fraction=train_fraction,
        max_train_examples=max_train_examples,
    )


def make_image_classification_full_dataset(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    source: str,
    representation_dim: int,
    seed: int,
    T: int,
    ambient_dim: int | None = None,
    class_names: tuple[str, ...] | None = None,
    train_fraction: float = 0.8,
    max_train_examples: int = 12_000,
) -> ImageClassificationFullDataset:
    if representation_dim <= 0 or T <= 0:
        raise ValueError("representation_dim and T must be positive.")
    if ambient_dim is not None and ambient_dim < representation_dim:
        raise ValueError("ambient_dim must be at least representation_dim.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")

    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError(
            "Image classification experiments require the optional scikit-learn dependency."
        ) from exc

    rng = np.random.default_rng(seed)
    X = _prepare_image_pixels(np.asarray(X, dtype=float))
    labels, inferred_names = _encode_labels(np.asarray(y))
    names = class_names if class_names is not None else inferred_names
    train_idx, heldout_idx = _stratified_split(labels, train_fraction=train_fraction, rng=rng)
    if heldout_idx.size == 0:
        raise ValueError("Need at least one held-out image to build bandit rounds.")
    if train_idx.size > max_train_examples:
        train_idx = rng.choice(train_idx, size=max_train_examples, replace=False)

    n_components = min(representation_dim, X.shape[1] - 1, train_idx.size - 1)
    if n_components <= 0:
        raise ValueError("representation_dim is too large for the available training data.")

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train_h = svd.fit_transform(X[train_idx])
    scaler = StandardScaler()
    train_h = scaler.fit_transform(train_h)
    clf = LogisticRegression(
        C=1.0,
        fit_intercept=False,
        max_iter=500,
        solver="lbfgs",
    )
    clf.fit(train_h, labels[train_idx])
    heldout_h = scaler.transform(svd.transform(X[heldout_idx]))
    heldout_labels = labels[heldout_idx]
    sampled = rng.choice(heldout_h.shape[0], size=T, replace=T > heldout_h.shape[0])
    H = heldout_h[sampled]
    sampled_labels = heldout_labels[sampled]

    weights = np.zeros((len(names), n_components), dtype=float)
    for class_position, class_id in enumerate(clf.classes_):
        weights[int(class_id)] = clf.coef_[class_position]
    full_arms = H[:, None, :] * weights[None, :, :]
    lift_rank = n_components
    lift_basis = None
    if ambient_dim is not None:
        lift_basis = _random_lift_basis(ambient_dim, n_components, rng)
        full_arms = np.einsum("tkm,dm->tkd", full_arms, lift_basis)
    full_arms = _normalize_arms(full_arms)
    rewards = np.zeros((T, len(names)), dtype=float)
    rewards[np.arange(T), sampled_labels] = 1.0

    train_pred = clf.predict(train_h)
    heldout_pred = clf.predict(heldout_h)
    metadata: dict[str, Any] = {
        "source": source,
        "seed": seed,
        "T": T,
        "K": len(names),
        "d": int(full_arms.shape[2]),
        "requested_representation_dim": representation_dim,
        "latent_dim": lift_rank,
        "ambient_dim": int(full_arms.shape[2]),
        "low_rank_lift": ambient_dim is not None,
        "train_examples": int(train_idx.size),
        "heldout_examples": int(heldout_idx.size),
        "train_accuracy": float(np.mean(train_pred == labels[train_idx])),
        "heldout_accuracy": float(np.mean(heldout_pred == heldout_labels)),
        "explained_variance": float(np.sum(svd.explained_variance_ratio_)),
    }
    return ImageClassificationFullDataset(
        name=(
            f"{source}_image_slb_m{lift_rank}_d{full_arms.shape[2]}"
            if lift_basis is not None
            else f"{source}_image_slb_d{n_components}"
        ),
        full_arms=full_arms,
        rewards=rewards,
        labels=sampled_labels.astype(np.int64),
        class_names=tuple(str(item) for item in names),
        metadata=metadata,
    )


def mask_image_classification_dataset(
    full_data: ImageClassificationFullDataset,
    *,
    p: float,
    seed: int,
) -> ImageClassificationBanditDataset:
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1].")
    rng = np.random.default_rng(seed)
    full_arms = np.asarray(full_data.full_arms, dtype=float)
    masks = np.ones_like(full_arms, dtype=bool) if p == 1.0 else rng.random(size=full_arms.shape) < p
    masked_arms = np.where(masks, full_arms, 0.0)
    metadata = dict(full_data.metadata)
    metadata.update({"p": p, "mask_rate": float(np.mean(masks))})
    return ImageClassificationBanditDataset(
        name=full_data.name,
        masked_arms=masked_arms,
        masks=masks,
        full_arms=full_arms,
        rewards=full_data.rewards,
        labels=full_data.labels,
        class_names=full_data.class_names,
        metadata=metadata,
    )


def _load_image_source(
    source: str,
    *,
    allow_downloads: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    if source == "digits_sklearn":
        try:
            from sklearn.datasets import load_digits
        except ModuleNotFoundError as exc:
            raise DatasetUnavailableError("digits_sklearn requires scikit-learn.") from exc
        data = load_digits()
        return (
            np.asarray(data.data, dtype=float),
            np.asarray(data.target, dtype=np.int64),
            tuple(str(item) for item in data.target_names),
        )
    if source == "mnist_openml":
        try:
            from sklearn.datasets import fetch_openml
        except ModuleNotFoundError as exc:
            raise DatasetUnavailableError("mnist_openml requires scikit-learn.") from exc
        if not allow_downloads:
            raise DatasetUnavailableError(
                "MNIST OpenML access is disabled; rerun with --allow-downloads."
            )
        try:
            import inspect

            kwargs: dict[str, Any] = {"version": 1, "as_frame": False}
            if "download_if_missing" in inspect.signature(fetch_openml).parameters:
                kwargs["download_if_missing"] = allow_downloads
            data = fetch_openml("mnist_784", **kwargs)
        except OSError as exc:
            raise DatasetUnavailableError(
                "MNIST OpenML is not cached locally; rerun with --allow-downloads."
            ) from exc
        return (
            np.asarray(data.data, dtype=float),
            np.asarray(data.target, dtype=np.int64),
            tuple(str(idx) for idx in range(10)),
        )
    raise ValueError(f"Unknown image source: {source}")


def _prepare_image_pixels(X: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.nan_to_num(X, copy=True)
    max_value = float(np.max(values)) if values.size else 1.0
    if max_value > 1.0:
        values = values / max_value
    return values


def _encode_labels(y: NDArray[Any]) -> tuple[NDArray[np.int64], tuple[str, ...]]:
    values, encoded = np.unique(y, return_inverse=True)
    names = tuple(str(value) for value in values)
    return encoded.astype(np.int64), names


def _stratified_split(
    labels: NDArray[np.int64],
    *,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    train_parts: list[NDArray[np.int64]] = []
    heldout_parts: list[NDArray[np.int64]] = []
    for class_id in np.unique(labels):
        class_idx = np.flatnonzero(labels == class_id)
        rng.shuffle(class_idx)
        if class_idx.size == 1:
            train_parts.append(class_idx)
            continue
        class_train_size = int(round(train_fraction * class_idx.size))
        class_train_size = min(max(class_train_size, 1), class_idx.size - 1)
        train_parts.append(class_idx[:class_train_size])
        heldout_parts.append(class_idx[class_train_size:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    heldout_idx = np.concatenate(heldout_parts) if heldout_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(heldout_idx)
    return train_idx.astype(np.int64), heldout_idx.astype(np.int64)


def _normalize_arms(arms: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(arms, axis=2)
    return arms / max(float(np.max(norms)), 1e-12)


def _random_lift_basis(
    ambient_dim: int,
    latent_dim: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    matrix = rng.normal(size=(ambient_dim, latent_dim))
    q, _ = np.linalg.qr(matrix)
    return q[:, :latent_dim]
