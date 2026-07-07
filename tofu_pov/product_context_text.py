"""Product-context text classification bandit reductions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.image_bandit import ImageClassificationFullDataset
from tofu_pov.real_world_datasets import DatasetUnavailableError


CACHE_VERSION = "text-product-v1"


DEFAULT_20NEWS4_CATEGORIES = (
    "rec.autos",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.misc",
)


def load_text_product_context_full_dataset(
    source: str,
    *,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    allow_downloads: bool = False,
    cache_dir: Path | str | None = None,
    train_fraction: float = 0.8,
    max_features: int = 20_000,
    min_df: int = 2,
    logistic_C: float = 1.0,
    max_iter: int = 1_000,
    lift_type: str = "dense_random",
    nuisance_mode: str = "raw",
    nuisance_scale: float = 0.0,
    nuisance_spectral_ratio: float = 0.0,
    force_rebuild: bool = False,
) -> ImageClassificationFullDataset:
    """Load or build a product-context text classification bandit dataset.

    A text classifier produces low-dimensional representations `h(x)` and
    class weights `w_k`. The bandit arm for class `k` is `h(x) * w_k`, lifted
    into ambient dimension `d`.
    """

    if source == "mock_text":
        return make_mock_text_product_context_full_dataset(
            latent_dim=latent_dim,
            ambient_dim=ambient_dim,
            seed=seed,
            T=T,
            lift_type=lift_type,
            nuisance_mode=nuisance_mode,
            nuisance_scale=nuisance_scale,
            nuisance_spectral_ratio=nuisance_spectral_ratio,
        )
    if latent_dim <= 0 or ambient_dim < latent_dim or T <= 0:
        raise ValueError("Require 0 < latent_dim <= ambient_dim and T > 0.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if max_features <= 0 or min_df <= 0 or max_iter <= 0:
        raise ValueError("max_features, min_df, and max_iter must be positive.")

    cache_root = Path(cache_dir) if cache_dir is not None else Path("data/text_product_cache")
    cache_path = _cache_path(
        source,
        latent_dim=latent_dim,
        ambient_dim=ambient_dim,
        seed=seed,
        T=T,
        train_fraction=train_fraction,
        max_features=max_features,
        min_df=min_df,
        logistic_C=logistic_C,
        max_iter=max_iter,
        lift_type=lift_type,
        nuisance_mode=nuisance_mode,
        nuisance_scale=nuisance_scale,
        nuisance_spectral_ratio=nuisance_spectral_ratio,
        cache_dir=cache_root,
    )
    if cache_path.exists() and not force_rebuild:
        return _load_cached_dataset(cache_path)

    texts, labels, class_names = _load_text_source(source, allow_downloads=allow_downloads)
    full = make_text_product_context_full_dataset(
        texts,
        labels,
        source=source,
        latent_dim=latent_dim,
        ambient_dim=ambient_dim,
        seed=seed,
        T=T,
        class_names=class_names,
        train_fraction=train_fraction,
        max_features=max_features,
        min_df=min_df,
        logistic_C=logistic_C,
        max_iter=max_iter,
        lift_type=lift_type,
        nuisance_mode=nuisance_mode,
        nuisance_scale=nuisance_scale,
        nuisance_spectral_ratio=nuisance_spectral_ratio,
        cache_path=cache_path,
    )
    _save_cached_dataset(cache_path, full)
    return full


def make_text_product_context_full_dataset(
    texts: list[str],
    labels: NDArray[np.int64],
    *,
    source: str,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    class_names: tuple[str, ...],
    train_fraction: float = 0.8,
    max_features: int = 20_000,
    min_df: int = 2,
    logistic_C: float = 1.0,
    max_iter: int = 1_000,
    lift_type: str = "dense_random",
    nuisance_mode: str = "raw",
    nuisance_scale: float = 0.0,
    nuisance_spectral_ratio: float = 0.0,
    cache_path: Path | None = None,
) -> ImageClassificationFullDataset:
    if latent_dim <= 0 or ambient_dim < latent_dim or T <= 0:
        raise ValueError("Require 0 < latent_dim <= ambient_dim and T > 0.")
    if len(texts) != labels.shape[0]:
        raise ValueError("texts and labels must have the same length.")
    if nuisance_scale < 0.0:
        raise ValueError("nuisance_scale must be nonnegative.")
    if nuisance_spectral_ratio < 0.0:
        raise ValueError("nuisance_spectral_ratio must be nonnegative.")

    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, normalize
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError(
            "Text product-context experiments require scikit-learn."
        ) from exc

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    train_idx, heldout_idx = _stratified_split(labels, train_fraction=train_fraction, rng=rng)
    if heldout_idx.size == 0:
        raise ValueError("Need at least one held-out example to build bandit rounds.")

    train_texts = [texts[int(idx)] for idx in train_idx]
    heldout_texts = [texts[int(idx)] for idx in heldout_idx]
    train_y = labels[train_idx]
    heldout_y = labels[heldout_idx]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    heldout_tfidf = vectorizer.transform(heldout_texts)
    n_components = min(latent_dim, train_tfidf.shape[1] - 1, train_idx.size - 1)
    if n_components <= 0:
        raise ValueError("latent_dim is too large for the available text data.")

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train_h = svd.fit_transform(train_tfidf)
    heldout_h = svd.transform(heldout_tfidf)
    scaler = StandardScaler()
    train_h = scaler.fit_transform(train_h)
    heldout_h = scaler.transform(heldout_h)
    train_h = normalize(train_h)
    heldout_h = normalize(heldout_h)

    clf = LogisticRegression(
        C=logistic_C,
        fit_intercept=False,
        max_iter=max_iter,
        solver="lbfgs",
    )
    clf.fit(train_h, train_y)
    weights = np.zeros((len(class_names), n_components), dtype=float)
    for class_position, class_id in enumerate(clf.classes_):
        weights[int(class_id)] = clf.coef_[class_position]

    sampled = rng.choice(heldout_h.shape[0], size=T, replace=T > heldout_h.shape[0])
    H = heldout_h[sampled]
    sampled_labels = heldout_y[sampled]
    low_dim_arms = H[:, None, :] * weights[None, :, :]
    lift_basis = _make_lift_basis(ambient_dim, n_components, rng, lift_type=lift_type)
    signal_arms = np.einsum("tkm,dm->tkd", low_dim_arms, lift_basis)
    full_arms, nuisance_metadata = _add_orthogonal_nuisance(
        signal_arms,
        signal_basis=lift_basis,
        nuisance_mode=nuisance_mode,
        nuisance_scale=nuisance_scale,
        nuisance_spectral_ratio=nuisance_spectral_ratio,
        rng=rng,
    )
    full_arms = _normalize_arms(full_arms)
    rewards = np.zeros((T, len(class_names)), dtype=float)
    rewards[np.arange(T), sampled_labels] = 1.0

    train_pred = clf.predict(train_h)
    heldout_pred = clf.predict(heldout_h)
    metadata: dict[str, Any] = {
        "source": source,
        "architecture": "text_product_context",
        "cache_version": CACHE_VERSION,
        "seed": seed,
        "T": T,
        "K": len(class_names),
        "d": ambient_dim,
        "latent_dim": n_components,
        "ambient_dim": ambient_dim,
        "train_fraction": train_fraction,
        "train_examples": int(train_idx.size),
        "heldout_examples": int(heldout_idx.size),
        "max_features": max_features,
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "min_df": min_df,
        "logistic_C": logistic_C,
        "max_iter": max_iter,
        "lift_type": lift_type,
        "nuisance_mode": nuisance_mode,
        "nuisance_scale": nuisance_scale,
        "nuisance_spectral_ratio": nuisance_spectral_ratio,
        **nuisance_metadata,
        "train_accuracy": float(np.mean(train_pred == train_y)),
        "heldout_accuracy": float(np.mean(heldout_pred == heldout_y)),
        "explained_variance": float(np.sum(svd.explained_variance_ratio_)),
        "cache_path": "" if cache_path is None else str(cache_path),
    }
    return ImageClassificationFullDataset(
        name=f"{source}_text_product_m{n_components}_d{ambient_dim}",
        full_arms=full_arms,
        rewards=rewards,
        labels=sampled_labels.astype(np.int64),
        class_names=tuple(class_names),
        metadata=metadata,
    )


def make_mock_text_product_context_full_dataset(
    *,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    K: int = 4,
    lift_type: str = "dense_random",
    nuisance_mode: str = "raw",
    nuisance_scale: float = 0.0,
    nuisance_spectral_ratio: float = 0.0,
) -> ImageClassificationFullDataset:
    """Create a deterministic product-context text-like fixture for tests."""

    if latent_dim <= 0 or ambient_dim < latent_dim or T <= 0 or K <= 1:
        raise ValueError("Require 0 < latent_dim <= ambient_dim, T > 0, and K > 1.")
    if nuisance_scale < 0.0:
        raise ValueError("nuisance_scale must be nonnegative.")
    if nuisance_spectral_ratio < 0.0:
        raise ValueError("nuisance_spectral_ratio must be nonnegative.")
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(K, latent_dim))
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    labels = rng.integers(0, K, size=T, endpoint=False)
    H = weights[labels] + 0.25 * rng.normal(size=(T, latent_dim))
    H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    low_dim_arms = H[:, None, :] * weights[None, :, :]
    lift_basis = _make_lift_basis(ambient_dim, latent_dim, rng, lift_type=lift_type)
    signal_arms = np.einsum("tkm,dm->tkd", low_dim_arms, lift_basis)
    full_arms, nuisance_metadata = _add_orthogonal_nuisance(
        signal_arms,
        signal_basis=lift_basis,
        nuisance_mode=nuisance_mode,
        nuisance_scale=nuisance_scale,
        nuisance_spectral_ratio=nuisance_spectral_ratio,
        rng=rng,
    )
    full_arms = _normalize_arms(full_arms)
    rewards = np.zeros((T, K), dtype=float)
    rewards[np.arange(T), labels] = 1.0
    metadata: dict[str, Any] = {
        "source": "mock_text",
        "architecture": "mock_text_product_context",
        "cache_version": CACHE_VERSION,
        "seed": seed,
        "T": T,
        "K": K,
        "d": ambient_dim,
        "latent_dim": latent_dim,
        "ambient_dim": ambient_dim,
        "lift_type": lift_type,
        "nuisance_mode": nuisance_mode,
        "nuisance_scale": nuisance_scale,
        "nuisance_spectral_ratio": nuisance_spectral_ratio,
        **nuisance_metadata,
        "train_accuracy": 1.0,
        "heldout_accuracy": 1.0,
        "train_examples": 0,
        "heldout_examples": T,
        "vocabulary_size": 0,
        "explained_variance": 1.0,
    }
    return ImageClassificationFullDataset(
        name=f"mock_text_product_m{latent_dim}_d{ambient_dim}",
        full_arms=full_arms,
        rewards=rewards,
        labels=labels.astype(np.int64),
        class_names=tuple(str(idx) for idx in range(K)),
        metadata=metadata,
    )


def _load_text_source(
    source: str,
    *,
    allow_downloads: bool,
) -> tuple[list[str], NDArray[np.int64], tuple[str, ...]]:
    if source not in {"20newsgroups4", "20newsgroups20"}:
        raise ValueError(f"Unsupported text product-context source: {source}")
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError("20newsgroups requires scikit-learn.") from exc

    categories = DEFAULT_20NEWS4_CATEGORIES if source == "20newsgroups4" else None
    try:
        import inspect

        kwargs: dict[str, Any] = {
            "subset": "all",
            "categories": categories,
            "remove": ("headers", "footers", "quotes"),
        }
        if "download_if_missing" in inspect.signature(fetch_20newsgroups).parameters:
            kwargs["download_if_missing"] = allow_downloads
        data = fetch_20newsgroups(**kwargs)
    except OSError as exc:
        raise DatasetUnavailableError(
            "20newsgroups is not cached locally; rerun with --allow-downloads."
        ) from exc

    labels, class_names = _encode_labels(np.asarray(data.target))
    target_names = tuple(str(data.target_names[idx]) for idx in range(len(data.target_names)))
    if len(target_names) == len(class_names):
        class_names = target_names
    return list(data.data), labels, class_names


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


def _grouped_lift_basis(ambient_dim: int, latent_dim: int) -> NDArray[np.float64]:
    basis = np.zeros((ambient_dim, latent_dim), dtype=float)
    for latent_idx, group in enumerate(np.array_split(np.arange(ambient_dim), latent_dim)):
        if group.size == 0:
            raise ValueError("ambient_dim must be at least latent_dim.")
        basis[group, latent_idx] = 1.0 / np.sqrt(group.size)
    return basis


def _make_lift_basis(
    ambient_dim: int,
    latent_dim: int,
    rng: np.random.Generator,
    *,
    lift_type: str,
) -> NDArray[np.float64]:
    if lift_type == "dense_random":
        return _random_lift_basis(ambient_dim, latent_dim, rng)
    if lift_type == "grouped":
        return _grouped_lift_basis(ambient_dim, latent_dim)
    raise ValueError("lift_type must be 'dense_random' or 'grouped'.")


def _orthogonal_complement_basis(
    signal_basis: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    ambient_dim, latent_dim = signal_basis.shape
    nuisance_dim = ambient_dim - latent_dim
    if nuisance_dim <= 0:
        return np.zeros((ambient_dim, 0), dtype=float)
    random_part = rng.normal(size=(ambient_dim, nuisance_dim))
    random_part -= signal_basis @ (signal_basis.T @ random_part)
    q, _ = np.linalg.qr(random_part)
    return q[:, :nuisance_dim]


def _add_orthogonal_nuisance(
    signal_arms: NDArray[np.float64],
    *,
    signal_basis: NDArray[np.float64],
    nuisance_mode: str,
    nuisance_scale: float,
    nuisance_spectral_ratio: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], dict[str, float | int | str]]:
    if nuisance_mode not in {"raw", "spectral_tail"}:
        raise ValueError("nuisance_mode must be 'raw' or 'spectral_tail'.")
    nuisance_active = (nuisance_mode == "raw" and nuisance_scale > 0.0) or (
        nuisance_mode == "spectral_tail" and nuisance_spectral_ratio > 0.0
    )
    metadata: dict[str, float | int | str] = {
        "nuisance_dim": 0,
        "signal_lambda_m": 0.0,
        "nuisance_top_eigenvalue": 0.0,
        "realized_nuisance_spectral_ratio": 0.0,
    }
    if not nuisance_active:
        return signal_arms, metadata
    nuisance_basis = _orthogonal_complement_basis(signal_basis, rng)
    nuisance_dim = nuisance_basis.shape[1]
    if nuisance_dim == 0:
        return signal_arms, metadata
    coefficients = rng.normal(size=(*signal_arms.shape[:2], nuisance_dim))
    signal_coefficients = signal_arms.reshape(-1, signal_arms.shape[-1]) @ signal_basis
    signal_eigenvalues = _covariance_eigenvalues_desc(signal_coefficients)
    signal_lambda_m = float(signal_eigenvalues[-1]) if signal_eigenvalues.size else 0.0
    metadata["signal_lambda_m"] = signal_lambda_m
    if nuisance_mode == "raw":
        coefficients /= np.sqrt(nuisance_dim)
        scale = nuisance_scale
        raw_top = _top_covariance_eigenvalue(coefficients.reshape(-1, nuisance_dim))
        realized_top = scale * scale * raw_top
    else:
        raw_top = _top_covariance_eigenvalue(coefficients.reshape(-1, nuisance_dim))
        target_top = nuisance_spectral_ratio * signal_lambda_m
        scale = np.sqrt(target_top / max(raw_top, 1e-12)) if target_top > 0.0 else 0.0
        realized_top = target_top
    nuisance_arms = np.einsum("tkr,dr->tkd", coefficients, nuisance_basis)
    metadata["nuisance_dim"] = nuisance_dim
    metadata["nuisance_top_eigenvalue"] = float(realized_top)
    metadata["realized_nuisance_spectral_ratio"] = float(
        realized_top / max(float(metadata["signal_lambda_m"]), 1e-12)
    )
    return signal_arms + scale * nuisance_arms, metadata


def _covariance_eigenvalues_desc(coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    if coefficients.size == 0:
        return np.array([], dtype=float)
    centered = coefficients - np.mean(coefficients, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(centered.shape[0], 1)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    return np.maximum(eigenvalues[::-1], 0.0)


def _top_covariance_eigenvalue(coefficients: NDArray[np.float64]) -> float:
    eigenvalues = _covariance_eigenvalues_desc(coefficients)
    return float(eigenvalues[0]) if eigenvalues.size else 0.0


def _cache_path(
    source: str,
    *,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    train_fraction: float,
    max_features: int,
    min_df: int,
    logistic_C: float,
    max_iter: int,
    lift_type: str,
    nuisance_mode: str,
    nuisance_scale: float,
    nuisance_spectral_ratio: float,
    cache_dir: Path,
) -> Path:
    payload = {
        "version": CACHE_VERSION,
        "source": source,
        "latent_dim": latent_dim,
        "ambient_dim": ambient_dim,
        "seed": seed,
        "T": T,
        "train_fraction": train_fraction,
        "max_features": max_features,
        "min_df": min_df,
        "logistic_C": logistic_C,
        "max_iter": max_iter,
        "lift_type": lift_type,
        "nuisance_mode": nuisance_mode,
        "nuisance_scale": nuisance_scale,
        "nuisance_spectral_ratio": nuisance_spectral_ratio,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    stem = f"{source}_text_product_m{latent_dim}_d{ambient_dim}_T{T}_seed{seed}_{digest}"
    return cache_dir / f"{stem}.npz"


def _save_cached_dataset(path: Path, data: ImageClassificationFullDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array(data.name),
        full_arms=data.full_arms,
        rewards=data.rewards,
        labels=data.labels,
        class_names=np.asarray(data.class_names),
        metadata=np.array(json.dumps(data.metadata, sort_keys=True)),
    )


def _load_cached_dataset(path: Path) -> ImageClassificationFullDataset:
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
        return ImageClassificationFullDataset(
            name=str(arrays["name"].item()),
            full_arms=np.asarray(arrays["full_arms"], dtype=float),
            rewards=np.asarray(arrays["rewards"], dtype=float),
            labels=np.asarray(arrays["labels"], dtype=np.int64),
            class_names=tuple(str(item) for item in arrays["class_names"].tolist()),
            metadata=metadata,
        )
