"""Low-rank CNN image-classification-to-bandit reductions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.image_bandit import (
    ImageClassificationFullDataset,
    _encode_labels,
    _normalize_arms,
    _prepare_image_pixels,
    _random_lift_basis,
    _stratified_split,
)
from tofu_pov.real_world_datasets import DatasetUnavailableError


CACHE_VERSION = "cnn-lowrank-v1"


def _architecture_for_source(source: str) -> str:
    if source == "mnist_openml":
        return "shared_theta"
    if source == "mnist_openml_product":
        return "product_context"
    raise ValueError(f"Unsupported CNN image source: {source}")


def load_cnn_image_classification_full_dataset(
    source: str,
    *,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    allow_downloads: bool = False,
    cache_dir: Path | str | None = None,
    train_fraction: float = 0.8,
    max_train_examples: int = 12_000,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    force_retrain: bool = False,
) -> ImageClassificationFullDataset:
    """Train or load low-rank CNN class contexts and lift them to ambient arms."""

    if source == "mock_cnn":
        return make_mock_cnn_lowrank_full_dataset(
            latent_dim=latent_dim,
            ambient_dim=ambient_dim,
            seed=seed,
            T=T,
        )
    if latent_dim <= 0 or ambient_dim < latent_dim or T <= 0:
        raise ValueError("Require 0 < latent_dim <= ambient_dim and T > 0.")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")

    cache_path = _cache_path(
        source,
        latent_dim=latent_dim,
        ambient_dim=ambient_dim,
        seed=seed,
        T=T,
        train_fraction=train_fraction,
        max_train_examples=max_train_examples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        cache_dir=Path(cache_dir) if cache_dir is not None else Path("data/cnn_image_cache"),
    )
    if cache_path.exists() and not force_retrain:
        return _load_cached_dataset(cache_path)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError(
            "CNN image experiments require PyTorch. Install the optional cnn dependencies."
        ) from exc

    X, y, class_names = _load_image_source_cached(source, allow_downloads=allow_downloads)
    X = _prepare_image_pixels(np.asarray(X, dtype=float))
    labels, inferred_names = _encode_labels(np.asarray(y))
    names = class_names if class_names is not None else inferred_names
    K = len(names)

    rng = np.random.default_rng(seed)
    train_idx, heldout_idx = _stratified_split(labels, train_fraction=train_fraction, rng=rng)
    if heldout_idx.size == 0:
        raise ValueError("Need at least one held-out image to build bandit rounds.")
    if train_idx.size > max_train_examples:
        train_idx = rng.choice(train_idx, size=max_train_examples, replace=False)

    side = _square_image_side(X.shape[1])
    architecture = _architecture_for_source(source)
    torch.manual_seed(seed)
    if architecture == "product_context":
        model = _ProductContextCNN(
            input_side=side,
            n_classes=K,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nn_module=nn,
            torch_module=torch,
            seed=seed,
        )
    else:
        model = _LowRankContextCNN(
            input_side=side,
            n_classes=K,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            nn_module=nn,
            torch_module=torch,
            seed=seed,
        )
    train_x = torch.as_tensor(X[train_idx].reshape(-1, 1, side, side), dtype=torch.float32)
    train_y = torch.as_tensor(labels[train_idx], dtype=torch.long)
    dataset = TensorDataset(train_x, train_y)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

    heldout_x = torch.as_tensor(X[heldout_idx].reshape(-1, 1, side, side), dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        train_logits = _batched_logits(model, train_x, batch_size)
        heldout_logits = _batched_logits(model, heldout_x, batch_size)
        heldout_contexts = _batched_contexts(model, heldout_x, batch_size).cpu().numpy()

    train_pred = train_logits.argmax(dim=1).cpu().numpy()
    heldout_pred = heldout_logits.argmax(dim=1).cpu().numpy()
    heldout_labels = labels[heldout_idx]
    sampled = rng.choice(heldout_contexts.shape[0], size=T, replace=T > heldout_contexts.shape[0])
    latent_arms = heldout_contexts[sampled]
    sampled_labels = heldout_labels[sampled]
    lift_basis = _random_lift_basis(ambient_dim, latent_dim, rng)
    full_arms = np.einsum("tkm,dm->tkd", latent_arms, lift_basis)
    full_arms = _normalize_arms(full_arms)
    rewards = np.zeros((T, K), dtype=float)
    rewards[np.arange(T), sampled_labels] = 1.0

    metadata: dict[str, Any] = {
        "source": source,
        "architecture": (
            "product_context_cnn"
            if architecture == "product_context"
            else "low_rank_context_cnn"
        ),
        "cache_version": CACHE_VERSION,
        "seed": seed,
        "T": T,
        "K": K,
        "d": ambient_dim,
        "latent_dim": latent_dim,
        "ambient_dim": ambient_dim,
        "train_fraction": train_fraction,
        "train_examples": int(train_idx.size),
        "heldout_examples": int(heldout_idx.size),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "train_accuracy": float(np.mean(train_pred == labels[train_idx])),
        "heldout_accuracy": float(np.mean(heldout_pred == heldout_labels)),
        "cache_path": str(cache_path),
    }
    result = ImageClassificationFullDataset(
        name=f"{source}_cnn_lowrank_m{latent_dim}_d{ambient_dim}",
        full_arms=full_arms,
        rewards=rewards,
        labels=sampled_labels.astype(np.int64),
        class_names=tuple(str(item) for item in names),
        metadata=metadata,
    )
    _save_cached_dataset(cache_path, result)
    return result


def make_mock_cnn_lowrank_full_dataset(
    *,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    K: int = 10,
) -> ImageClassificationFullDataset:
    """Create deterministic low-rank class contexts for quick script tests."""

    if latent_dim <= 0 or ambient_dim < latent_dim or T <= 0 or K <= 1:
        raise ValueError("Require 0 < latent_dim <= ambient_dim, T > 0, and K > 1.")
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=latent_dim)
    theta = theta / max(float(np.linalg.norm(theta)), 1e-12)
    class_offsets = rng.normal(scale=0.35, size=(K, latent_dim))
    labels = rng.integers(0, K, size=T, endpoint=False)
    latent_arms = np.empty((T, K, latent_dim), dtype=float)
    for t, label in enumerate(labels):
        context_noise = rng.normal(scale=0.20, size=(K, latent_dim))
        latent_arms[t] = class_offsets + context_noise
        latent_arms[t, label] += 1.75 * theta
    lift_basis = _random_lift_basis(ambient_dim, latent_dim, rng)
    full_arms = np.einsum("tkm,dm->tkd", latent_arms, lift_basis)
    full_arms = _normalize_arms(full_arms)
    rewards = np.zeros((T, K), dtype=float)
    rewards[np.arange(T), labels] = 1.0
    metadata: dict[str, Any] = {
        "source": "mock_cnn",
        "architecture": "mock_low_rank_context_cnn",
        "cache_version": CACHE_VERSION,
        "seed": seed,
        "T": T,
        "K": K,
        "d": ambient_dim,
        "latent_dim": latent_dim,
        "ambient_dim": ambient_dim,
        "train_accuracy": 1.0,
        "heldout_accuracy": 1.0,
        "train_examples": 0,
        "heldout_examples": T,
    }
    return ImageClassificationFullDataset(
        name=f"mock_cnn_lowrank_m{latent_dim}_d{ambient_dim}",
        full_arms=full_arms,
        rewards=rewards,
        labels=labels.astype(np.int64),
        class_names=tuple(str(idx) for idx in range(K)),
        metadata=metadata,
    )


class _LowRankContextCNN:
    def __new__(
        cls,
        *,
        input_side: int,
        n_classes: int,
        latent_dim: int,
        hidden_dim: int,
        nn_module,
        torch_module,
        seed: int,
    ):
        class Module(nn_module.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = nn_module.Sequential(
                    nn_module.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn_module.ReLU(),
                    nn_module.MaxPool2d(2),
                    nn_module.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn_module.ReLU(),
                    nn_module.AdaptiveAvgPool2d((4, 4)),
                    nn_module.Flatten(),
                    nn_module.Linear(32 * 4 * 4, hidden_dim),
                    nn_module.ReLU(),
                )
                self.context_head = nn_module.Linear(hidden_dim, n_classes * latent_dim)
                theta = np.random.default_rng(seed + 17).normal(size=latent_dim)
                theta = theta / max(float(np.linalg.norm(theta)), 1e-12)
                self.theta = nn_module.Parameter(torch_module.as_tensor(theta, dtype=torch_module.float32))

            def contexts(self, x):
                features = self.trunk(x)
                return self.context_head(features).reshape(-1, n_classes, latent_dim)

            def forward(self, x):
                contexts = self.contexts(x)
                return (contexts * self.theta.reshape(1, 1, -1)).sum(dim=2)

        del input_side
        return Module()


class _ProductContextCNN:
    """Standard CNN classifier exposing per-class arm features X_k(x) = h(x) * w_k.

    The penultimate features h(x) live in R^m and the linear classifier head has
    per-class weights w_k in R^m (no bias). The standard classifier score
    w_k^T h(x) equals <X_k(x), 1_m>, so the trained classifier is recovered by
    the bandit's optimal linear policy theta = 1_m. Unlike the shared-theta
    architecture used by `_LowRankContextCNN`, this places the per-class signal
    in the arm features themselves (which vary across k via w_k) rather than in
    a single shared score direction, so the bandit reward signal can use all m
    feature coordinates non-trivially.
    """

    def __new__(
        cls,
        *,
        input_side: int,
        n_classes: int,
        latent_dim: int,
        hidden_dim: int,
        nn_module,
        torch_module,
        seed: int,
    ):
        K = n_classes
        m = latent_dim

        class Module(nn_module.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = nn_module.Sequential(
                    nn_module.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn_module.ReLU(),
                    nn_module.MaxPool2d(2),
                    nn_module.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn_module.ReLU(),
                    nn_module.AdaptiveAvgPool2d((4, 4)),
                    nn_module.Flatten(),
                    nn_module.Linear(32 * 4 * 4, hidden_dim),
                    nn_module.ReLU(),
                    nn_module.Linear(hidden_dim, m),
                )
                self.head = nn_module.Linear(m, K, bias=False)

            def features(self, x):
                return self.trunk(x)

            def contexts(self, x):
                h = self.features(x)
                W = self.head.weight
                return h.unsqueeze(1) * W.unsqueeze(0)

            def forward(self, x):
                return self.head(self.features(x))

        del seed
        del input_side
        return Module()


def _batched_logits(model, x, batch_size: int):
    import torch

    chunks = []
    for start in range(0, x.shape[0], batch_size):
        chunks.append(model(x[start : start + batch_size]))
    return torch.cat(chunks, dim=0)


def _batched_contexts(model, x, batch_size: int):
    import torch

    chunks = []
    for start in range(0, x.shape[0], batch_size):
        chunks.append(model.contexts(x[start : start + batch_size]))
    return torch.cat(chunks, dim=0)


def _load_image_source_cached(
    source: str,
    *,
    allow_downloads: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    if source not in ("mnist_openml", "mnist_openml_product"):
        raise ValueError(f"Unsupported CNN image source: {source}")
    try:
        from sklearn.datasets import fetch_openml
    except ModuleNotFoundError as exc:
        raise DatasetUnavailableError("mnist_openml requires scikit-learn.") from exc
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


def _square_image_side(n_features: int) -> int:
    side = int(round(np.sqrt(n_features)))
    if side * side != n_features:
        raise ValueError(f"Expected square image vectors, got {n_features} features.")
    return side


def _cache_path(
    source: str,
    *,
    latent_dim: int,
    ambient_dim: int,
    seed: int,
    T: int,
    train_fraction: float,
    max_train_examples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
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
        "max_train_examples": max_train_examples,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    stem = f"{source}_cnn_m{latent_dim}_d{ambient_dim}_T{T}_seed{seed}_{digest}"
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
