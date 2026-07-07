"""MovieLens recommendation-bandit dataset builders."""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tofu_pov.real_world import ArrayBanditEnv
from tofu_pov.real_world_datasets import DatasetUnavailableError


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


MovieLensFeatureMode = Literal["mf", "mf_product", "side_info", "hybrid"]
MovieLensSlateMode = Literal["random", "contrastive"]


@dataclass(frozen=True)
class MovieLensRawData:
    """Parsed MovieLens 100K tables."""

    ratings: NDArray[np.float64]
    user_ids: NDArray[np.int64]
    movie_ids: NDArray[np.int64]
    user_features: NDArray[np.float64]
    movie_features: NDArray[np.float64]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MovieLensBanditDataset:
    """Array-backed MovieLens recommendation bandit."""

    name: str
    masked_arms: NDArray[np.float64]
    masks: NDArray[np.bool_]
    full_arms: NDArray[np.float64]
    rewards: NDArray[np.float64]
    user_ids: NDArray[np.int64]
    movie_ids: NDArray[np.int64]
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


def load_movielens_bandit_dataset(
    *,
    p: float,
    seed: int,
    T: int,
    K: int,
    d: int,
    rank: int,
    feature_mode: MovieLensFeatureMode = "hybrid",
    slate_mode: MovieLensSlateMode = "random",
    data_dir: Path | str = Path("data"),
    allow_downloads: bool = False,
    train_fraction: float = 0.8,
    synthetic: bool = False,
) -> MovieLensBanditDataset:
    """Load MovieLens and convert held-out ratings into a finite-arm bandit."""

    if synthetic:
        raw = make_synthetic_movielens_raw(seed=seed)
    else:
        raw = load_movielens_100k(data_dir=Path(data_dir), allow_downloads=allow_downloads)
    return make_movielens_bandit_dataset(
        raw,
        p=p,
        seed=seed,
        T=T,
        K=K,
        d=d,
        rank=rank,
        feature_mode=feature_mode,
        slate_mode=slate_mode,
        train_fraction=train_fraction,
    )


def make_movielens_bandit_dataset(
    raw: MovieLensRawData,
    *,
    p: float,
    seed: int,
    T: int,
    K: int,
    d: int,
    rank: int,
    feature_mode: MovieLensFeatureMode = "hybrid",
    slate_mode: MovieLensSlateMode = "random",
    train_fraction: float = 0.8,
) -> MovieLensBanditDataset:
    """Build slates from held-out MovieLens ratings."""

    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1].")
    if T <= 0 or K <= 1 or d <= 0 or rank <= 0:
        raise ValueError("T, K, d, and rank must be positive, with K > 1.")
    if rank > d:
        raise ValueError("rank must be no larger than d.")
    if feature_mode not in {"mf", "mf_product", "side_info", "hybrid"}:
        raise ValueError("feature_mode must be 'mf', 'mf_product', 'side_info', or 'hybrid'.")
    if slate_mode not in {"random", "contrastive"}:
        raise ValueError("slate_mode must be 'random' or 'contrastive'.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    ratings = np.asarray(raw.ratings, dtype=float)
    n_ratings = ratings.shape[0]
    order = rng.permutation(n_ratings)
    train_size = max(1, min(n_ratings - 1, int(train_fraction * n_ratings)))
    train = ratings[order[:train_size]]
    heldout = ratings[order[train_size:]]

    n_users = int(len(raw.user_ids))
    n_movies = int(len(raw.movie_ids))
    user_factors, movie_factors = _fit_matrix_factors(
        train,
        n_users=n_users,
        n_movies=n_movies,
        rank=rank,
        seed=seed,
    )
    user_table, movie_table = _movie_lens_feature_tables(
        raw,
        user_factors=user_factors,
        movie_factors=movie_factors,
        feature_mode=feature_mode,
    )
    slates = _sample_heldout_slates(heldout, K=K, T=T, rng=rng, slate_mode=slate_mode)
    user_ids = slates[:, 0, 0].astype(np.int64)
    movie_ids = slates[:, :, 1].astype(np.int64)
    rewards = (slates[:, :, 2] - 1.0) / 4.0
    if feature_mode == "mf_product":
        full_arms = _slate_product_features(user_factors, movie_factors, user_ids, movie_ids)
        full_arms = _lift_low_dimensional_arms(full_arms, d=d, seed=seed + 17)
    else:
        full_arms = _slate_features(user_table, movie_table, user_ids, movie_ids)
        full_arms = _compress_or_pad(full_arms, d=d, seed=seed + 17)
    full_arms = _normalize_arms(full_arms)

    masks = np.ones_like(full_arms, dtype=bool) if p == 1.0 else rng.random(size=full_arms.shape) < p
    masked_arms = np.where(masks, full_arms, 0.0)
    name = f"movielens100k_{feature_mode}_{slate_mode}_d{d}_r{rank}"
    return MovieLensBanditDataset(
        name=name,
        masked_arms=masked_arms,
        masks=masks,
        full_arms=full_arms,
        rewards=rewards,
        user_ids=user_ids,
        movie_ids=movie_ids,
        metadata={
            "name": name,
            "seed": seed,
            "p": p,
            "T": T,
            "K": K,
            "d": d,
            "rank": rank,
            "feature_mode": feature_mode,
            "slate_mode": slate_mode,
            "train_ratings": int(train.shape[0]),
            "heldout_ratings": int(heldout.shape[0]),
            "mask_rate": float(np.mean(masks)),
            "mean_best_reward": float(np.mean(np.max(rewards, axis=1))),
            "mean_reward": float(np.mean(rewards)),
        },
    )


def load_movielens_100k(
    *,
    data_dir: Path,
    allow_downloads: bool = False,
) -> MovieLensRawData:
    """Load MovieLens 100K from cache, optionally downloading it."""

    root = data_dir / "ml-100k"
    if not root.exists():
        if not allow_downloads:
            raise DatasetUnavailableError(
                f"MovieLens 100K is not cached at {root}; rerun with --allow-downloads."
            )
        _download_movielens_100k(data_dir)
    if not root.exists():
        raise DatasetUnavailableError(f"MovieLens 100K directory not found after download: {root}")

    users, user_features = _read_u_user(root / "u.user")
    movies, movie_features = _read_u_item(root / "u.item")
    user_index = {user_id: idx for idx, user_id in enumerate(users)}
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movies)}
    ratings = _read_u_data(root / "u.data", user_index=user_index, movie_index=movie_index)
    return MovieLensRawData(
        ratings=ratings,
        user_ids=users,
        movie_ids=movies,
        user_features=user_features,
        movie_features=movie_features,
        metadata={
            "source": "ml-100k",
            "n_users": int(len(users)),
            "n_movies": int(len(movies)),
            "n_ratings": int(ratings.shape[0]),
        },
    )


def make_synthetic_movielens_raw(
    *,
    seed: int,
    n_users: int = 40,
    n_movies: int = 80,
    n_ratings: int = 1200,
    rank: int = 4,
) -> MovieLensRawData:
    """Small MovieLens-like fixture for tests and quick runs."""

    rng = np.random.default_rng(seed)
    user_ids = np.arange(n_users, dtype=np.int64)
    movie_ids = np.arange(n_movies, dtype=np.int64)
    user_latent = rng.normal(size=(n_users, rank))
    movie_latent = rng.normal(size=(n_movies, rank))
    users = rng.integers(n_users, size=n_ratings)
    movies = rng.integers(n_movies, size=n_ratings)
    raw_scores = np.einsum("ij,ij->i", user_latent[users], movie_latent[movies])
    raw_scores = (raw_scores - np.mean(raw_scores)) / max(np.std(raw_scores), 1e-12)
    ratings = np.clip(np.rint(3.0 + raw_scores), 1.0, 5.0)
    user_features = np.column_stack(
        [
            rng.normal(size=n_users),
            rng.integers(2, size=n_users),
            rng.normal(size=n_users),
        ]
    )
    genres = rng.integers(2, size=(n_movies, 6)).astype(float)
    movie_features = np.column_stack([rng.normal(size=n_movies), genres])
    return MovieLensRawData(
        ratings=np.column_stack([users, movies, ratings]).astype(float),
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_features=user_features,
        movie_features=movie_features,
        metadata={
            "source": "synthetic_movielens",
            "n_users": n_users,
            "n_movies": n_movies,
            "n_ratings": n_ratings,
        },
    )


def _download_movielens_100k(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MOVIELENS_100K_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(data_dir)


def _read_u_user(path: Path) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    rows: list[tuple[int, int, str, str]] = []
    occupations: set[str] = set()
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            user_id, age, gender, occupation, _zip = line.rstrip("\n").split("|")
            rows.append((int(user_id), int(age), gender, occupation))
            occupations.add(occupation)
    occupation_index = {name: idx for idx, name in enumerate(sorted(occupations))}
    features = np.zeros((len(rows), 2 + len(occupation_index)), dtype=float)
    user_ids = np.empty(len(rows), dtype=np.int64)
    for row_idx, (user_id, age, gender, occupation) in enumerate(rows):
        user_ids[row_idx] = user_id
        features[row_idx, 0] = age
        features[row_idx, 1] = 1.0 if gender == "M" else 0.0
        features[row_idx, 2 + occupation_index[occupation]] = 1.0
    return user_ids, _standardize_columns(features)


def _read_u_item(path: Path) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    rows: list[list[str]] = []
    with path.open(encoding="latin-1") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            rows.append(row)
    movie_ids = np.empty(len(rows), dtype=np.int64)
    features = np.zeros((len(rows), 1 + 19), dtype=float)
    for row_idx, row in enumerate(rows):
        movie_ids[row_idx] = int(row[0])
        release_year = _parse_release_year(row[2])
        features[row_idx, 0] = release_year
        features[row_idx, 1:] = np.asarray(row[5:24], dtype=float)
    return movie_ids, _standardize_columns(features)


def _read_u_data(
    path: Path,
    *,
    user_index: dict[int, int],
    movie_index: dict[int, int],
) -> NDArray[np.float64]:
    rows: list[tuple[int, int, float]] = []
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            user_id, movie_id, rating, _timestamp = line.split()
            rows.append((user_index[int(user_id)], movie_index[int(movie_id)], float(rating)))
    return np.asarray(rows, dtype=float)


def _parse_release_year(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value[-4:])
    except ValueError:
        return 0.0


def _fit_matrix_factors(
    ratings: NDArray[np.float64],
    *,
    n_users: int,
    n_movies: int,
    rank: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.full((n_users, n_movies), np.nan, dtype=float)
    for user, movie, rating in ratings:
        matrix[int(user), int(movie)] = float(rating)
    global_mean = float(np.nanmean(matrix))
    observed = np.isfinite(matrix)
    user_counts = np.sum(observed, axis=1)
    movie_counts = np.sum(observed, axis=0)
    user_sums = np.where(observed, matrix, 0.0).sum(axis=1)
    movie_sums = np.where(observed, matrix, 0.0).sum(axis=0)
    user_means = np.divide(
        user_sums,
        user_counts,
        out=np.full(n_users, global_mean, dtype=float),
        where=user_counts > 0,
    )
    movie_means = np.divide(
        movie_sums,
        movie_counts,
        out=np.full(n_movies, global_mean, dtype=float),
        where=movie_counts > 0,
    )
    user_means = np.where(np.isfinite(user_means), user_means, global_mean)
    movie_means = np.where(np.isfinite(movie_means), movie_means, global_mean)
    filled = np.where(np.isnan(matrix), 0.5 * (user_means[:, None] + movie_means[None, :]), matrix)
    centered = filled - global_mean
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    active_rank = min(rank, singular_values.shape[0])
    user_factors = u[:, :active_rank] * np.sqrt(singular_values[:active_rank])
    movie_factors = vt[:active_rank].T * np.sqrt(singular_values[:active_rank])
    if active_rank < rank:
        rng = np.random.default_rng(seed)
        user_pad = 1e-6 * rng.normal(size=(n_users, rank - active_rank))
        movie_pad = 1e-6 * rng.normal(size=(n_movies, rank - active_rank))
        user_factors = np.hstack([user_factors, user_pad])
        movie_factors = np.hstack([movie_factors, movie_pad])
    return _standardize_columns(user_factors), _standardize_columns(movie_factors)


def _movie_lens_feature_tables(
    raw: MovieLensRawData,
    *,
    user_factors: NDArray[np.float64],
    movie_factors: NDArray[np.float64],
    feature_mode: MovieLensFeatureMode,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if feature_mode in {"mf", "mf_product"}:
        return user_factors, movie_factors
    if feature_mode == "side_info":
        return raw.user_features, raw.movie_features
    return (
        np.hstack([raw.user_features, user_factors]),
        np.hstack([raw.movie_features, movie_factors]),
    )


def _sample_heldout_slates(
    heldout: NDArray[np.float64],
    *,
    K: int,
    T: int,
    rng: np.random.Generator,
    slate_mode: MovieLensSlateMode = "random",
) -> NDArray[np.float64]:
    by_user: dict[int, list[NDArray[np.float64]]] = {}
    for row in heldout:
        by_user.setdefault(int(row[0]), []).append(row)
    if slate_mode == "contrastive":
        eligible = [
            user
            for user, rows in by_user.items()
            if len(rows) >= K
            and any(float(row[2]) >= 4.0 for row in rows)
            and any(float(row[2]) <= 2.0 for row in rows)
        ]
    else:
        eligible = [user for user, rows in by_user.items() if len(rows) >= K]
    if not eligible:
        raise DatasetUnavailableError("No held-out users have enough rated movies for the requested K.")
    slates = np.empty((T, K, 3), dtype=float)
    for t in range(T):
        user = int(rng.choice(eligible))
        rows = np.asarray(by_user[user], dtype=float)
        if slate_mode == "contrastive":
            slate = _sample_contrastive_slate(rows, K=K, rng=rng)
        else:
            selected = rng.choice(rows.shape[0], size=K, replace=False)
            slate = rows[selected]
        order = rng.permutation(K)
        slates[t] = slate[order]
    return slates


def _sample_contrastive_slate(
    rows: NDArray[np.float64],
    *,
    K: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    high = np.flatnonzero(rows[:, 2] >= 4.0)
    low = np.flatnonzero(rows[:, 2] <= 2.0)
    selected: list[int] = [
        int(rng.choice(high)),
        int(rng.choice(low)),
    ]
    remaining = np.setdiff1d(np.arange(rows.shape[0]), np.asarray(selected), assume_unique=False)
    n_extra = K - len(selected)
    if n_extra > 0:
        extra = rng.choice(remaining, size=n_extra, replace=False)
        selected.extend(int(item) for item in extra)
    return rows[np.asarray(selected, dtype=int)]


def _slate_features(
    user_table: NDArray[np.float64],
    movie_table: NDArray[np.float64],
    user_ids: NDArray[np.int64],
    movie_ids: NDArray[np.int64],
) -> NDArray[np.float64]:
    T, K = movie_ids.shape
    user_features = user_table[user_ids]
    movie_features = movie_table[movie_ids.reshape(-1)].reshape(T, K, -1)
    repeated_users = np.repeat(user_features[:, None, :], K, axis=1)
    interactions = repeated_users[:, :, :, None] * movie_features[:, :, None, :]
    return np.concatenate(
        [
            repeated_users,
            movie_features,
            interactions.reshape(T, K, -1),
        ],
        axis=2,
    )


def _slate_product_features(
    user_factors: NDArray[np.float64],
    movie_factors: NDArray[np.float64],
    user_ids: NDArray[np.int64],
    movie_ids: NDArray[np.int64],
) -> NDArray[np.float64]:
    T, K = movie_ids.shape
    users = user_factors[user_ids]
    movies = movie_factors[movie_ids.reshape(-1)].reshape(T, K, -1)
    return users[:, None, :] * movies


def _compress_or_pad(
    arms: NDArray[np.float64],
    *,
    d: int,
    seed: int,
) -> NDArray[np.float64]:
    T, K, q = arms.shape
    if q == d:
        return arms.copy()
    if q < d:
        output = np.zeros((T, K, d), dtype=float)
        output[:, :, :q] = arms
        return output
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(q, d)) / np.sqrt(float(d))
    return arms @ projection


def _lift_low_dimensional_arms(
    arms: NDArray[np.float64],
    *,
    d: int,
    seed: int,
) -> NDArray[np.float64]:
    T, K, q = arms.shape
    if q == d:
        return arms.copy()
    if q < d:
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(d, q))
        basis, _ = np.linalg.qr(raw, mode="reduced")
        return np.einsum("tkq,dq->tkd", arms, basis, optimize=True)
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(q, d)) / np.sqrt(float(d))
    return arms @ projection


def _normalize_arms(arms: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = arms - np.mean(arms.reshape(-1, arms.shape[-1]), axis=0).reshape(1, 1, -1)
    scale = np.std(centered.reshape(-1, centered.shape[-1]), axis=0).reshape(1, 1, -1)
    normalized = centered / np.where(scale > 1e-12, scale, 1.0)
    norms = np.linalg.norm(normalized, axis=2)
    return normalized / max(float(np.max(norms)), 1e-12)


def _standardize_columns(features: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(features, dtype=float)
    values = np.nan_to_num(values, copy=True)
    values = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(values, axis=0, keepdims=True)
    return values / np.where(scale > 1e-12, scale, 1.0)
