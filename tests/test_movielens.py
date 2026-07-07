from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tofu_pov import (
    load_movielens_bandit_dataset,
    make_movielens_bandit_dataset,
    make_synthetic_movielens_raw,
)


def test_movielens_builder_creates_masked_recommendation_bandit():
    raw = make_synthetic_movielens_raw(seed=0, n_users=20, n_movies=30, n_ratings=500, rank=3)

    data = make_movielens_bandit_dataset(
        raw,
        p=0.5,
        seed=1,
        T=15,
        K=4,
        d=12,
        rank=3,
        feature_mode="hybrid",
    )

    assert data.full_arms.shape == (15, 4, 12)
    assert data.masked_arms.shape == data.full_arms.shape
    assert data.masks.shape == data.full_arms.shape
    assert data.rewards.shape == (15, 4)
    assert data.movie_ids.shape == (15, 4)
    assert np.all(data.rewards >= 0.0)
    assert np.all(data.rewards <= 1.0)
    assert np.allclose(data.masked_arms, np.where(data.masks, data.full_arms, 0.0))
    assert data.metadata["mask_rate"] > 0.0


def test_movielens_loader_synthetic_quick_path_is_deterministic_and_p_one_is_full_observation():
    first = load_movielens_bandit_dataset(
        p=1.0,
        seed=2,
        T=10,
        K=4,
        d=10,
        rank=3,
        feature_mode="mf",
        synthetic=True,
    )
    second = load_movielens_bandit_dataset(
        p=1.0,
        seed=2,
        T=10,
        K=4,
        d=10,
        rank=3,
        feature_mode="mf",
        synthetic=True,
    )

    assert np.all(first.masks)
    assert np.allclose(first.masked_arms, first.full_arms)
    assert np.allclose(first.full_arms, second.full_arms)
    assert np.allclose(first.rewards, second.rewards)


def test_movielens_mf_product_contrastive_lifts_to_dense_low_rank_ambient_features():
    data = load_movielens_bandit_dataset(
        p=0.5,
        seed=3,
        T=20,
        K=5,
        d=30,
        rank=4,
        feature_mode="mf_product",
        slate_mode="contrastive",
        synthetic=True,
    )

    assert data.full_arms.shape == (20, 5, 30)
    assert np.linalg.matrix_rank(data.full_arms.reshape(-1, 30), tol=1e-8) <= 4
    assert np.count_nonzero(np.abs(data.full_arms) > 1e-12) / data.full_arms.size > 0.9
    assert np.all(data.rewards.max(axis=1) >= 0.75)
    assert np.all(data.rewards.min(axis=1) <= 0.25)


def test_movielens_quick_scripts_create_expected_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    subprocess.run(
        [
            sys.executable,
            "experiments/run_movielens_experiments.py",
            "--quick",
            "--horizon",
            "30",
            "--p-values",
            "0.8",
            "--seeds",
            "1",
            "--results-dir",
            str(results_dir),
        ],
        cwd=root,
        env=env,
        check=True,
    )

    trajectory_path = results_dir / "movielens_trajectories.csv"
    summary_path = results_dir / "movielens_summary.csv"
    rank_selection_path = results_dir / "movielens_rank_selection.csv"
    calibration_path = results_dir / "movielens_calibration.csv"
    assert trajectory_path.exists()
    assert summary_path.exists()
    assert rank_selection_path.exists()
    assert calibration_path.exists()
    with trajectory_path.open() as handle:
        columns = set(next(csv.reader(handle)))
    assert {
        "scenario",
        "feature_mode",
        "p",
        "seed",
        "method",
        "rank_label",
        "configured_rank",
        "t",
        "instant_regret",
        "cumulative_regret",
        "reward",
        "optimal_reward",
        "active_rank",
    }.issubset(columns)

    with calibration_path.open() as handle:
        calibration_methods = {row["method_family"] for row in csv.DictReader(handle)}
    assert "TOFU adaptive-rank" in calibration_methods
    assert "Masked PSLB adaptive-rank" in calibration_methods
    assert "Zero-imputed OFUL" in calibration_methods

    subprocess.run(
        [
            sys.executable,
            "experiments/make_movielens_plots.py",
            "--quick",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
        ],
        cwd=root,
        env=env,
        check=True,
    )

    for stem in [
        "movielens_missingness_sweep",
        "movielens_regret_over_time",
        "movielens_rank_diagnostics",
        "movielens_fixed_rank_validation",
        "movielens_calibration_diagnostics",
    ]:
        for suffix in [".pdf", ".png"]:
            path = figures_dir / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path
