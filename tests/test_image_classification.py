from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tofu_pov import (
    make_image_classification_full_dataset,
    mask_image_classification_dataset,
)


pytest.importorskip("sklearn")


EXPECTED_IMAGE_FIGURE_STEMS = [
    "image_classification_missingness_sweep",
    "image_classification_regret_over_time",
    "image_classification_rank_diagnostics",
    "image_classification_full_info_refs",
    "image_classification_fixed_rank_validation",
    "image_classification_calibration_diagnostics",
]


def synthetic_image_pixels(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes = 3
    examples_per_class = 30
    n_features = 16
    centers = rng.normal(size=(n_classes, n_features))
    X_parts = []
    y_parts = []
    for class_id in range(n_classes):
        X_parts.append(centers[class_id] + 0.25 * rng.normal(size=(examples_per_class, n_features)))
        y_parts.append(np.full(examples_per_class, class_id, dtype=np.int64))
    return np.vstack(X_parts), np.concatenate(y_parts)


def test_image_classification_builder_shapes_rewards_masks_and_determinism():
    X, y = synthetic_image_pixels()

    first = make_image_classification_full_dataset(
        X,
        y,
        source="synthetic_images",
        representation_dim=5,
        seed=7,
        T=12,
        class_names=("zero", "one", "two"),
        train_fraction=0.7,
    )
    second = make_image_classification_full_dataset(
        X,
        y,
        source="synthetic_images",
        representation_dim=5,
        seed=7,
        T=12,
        class_names=("zero", "one", "two"),
        train_fraction=0.7,
    )

    assert first.full_arms.shape == (12, 3, 5)
    assert first.rewards.shape == (12, 3)
    assert np.allclose(first.rewards.sum(axis=1), 1.0)
    assert np.all(first.rewards[np.arange(first.T), first.labels] == 1.0)
    assert np.allclose(first.full_arms, second.full_arms)
    assert np.array_equal(first.labels, second.labels)
    assert first.metadata["heldout_accuracy"] >= 0.0
    assert first.metadata["heldout_accuracy"] <= 1.0

    full_observed = mask_image_classification_dataset(first, p=1.0, seed=3)
    masked = mask_image_classification_dataset(first, p=0.5, seed=3)

    assert np.all(full_observed.masks)
    assert np.allclose(full_observed.masked_arms, full_observed.full_arms)
    assert masked.masked_arms.shape == first.full_arms.shape
    assert masked.masks.shape == first.full_arms.shape
    assert np.allclose(masked.masked_arms, np.where(masked.masks, masked.full_arms, 0.0))
    assert masked.metadata["mask_rate"] > 0.0


def test_image_classification_quick_scripts_create_expected_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    subprocess.run(
        [
            sys.executable,
            "experiments/run_image_classification_experiments.py",
            "--quick",
            "--horizon",
            "24",
            "--p-values",
            "0.8",
            "--seeds",
            "1",
            "--validation-seeds",
            "1",
            "--results-dir",
            str(results_dir),
        ],
        cwd=root,
        env=env,
        check=True,
    )

    trajectory_path = results_dir / "image_classification_trajectories.csv"
    summary_path = results_dir / "image_classification_summary.csv"
    rank_selection_path = results_dir / "image_classification_rank_selection.csv"
    calibration_path = results_dir / "image_classification_calibration.csv"
    table_path = results_dir / "image_classification_table.md"
    for path in [trajectory_path, summary_path, rank_selection_path, calibration_path, table_path]:
        assert path.exists()
        assert path.stat().st_size > 0

    with trajectory_path.open() as handle:
        columns = set(next(csv.reader(handle)))
    assert {
        "scenario",
        "source",
        "p",
        "seed",
        "split",
        "method",
        "rank_label",
        "configured_rank",
        "t",
        "instant_regret",
        "cumulative_regret",
        "reward",
        "optimal_reward",
        "active_rank",
        "representation_dim",
        "heldout_accuracy",
    }.issubset(columns)

    with summary_path.open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert "Adaptive TOFU" in methods
    assert "TOFU fixed-rank best-val" in methods
    assert "Zero-imputed OFUL" in methods
    assert "Masked PSLB fixed-rank best-val" in methods
    assert "Masked PSLB adaptive-rank" in methods
    assert "Full-info OFUL" in methods
    assert "Full-info PSLB" in methods

    with calibration_path.open() as handle:
        calibration_methods = {row["method_family"] for row in csv.DictReader(handle)}
    assert "TOFU adaptive-rank" in calibration_methods
    assert "Masked PSLB adaptive-rank" in calibration_methods
    assert "Zero-imputed OFUL" in calibration_methods

    subprocess.run(
        [
            sys.executable,
            "experiments/make_image_classification_plots.py",
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

    for stem in EXPECTED_IMAGE_FIGURE_STEMS:
        for suffix in [".pdf", ".png"]:
            path = figures_dir / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path
