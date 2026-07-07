from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tofu_pov import make_mock_cnn_lowrank_full_dataset, mask_image_classification_dataset


EXPECTED_CNN_IMAGE_FIGURE_STEMS = [
    "cnn_image_missingness_sweep",
    "cnn_image_regret_over_time_p0p2",
    "cnn_image_effective_observations",
    "cnn_image_rank_diagnostics",
    "cnn_image_full_info_refs",
    "cnn_image_fixed_rank_validation",
    "cnn_image_calibration_diagnostics",
]


def test_mock_cnn_lowrank_builder_shapes_rank_masks_and_determinism():
    first = make_mock_cnn_lowrank_full_dataset(
        latent_dim=4,
        ambient_dim=20,
        seed=7,
        T=18,
        K=5,
    )
    second = make_mock_cnn_lowrank_full_dataset(
        latent_dim=4,
        ambient_dim=20,
        seed=7,
        T=18,
        K=5,
    )

    assert first.full_arms.shape == (18, 5, 20)
    assert first.rewards.shape == (18, 5)
    assert np.allclose(first.full_arms, second.full_arms)
    assert np.array_equal(first.labels, second.labels)
    assert np.allclose(first.rewards.sum(axis=1), 1.0)
    assert np.all(first.rewards[np.arange(first.T), first.labels] == 1.0)
    assert first.metadata["latent_dim"] == 4

    flattened = first.full_arms.reshape(-1, first.d)
    assert np.linalg.matrix_rank(flattened, tol=1e-8) <= 4

    full_observed = mask_image_classification_dataset(first, p=1.0, seed=3)
    masked = mask_image_classification_dataset(first, p=0.4, seed=3)

    assert np.all(full_observed.masks)
    assert np.allclose(full_observed.masked_arms, full_observed.full_arms)
    assert masked.masked_arms.shape == first.full_arms.shape
    assert masked.masks.shape == first.full_arms.shape
    assert np.allclose(masked.masked_arms, np.where(masked.masks, masked.full_arms, 0.0))


def test_cnn_image_quick_runner_creates_expected_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    subprocess.run(
        [
            sys.executable,
            "experiments/run_cnn_image_experiments.py",
            "--quick",
            "--horizon",
            "20",
            "--p-values",
            "0.2",
            "--seeds",
            "1",
            "--validation-seeds",
            "1",
            "--results-dir",
            str(results_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=root,
        env=env,
        check=True,
    )

    trajectory_path = results_dir / "cnn_image_trajectories.csv"
    summary_path = results_dir / "cnn_image_summary.csv"
    rank_selection_path = results_dir / "cnn_image_rank_selection.csv"
    calibration_path = results_dir / "cnn_image_calibration.csv"
    table_path = results_dir / "cnn_image_table.md"
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
        "warm_start_replay",
        "latent_dim",
        "ambient_dim",
        "heldout_accuracy",
    }.issubset(columns)

    with summary_path.open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert "TOFU full-history replay fixed-rank best-val" in methods
    assert "TOFU full-history replay adaptive-rank" in methods
    assert "TOFU first-epoch replay fixed-rank best-val" in methods
    assert "Zero-imputed OFUL" in methods
    assert "Masked PSLB fixed-rank best-val" in methods
    assert "Masked PSLB adaptive-rank" in methods
    assert "Full-info OFUL" in methods
    assert "Full-info PSLB" in methods

    with calibration_path.open() as handle:
        calibration_methods = {row["method_family"] for row in csv.DictReader(handle)}
    assert "TOFU full-history replay fixed-rank" in calibration_methods
    assert "TOFU first-epoch replay fixed-rank" in calibration_methods
    assert "TOFU full-history replay adaptive-rank" in calibration_methods
    assert "Masked PSLB adaptive-rank" in calibration_methods
    assert "Zero-imputed OFUL" in calibration_methods


def test_cnn_image_runner_supports_resume_and_partial_cleanup(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    base_cmd = [
        sys.executable,
        "experiments/run_cnn_image_experiments.py",
        "--quick",
        "--horizon",
        "20",
        "--p-values",
        "0.2",
        "--seeds",
        "1",
        "--validation-seeds",
        "1",
        "--results-dir",
        str(results_dir),
        "--cache-dir",
        str(cache_dir),
    ]

    subprocess.run(base_cmd, cwd=root, env=env, check=True)

    trajectories_path = results_dir / "cnn_image_trajectories.csv"
    selections_path = results_dir / "cnn_image_selections.csv"
    rank_path = results_dir / "cnn_image_rank_selection.csv"
    calibration_path = results_dir / "cnn_image_calibration.csv"

    for path in [trajectories_path, selections_path, rank_path, calibration_path]:
        assert path.exists() and path.stat().st_size > 0

    def read_rows(path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    initial_trajectory_rows = read_rows(trajectories_path)
    initial_selection_rows = read_rows(selections_path)
    initial_rank_rows = read_rows(rank_path)
    initial_calibration_rows = read_rows(calibration_path)

    refusal = subprocess.run(base_cmd, cwd=root, env=env, capture_output=True, text=True)
    assert refusal.returncode != 0
    assert "--resume" in (refusal.stderr + refusal.stdout)

    resume_run = subprocess.run(
        base_cmd + ["--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[skip]" in resume_run.stdout
    assert "resume from selections" in resume_run.stdout

    assert read_rows(trajectories_path) == initial_trajectory_rows
    assert read_rows(selections_path) == initial_selection_rows
    assert read_rows(rank_path) == initial_rank_rows
    assert read_rows(calibration_path) == initial_calibration_rows

    method_to_corrupt = "Zero-imputed OFUL"
    with trajectories_path.open() as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        all_rows = list(reader)
    truncated = []
    dropped_count = 0
    for row in all_rows:
        if row["method"] == method_to_corrupt and int(row["t"]) > 10:
            dropped_count += 1
            continue
        truncated.append(row)
    assert dropped_count > 0
    with trajectories_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(truncated)

    cleanup_run = subprocess.run(
        base_cmd + ["--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "dropping" in cleanup_run.stdout
    assert f"[run] {method_to_corrupt}" in cleanup_run.stdout

    final_rows = read_rows(trajectories_path)
    assert len(final_rows) == len(initial_trajectory_rows)
    final_method_counts = {row["method"]: 0 for row in final_rows}
    for row in final_rows:
        final_method_counts[row["method"]] += 1
    assert final_method_counts[method_to_corrupt] == 20


def test_cnn_image_quick_plotter_creates_expected_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")

    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    cache_dir = tmp_path / "cache"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    subprocess.run(
        [
            sys.executable,
            "experiments/run_cnn_image_experiments.py",
            "--quick",
            "--horizon",
            "20",
            "--p-values",
            "0.2",
            "--seeds",
            "1",
            "--validation-seeds",
            "1",
            "--results-dir",
            str(results_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "experiments/make_cnn_image_plots.py",
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

    for stem in EXPECTED_CNN_IMAGE_FIGURE_STEMS:
        for suffix in [".pdf", ".png"]:
            path = figures_dir / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path
