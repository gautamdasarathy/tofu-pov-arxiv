import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tofu_pov import load_text_product_context_full_dataset, mask_image_classification_dataset


def test_mock_text_product_context_builder_shapes_and_masks(tmp_path):
    full = load_text_product_context_full_dataset(
        "mock_text",
        latent_dim=4,
        ambient_dim=20,
        seed=0,
        T=25,
        cache_dir=tmp_path,
    )

    assert full.full_arms.shape == (25, 4, 20)
    assert full.rewards.shape == (25, 4)
    assert full.metadata["latent_dim"] == 4
    assert full.metadata["ambient_dim"] == 20
    assert np.linalg.matrix_rank(full.full_arms.reshape(-1, 20), tol=1e-8) <= 4

    masked = mask_image_classification_dataset(full, p=1.0, seed=1)
    assert masked.masks.all()
    np.testing.assert_allclose(masked.masked_arms, masked.full_arms)

    masked = mask_image_classification_dataset(full, p=0.4, seed=2)
    np.testing.assert_allclose(masked.masked_arms, np.where(masked.masks, full.full_arms, 0.0))

    grouped = load_text_product_context_full_dataset(
        "mock_text",
        latent_dim=4,
        ambient_dim=20,
        seed=0,
        T=25,
        cache_dir=tmp_path,
        lift_type="grouped",
    )
    assert grouped.metadata["lift_type"] == "grouped"
    assert np.linalg.matrix_rank(grouped.full_arms.reshape(-1, 20), tol=1e-8) <= 4
    assert not np.allclose(full.full_arms, grouped.full_arms)

    spiked = load_text_product_context_full_dataset(
        "mock_text",
        latent_dim=4,
        ambient_dim=20,
        seed=0,
        T=25,
        cache_dir=tmp_path,
        nuisance_scale=0.5,
    )
    assert spiked.metadata["nuisance_scale"] == 0.5
    assert spiked.metadata["nuisance_dim"] == 16
    assert np.linalg.matrix_rank(spiked.full_arms.reshape(-1, 20), tol=1e-8) > 4

    spectral = load_text_product_context_full_dataset(
        "mock_text",
        latent_dim=4,
        ambient_dim=20,
        seed=0,
        T=25,
        cache_dir=tmp_path,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.5,
    )
    spectral_weaker = load_text_product_context_full_dataset(
        "mock_text",
        latent_dim=4,
        ambient_dim=20,
        seed=0,
        T=25,
        cache_dir=tmp_path,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.25,
    )
    assert spectral.metadata["nuisance_mode"] == "spectral_tail"
    assert spectral.metadata["nuisance_spectral_ratio"] == 0.5
    assert spectral.metadata["nuisance_dim"] == 16
    assert spectral.metadata["realized_nuisance_spectral_ratio"] == 0.5
    assert np.linalg.matrix_rank(spectral.full_arms.reshape(-1, 20), tol=1e-8) > 4
    assert not np.allclose(spectral.full_arms, spectral_weaker.full_arms)


def test_text_product_quick_script_creates_expected_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "experiments/run_text_product_context_experiments.py",
            "--quick",
            "--results-dir",
            str(results_dir),
            "--cache-dir",
            str(cache_dir),
            "--no-references",
        ],
        check=True,
        env=env,
    )

    trajectory_path = results_dir / "text_product_trajectories.csv"
    summary_path = results_dir / "text_product_summary.csv"
    calibration_path = results_dir / "text_product_calibration.csv"
    rank_path = results_dir / "text_product_rank_selection.csv"
    table_path = results_dir / "text_product_table.md"
    for path in [trajectory_path, summary_path, calibration_path, rank_path, table_path]:
        assert path.exists()
        assert path.stat().st_size > 0

    with trajectory_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {
        "scenario",
        "p",
        "seed",
        "method",
        "t",
        "cumulative_regret",
        "active_rank",
    }.issubset(rows[0])
    methods = {row["method"] for row in rows}
    assert "TOFU full-history replay adaptive-rank" in methods
    assert "Zero-imputed OFUL" in methods
    assert "Masked PSLB adaptive-rank" in methods

    with calibration_path.open() as handle:
        calibration_rows = list(csv.DictReader(handle))
    assert calibration_rows
    selected = [row for row in calibration_rows if row["selected"] == "1"]
    assert selected


def test_text_product_pilot_mode_skips_calibration_and_filters_methods(tmp_path):
    results_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "experiments/run_text_product_context_experiments.py",
            "--quick",
            "--method-set",
            "pilot",
            "--skip-calibration",
            "--results-dir",
            str(results_dir),
            "--cache-dir",
            str(cache_dir),
            "--no-references",
        ],
        check=True,
        env=env,
    )

    with (results_dir / "text_product_trajectories.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["method"] for row in rows} == {
        "TOFU full-history replay fixed-rank best-val",
        "TOFU full-history replay adaptive-rank",
        "Zero-imputed OFUL",
        "Masked PSLB fixed-rank best-val",
    }

    with (results_dir / "text_product_calibration.csv").open() as handle:
        assert list(csv.DictReader(handle)) == []


def test_text_product_diagnostic_script_creates_expected_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")

    results_dir = tmp_path / "diagnostics"
    figures_dir = tmp_path / "figures"
    cache_dir = tmp_path / "cache"
    env = dict(os.environ)
    env["PYTHONPATH"] = f".{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            "experiments/diagnose_text_product_context.py",
            "--quick",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
            "--cache-dir",
            str(cache_dir),
        ],
        check=True,
        env=env,
    )

    expected_results = [
        "text_product_diagnostics.csv",
        "text_product_diagnostics_summary.csv",
        "text_product_diagnostics_table.md",
    ]
    for name in expected_results:
        path = results_dir / name
        assert path.exists()
        assert path.stat().st_size > 0

    with (results_dir / "text_product_diagnostics_summary.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert {
        "masked_refit_regret",
        "full_masked_score_corr",
        "subspace_overlap",
        "masked_energy_outside_full_subspace",
    }.issubset(row)

    for suffix in [".pdf", ".png"]:
        path = figures_dir / f"text_product_diagnostics_mock_text_product_m4_d20{suffix}"
        assert path.exists()
        assert path.stat().st_size > 0
