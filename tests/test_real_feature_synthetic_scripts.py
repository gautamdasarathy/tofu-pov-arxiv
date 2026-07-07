from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_FIGURE_STEMS = [
    "real_feature_synthetic_regret_over_time",
    "real_feature_synthetic_missingness_sweep",
    "real_feature_synthetic_references",
    "real_feature_synthetic_rank_diagnostics",
    "real_feature_synthetic_calibration_diagnostics",
]


def test_real_feature_synthetic_quick_mode_creates_expected_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    subprocess.run(
        [
            sys.executable,
            "experiments/run_real_feature_synthetic_experiments.py",
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

    trajectory_path = results_dir / "real_feature_synthetic_trajectories.csv"
    summary_path = results_dir / "real_feature_synthetic_summary.csv"
    calibration_path = results_dir / "real_feature_synthetic_calibration.csv"
    assert trajectory_path.exists()
    assert summary_path.exists()
    assert calibration_path.exists()
    assert trajectory_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    assert calibration_path.stat().st_size > 0

    with trajectory_path.open() as handle:
        columns = set(next(csv.reader(handle)))
    assert {
        "scenario",
        "source_dataset",
        "p",
        "seed",
        "method",
        "t",
        "instant_regret",
        "cumulative_regret",
        "reward",
        "active_rank",
    }.issubset(columns)

    with summary_path.open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert "Adaptive TOFU" in methods
    assert "Known-rank TOFU" in methods
    assert "Zero-imputed OFUL" in methods
    assert "Masked PSLB known-rank" in methods
    assert "Masked PSLB adaptive-rank" in methods

    with calibration_path.open() as handle:
        calibration_methods = {row["method_family"] for row in csv.DictReader(handle)}
    assert "TOFU adaptive-rank" in calibration_methods
    assert "Masked PSLB adaptive-rank" in calibration_methods
    assert "Zero-imputed OFUL" in calibration_methods

    subprocess.run(
        [
            sys.executable,
            "experiments/make_real_feature_synthetic_plots.py",
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

    for stem in EXPECTED_FIGURE_STEMS:
        for suffix in [".pdf", ".png"]:
            path = figures_dir / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path
