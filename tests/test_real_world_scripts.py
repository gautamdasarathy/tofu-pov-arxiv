from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_REAL_WORLD_FIGURE_STEMS = [
    "real_world_missingness_sweep",
    "real_world_regret_over_time",
    "real_world_rank_diagnostics",
    "real_world_full_info_refs",
    "real_world_fixed_rank_validation",
    "real_world_calibration_diagnostics",
]


def test_real_world_quick_mode_creates_expected_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    subprocess.run(
        [
            sys.executable,
            "experiments/run_real_world_experiments.py",
            "--quick",
            "--datasets",
            "toy_classification",
            "--horizon",
            "30",
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

    trajectory_path = results_dir / "real_world_trajectories.csv"
    summary_path = results_dir / "real_world_summary.csv"
    rank_selection_path = results_dir / "real_world_rank_selection.csv"
    calibration_path = results_dir / "real_world_calibration.csv"
    table_path = results_dir / "real_world_table.md"
    for path in [trajectory_path, summary_path, rank_selection_path, calibration_path, table_path]:
        assert path.exists()
        assert path.stat().st_size > 0

    with trajectory_path.open() as handle:
        columns = set(next(csv.reader(handle)))
    assert {
        "dataset",
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
            "experiments/make_real_world_plots.py",
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

    for stem in EXPECTED_REAL_WORLD_FIGURE_STEMS:
        for suffix in [".pdf", ".png"]:
            path = figures_dir / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path
