from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_FIGURE_STEMS = [
    "main_regret_over_time",
    "main_missingness_sweep",
    "main_adaptive_rank",
    "appendix_random_sanity",
    "appendix_rank_misspecification",
    "appendix_warm_start",
    "appendix_rank_recovery",
    "appendix_calibration",
    "appendix_feature_noise",
    "appendix_full_information_references",
    "appendix_pslb_intersection_diagnostic",
]


def test_plotting_quick_mode_creates_expected_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    subprocess.run(
        [
            sys.executable,
            "experiments/run_main_plot_experiments.py",
            "--quick",
            "--results-dir",
            str(results_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )
    _write_minimal_support_csvs(results_dir)

    trajectory_path = results_dir / "main_plot_trajectories.csv"
    summary_path = results_dir / "main_plot_summary.csv"
    assert trajectory_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    with summary_path.open() as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert "Masked PSLB known-rank" in methods
    assert "Masked PSLB adaptive-rank" in methods
    assert "Random" in methods
    from experiments import make_paper_plots

    assert "Random" not in make_paper_plots.MAIN_METHOD_ORDER
    assert "Random" in make_paper_plots.APPENDIX_TRAJECTORY_METHOD_ORDER
    with trajectory_path.open() as handle:
        columns = set(next(csv.reader(handle)))
    assert {
        "scenario",
        "p",
        "seed",
        "method",
        "t",
        "instant_regret",
        "cumulative_regret",
        "reward",
        "active_rank",
    }.issubset(columns)

    subprocess.run(
        [
            sys.executable,
            "experiments/make_paper_plots.py",
            "--quick",
            "--results-dir",
            str(results_dir),
            "--figures-dir",
            str(figures_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )

    for stem in EXPECTED_FIGURE_STEMS:
        for suffix in [".pdf", ".png"]:
            path = figures_dir / f"{stem}{suffix}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_minimal_support_csvs(results_dir: Path) -> None:
    mask_rows = []
    policies = [
        "TOFU-POV practical",
        "TOFU-POV zero-burnin",
        "TOFU-POV warm-start first-epoch",
        "TOFU-POV warm-start every-epoch",
        "Zero-imputed OFUL",
        "Masked PSLB",
        "Masked PSLB min-UCB",
        "Oracle-subspace OFUL",
        "PSLB full-action",
    ]
    for p in [0.8, 0.6, 0.4, 0.2]:
        for seed in [0, 1]:
            for i, policy in enumerate(policies):
                mask_rows.append(
                    {
                        "seed": seed,
                        "policy": policy,
                        "d": 12,
                        "m": 3,
                        "K": 8,
                        "T": 60,
                        "p": p,
                        "t_b": 10,
                        "c_b": 0.0,
                        "final_regret": (i + 1) * 10.0 / p + seed,
                        "avg_regret": 0.1,
                        "avg_reward": 1.0,
                        "action_entropy": 1.0,
                    }
                )
    _write_csv(results_dir / "synthetic_mask_sweep.csv", mask_rows)
    _write_csv(results_dir / "synthetic_burnin_sweep.csv", mask_rows)

    feature_noise_rows = []
    feature_methods = [
        "TOFU-POV warm-start first-epoch",
        "TOFU-POV warm-start every-epoch",
        "Zero-imputed OFUL",
        "Masked PSLB",
        "Oracle-subspace OFUL",
    ]
    for scenario in ["d30_m3_p03", "d50_m3_p02"]:
        for perturbation_std in [0.0, 0.03]:
            for seed in [0, 1]:
                for i, policy in enumerate(feature_methods):
                    feature_noise_rows.append(
                        {
                            "seed": seed,
                            "policy": policy,
                            "d": 30 if scenario == "d30_m3_p03" else 50,
                            "m": 3,
                            "K": 8,
                            "T": 60,
                            "p": 0.3 if scenario == "d30_m3_p03" else 0.2,
                            "t_b": 10,
                            "perturbation_std": perturbation_std,
                            "c_b": 0.0,
                            "final_regret": (i + 1) * 10.0 + 100.0 * perturbation_std + seed,
                            "avg_regret": 0.1,
                            "avg_reward": 1.0,
                            "action_entropy": 1.0,
                            "scenario": scenario,
                        }
                    )
    _write_csv(results_dir / "synthetic_feature_noise_sweep.csv", feature_noise_rows)

    _write_csv(
        results_dir / "synthetic_bias_sweep.csv",
        [
            {
                "seed": seed,
                "policy": f"TOFU-POV c_b={c_b:g}",
                "d": 12,
                "m": 3,
                "K": 8,
                "T": 60,
                "p": 0.8,
                "t_b": 10,
                "c_b": c_b,
                "final_regret": 20.0 + 100.0 * c_b + seed,
                "avg_regret": 0.1,
                "avg_reward": 1.0,
                "action_entropy": 1.0,
            }
            for c_b in [0.0, 0.1, 1.0]
            for seed in [0, 1]
        ],
    )

    scenarios = [
        ("dense_d12_m3_p08", 3),
        ("sparse_d30_m3_p03", 3),
        ("sparse_d50_m5_p02", 5),
    ]
    rank_summary_rows = []
    rank_rows = []
    method_specs = [
        ("TOFU fixed-rank", "under", 2),
        ("TOFU fixed-rank", "true", None),
        ("TOFU fixed-rank", "over", 6),
        ("TOFU adaptive-rank", "adaptive", 6),
        ("Masked PSLB fixed-rank", "under", 2),
        ("Masked PSLB fixed-rank", "true", None),
        ("Masked PSLB fixed-rank", "over", 6),
        ("Masked PSLB adaptive-rank", "adaptive", 6),
        ("Zero-imputed OFUL", "ambient", 30),
    ]
    for scenario, true_m in scenarios:
        for method, rank_label, configured in method_specs:
            configured_rank = true_m if configured is None else configured
            rank_summary_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "rank_label": rank_label,
                    "configured_rank": configured_rank,
                    "true_m": true_m,
                    "p": 0.8,
                    "n": 2,
                    "mean_final_regret": 10.0 + configured_rank,
                    "stderr_final_regret": 1.0,
                    "median_final_regret": 10.0 + configured_rank,
                    "mean_final_rank": float(configured_rank if rank_label != "adaptive" else true_m),
                    "mean_correct_rank_rate": 1.0 if rank_label in {"true", "adaptive"} else 0.0,
                }
            )
        for seed in [0, 1]:
            for method in ["TOFU adaptive-rank", "Masked PSLB adaptive-rank"]:
                rank_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "method": method,
                        "rank_label": "adaptive",
                        "configured_rank": 6,
                        "adaptive": 1,
                        "true_m": true_m,
                        "max_rank": 6,
                        "p": 0.8,
                        "d": 30,
                        "K": 8,
                        "T": 60,
                        "final_regret": 20.0,
                        "avg_regret": 0.1,
                        "final_rank": true_m,
                        "correct_rank_rate": 1.0,
                        "rank_history": f"{true_m} {true_m} {true_m}",
                        "rank_times": "11 22 44",
                    }
                )
    _write_csv(results_dir / "rank_adaptation_summary.csv", rank_summary_rows)
    _write_csv(results_dir / "rank_adaptation.csv", rank_rows)
