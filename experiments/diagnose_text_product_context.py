"""Diagnose why zero-filled OFUL is strong in text product-context experiments."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_plot_cache_dir = tempfile.mkdtemp(prefix="tofu-text-diagnostic-plot-cache-")
os.environ.setdefault("MPLCONFIGDIR", _plot_cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", _plot_cache_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_text_product_context_experiments import (  # noqa: E402
    CACHE_DIR,
    RESULTS_DIR,
    SCENARIOS,
    load_full_dataset_for_run,
    mask_seed,
    materialize_scenario,
    parse_float_list,
    parse_scenario_list,
)
from tofu_pov import DatasetUnavailableError, mask_image_classification_dataset  # noqa: E402
from tofu_pov.subspace import corrected_covariance, sorted_eigendecomposition  # noqa: E402


DEFAULT_RESULTS_DIR = RESULTS_DIR / "diagnostics"
DEFAULT_FIGURES_DIR = ROOT / "figures" / "text_product_context_diagnostics"

DIAGNOSTIC_COLUMNS = [
    "scenario",
    "source",
    "p",
    "seed",
    "T",
    "K",
    "m",
    "d",
    "heldout_accuracy",
    "mask_rate",
    "observed_coordinates_per_arm",
    "effective_observations_per_latent_dim",
    "mean_norm_retention",
    "full_oracle_regret",
    "masked_same_theta_regret",
    "masked_refit_regret",
    "full_oracle_accuracy",
    "masked_same_theta_accuracy",
    "masked_refit_accuracy",
    "full_masked_score_corr",
    "full_masked_winner_agreement",
    "full_winner_still_best_masked",
    "mean_full_margin",
    "mean_masked_margin_on_full_winner",
    "full_lambda_m",
    "full_lambda_m_plus_1",
    "full_eigengap_m",
    "corrected_lambda_m",
    "corrected_lambda_m_plus_1",
    "corrected_eigengap_m",
    "subspace_overlap",
    "corrected_subspace_overlap",
    "mean_cos2_principal_angle",
    "corrected_mean_cos2_principal_angle",
    "min_cos_principal_angle",
    "corrected_min_cos_principal_angle",
    "masked_top_m_energy_fraction",
    "masked_energy_outside_full_subspace",
    "masked_effective_rank",
]


def ridge_theta(X: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
    d = X.shape[1]
    gram = X.T @ X + lambda_reg * np.eye(d)
    rhs = X.T @ y
    return np.linalg.solve(gram, rhs)


def per_round_corr(a: np.ndarray, b: np.ndarray) -> float:
    values: list[float] = []
    for row_a, row_b in zip(a, b):
        std_a = float(np.std(row_a))
        std_b = float(np.std(row_b))
        if std_a <= 1e-12 or std_b <= 1e-12:
            continue
        values.append(float(np.corrcoef(row_a, row_b)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def action_regret_and_accuracy(scores: np.ndarray, rewards: np.ndarray) -> tuple[float, float]:
    actions = np.argmax(scores, axis=1)
    selected = rewards[np.arange(rewards.shape[0]), actions]
    optimal = np.max(rewards, axis=1)
    return float(np.sum(optimal - selected)), float(np.mean(selected == optimal))


def top_basis_and_eigs(arms: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    X = arms.reshape(-1, arms.shape[-1])
    cov = X.T @ X / max(X.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    return eigvecs[:, :rank], eigvals


def basis_overlap_metrics(reference_basis: np.ndarray, candidate_basis: np.ndarray) -> dict[str, float]:
    rank = reference_basis.shape[1]
    singular_values = np.linalg.svd(reference_basis.T @ candidate_basis, compute_uv=False)
    return {
        "overlap": float(np.sum(singular_values**2) / rank),
        "mean_cos2": float(np.mean(singular_values**2)),
        "min_cos": float(np.min(singular_values)) if singular_values.size else float("nan"),
    }


def eigengap_at_rank(eigenvalues: np.ndarray, rank: int) -> tuple[float, float, float]:
    if eigenvalues.size < rank:
        return float("nan"), float("nan"), float("nan")
    lambda_m = float(eigenvalues[rank - 1])
    lambda_next = float(eigenvalues[rank]) if eigenvalues.size > rank else 0.0
    gap = lambda_m / max(lambda_next, 1e-12)
    return lambda_m, lambda_next, float(gap)


def subspace_metrics(
    full_arms: np.ndarray,
    masked_arms: np.ndarray,
    *,
    p: float,
    rank: int,
) -> dict[str, float]:
    full_basis, full_eigs = top_basis_and_eigs(full_arms, rank)
    masked_basis, masked_eigs = top_basis_and_eigs(masked_arms, rank)
    raw_overlap = basis_overlap_metrics(full_basis, masked_basis)
    corrected_cov = corrected_covariance(masked_arms.reshape(-1, masked_arms.shape[-1]), p)
    corrected_eigs, corrected_eigvecs = sorted_eigendecomposition(corrected_cov)
    corrected_overlap = basis_overlap_metrics(full_basis, corrected_eigvecs[:, :rank])
    full_lambda_m, full_lambda_next, full_gap = eigengap_at_rank(full_eigs, rank)
    corrected_lambda_m, corrected_lambda_next, corrected_gap = eigengap_at_rank(
        corrected_eigs,
        rank,
    )
    total_energy = float(np.sum(masked_eigs))
    top_energy = float(np.sum(masked_eigs[:rank]))
    X_masked = masked_arms.reshape(-1, masked_arms.shape[-1])
    projected = X_masked @ full_basis @ full_basis.T
    residual_energy = float(np.sum((X_masked - projected) ** 2))
    masked_energy = float(np.sum(X_masked**2))
    effective_rank = float(total_energy**2 / max(float(np.sum(masked_eigs**2)), 1e-12))
    return {
        "full_lambda_m": full_lambda_m,
        "full_lambda_m_plus_1": full_lambda_next,
        "full_eigengap_m": full_gap,
        "corrected_lambda_m": corrected_lambda_m,
        "corrected_lambda_m_plus_1": corrected_lambda_next,
        "corrected_eigengap_m": corrected_gap,
        "subspace_overlap": raw_overlap["overlap"],
        "corrected_subspace_overlap": corrected_overlap["overlap"],
        "mean_cos2_principal_angle": raw_overlap["mean_cos2"],
        "corrected_mean_cos2_principal_angle": corrected_overlap["mean_cos2"],
        "min_cos_principal_angle": raw_overlap["min_cos"],
        "corrected_min_cos_principal_angle": corrected_overlap["min_cos"],
        "masked_top_m_energy_fraction": top_energy / max(total_energy, 1e-12),
        "masked_energy_outside_full_subspace": residual_energy / max(masked_energy, 1e-12),
        "masked_effective_rank": effective_rank,
    }


def diagnose_dataset(
    scenario_name: str,
    *,
    p: float,
    seed: int,
    quick: bool,
    horizon: int | None,
    allow_downloads: bool,
    cache_dir: Path,
    force_rebuild: bool,
    lambda_reg: float,
) -> dict[str, float | int | str]:
    scenario = materialize_scenario(scenario_name, quick=quick, horizon=horizon)
    full_data = load_full_dataset_for_run(
        scenario,
        seed=seed,
        allow_downloads=allow_downloads,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )
    data = mask_image_classification_dataset(full_data, p=p, seed=mask_seed(seed, p))
    rank = int(data.metadata.get("latent_dim", scenario.latent_dim))

    X_full = data.full_arms.reshape(-1, data.d)
    X_masked = data.masked_arms.reshape(-1, data.d)
    y = data.rewards.reshape(-1)
    theta_full = ridge_theta(X_full, y, lambda_reg)
    theta_masked = ridge_theta(X_masked, y, lambda_reg)

    full_scores = data.full_arms @ theta_full
    masked_scores_same_theta = data.masked_arms @ theta_full
    masked_scores_refit = data.masked_arms @ theta_masked

    full_regret, full_accuracy = action_regret_and_accuracy(full_scores, data.rewards)
    same_regret, same_accuracy = action_regret_and_accuracy(masked_scores_same_theta, data.rewards)
    refit_regret, refit_accuracy = action_regret_and_accuracy(masked_scores_refit, data.rewards)

    full_winner = np.argmax(full_scores, axis=1)
    masked_winner_same_theta = np.argmax(masked_scores_same_theta, axis=1)
    full_masked_winner_agreement = float(np.mean(full_winner == masked_winner_same_theta))
    masked_sorted = np.sort(masked_scores_same_theta, axis=1)
    full_sorted = np.sort(full_scores, axis=1)
    full_winner_scores_under_mask = masked_scores_same_theta[
        np.arange(data.T),
        full_winner,
    ]
    masked_scores_without_full_winner = masked_scores_same_theta.copy()
    masked_scores_without_full_winner[np.arange(data.T), full_winner] = -np.inf
    masked_margin_on_full_winner = full_winner_scores_under_mask - np.max(
        masked_scores_without_full_winner,
        axis=1,
    )
    full_winner_still_best_masked = float(np.mean(masked_margin_on_full_winner >= 0.0))
    full_norm = np.linalg.norm(data.full_arms, axis=2)
    masked_norm = np.linalg.norm(data.masked_arms, axis=2)
    norm_retention = masked_norm / np.maximum(full_norm, 1e-12)

    row: dict[str, float | int | str] = {
        "scenario": scenario.name,
        "source": scenario.source,
        "p": p,
        "seed": seed,
        "T": data.T,
        "K": data.K,
        "m": rank,
        "d": data.d,
        "heldout_accuracy": float(data.metadata.get("heldout_accuracy", np.nan)),
        "mask_rate": float(np.mean(data.masks)),
        "observed_coordinates_per_arm": float(np.mean(np.sum(data.masks, axis=2))),
        "effective_observations_per_latent_dim": float(p * data.d / rank),
        "mean_norm_retention": float(np.mean(norm_retention)),
        "full_oracle_regret": full_regret,
        "masked_same_theta_regret": same_regret,
        "masked_refit_regret": refit_regret,
        "full_oracle_accuracy": full_accuracy,
        "masked_same_theta_accuracy": same_accuracy,
        "masked_refit_accuracy": refit_accuracy,
        "full_masked_score_corr": per_round_corr(full_scores, masked_scores_same_theta),
        "full_masked_winner_agreement": full_masked_winner_agreement,
        "full_winner_still_best_masked": full_winner_still_best_masked,
        "mean_full_margin": float(np.mean(full_sorted[:, -1] - full_sorted[:, -2])),
        "mean_masked_margin_on_full_winner": float(np.mean(masked_margin_on_full_winner)),
    }
    row.update(subspace_metrics(data.full_arms, data.masked_arms, p=p, rank=rank))
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["scenario", "source", "p"]
    numeric_cols = [col for col in DIAGNOSTIC_COLUMNS if col not in {*group_keys, "seed"}]
    grouped = rows.groupby(group_keys, as_index=False)
    mean = grouped[numeric_cols].mean()
    counts = rows.groupby(group_keys, as_index=False)["seed"].count()
    counts = counts.rename(columns={"seed": "n"})
    return mean.merge(counts, on=group_keys).sort_values(["scenario", "p"])


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(summary: pd.DataFrame, figures_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for scenario, frame in summary.groupby("scenario"):
        frame = frame.sort_values("p")
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))

        axes[0].plot(frame["p"], frame["masked_refit_regret"], marker="o", label="Zero-filled ridge oracle")
        axes[0].plot(frame["p"], frame["masked_same_theta_regret"], marker="s", label="Masked with full oracle theta")
        axes[0].plot(frame["p"], frame["full_oracle_regret"], marker="^", label="Full-arm ridge oracle")
        axes[0].invert_xaxis()
        axes[0].set_title("Batch linear oracle regret")
        axes[0].set_xlabel("Observation probability p")
        axes[0].set_ylabel("Cumulative regret")
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].plot(frame["p"], frame["full_masked_score_corr"], marker="o", label="Score correlation")
        axes[1].plot(frame["p"], frame["full_masked_winner_agreement"], marker="s", label="Winner agreement")
        axes[1].plot(frame["p"], frame["masked_refit_accuracy"], marker="^", label="Zero-filled oracle accuracy")
        axes[1].invert_xaxis()
        axes[1].set_ylim(0.0, 1.02)
        axes[1].set_title("Signal preserved by zero-fill")
        axes[1].set_xlabel("Observation probability p")
        axes[1].legend(frameon=False, fontsize=8)

        axes[2].plot(frame["p"], frame["subspace_overlap"], marker="o", label="Raw top-m subspace overlap")
        axes[2].plot(
            frame["p"],
            frame["corrected_subspace_overlap"],
            marker="D",
            label="Corrected top-m subspace overlap",
        )
        axes[2].plot(
            frame["p"],
            1.0 - frame["masked_energy_outside_full_subspace"],
            marker="s",
            label="Masked energy in true subspace",
        )
        axes[2].plot(frame["p"], frame["masked_top_m_energy_fraction"], marker="^", label="Masked top-m energy")
        axes[2].invert_xaxis()
        axes[2].set_ylim(0.0, 1.02)
        axes[2].set_title("Subspace damage under masking")
        axes[2].set_xlabel("Observation probability p")
        axes[2].legend(frameon=False, fontsize=8)

        fig.suptitle(f"Text product-context diagnostics: {scenario}", y=1.08)
        save_figure(fig, figures_dir, f"text_product_diagnostics_{scenario}")


def write_table(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "| Scenario | p | n | p d / m | Zero-fill ridge regret | Score corr. | Winner agreement | Corrected subspace overlap | Full eigengap | Corrected eigengap | Masked effective rank |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {scenario} | {p:.3f} | {n:d} | {eff:.3f} | {regret:.3f} | {corr:.3f} | {agree:.3f} | {corrected_overlap:.3f} | {full_gap:.3f} | {corrected_gap:.3f} | {effective_rank:.3f} |\n".format(
                scenario=row["scenario"],
                p=float(row["p"]),
                n=int(row["n"]),
                eff=float(row["effective_observations_per_latent_dim"]),
                regret=float(row["masked_refit_regret"]),
                corr=float(row["full_masked_score_corr"]),
                agree=float(row["full_masked_winner_agreement"]),
                corrected_overlap=float(row["corrected_subspace_overlap"]),
                full_gap=float(row["full_eigengap_m"]),
                corrected_gap=float(row["corrected_eigengap_m"]),
                effective_rank=float(row["masked_effective_rank"]),
            )
        )
    path.write_text("".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a tiny mock diagnostic.")
    parser.add_argument("--scenarios", type=parse_scenario_list, default=None)
    parser.add_argument("--p-values", type=parse_float_list, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenario_names = args.scenarios or (["mock_text_product_m4_d20"] if args.quick else ["text20news4_product_m20_d500"])
    p_values = args.p_values or ([0.4] if args.quick else [0.075, 0.05, 0.03, 0.02])
    seed_count = args.seeds if args.seeds is not None else (1 if args.quick else 3)
    rows = []
    for scenario_name in scenario_names:
        for p in p_values:
            for seed in range(seed_count):
                print(f"Diagnosing {scenario_name} p={p:g} seed={seed}")
                rows.append(
                    diagnose_dataset(
                        scenario_name,
                        p=p,
                        seed=seed,
                        quick=args.quick,
                        horizon=args.horizon,
                        allow_downloads=args.allow_downloads,
                        cache_dir=args.cache_dir,
                        force_rebuild=args.force_rebuild,
                        lambda_reg=args.lambda_reg,
                    )
                )
    frame = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    summary = summarize(frame)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.results_dir / "text_product_diagnostics.csv", index=False)
    summary.to_csv(args.results_dir / "text_product_diagnostics_summary.csv", index=False)
    write_table(args.results_dir / "text_product_diagnostics_table.md", summary)
    plot_diagnostics(summary, args.figures_dir)
    print(f"Wrote diagnostics to {args.results_dir}")
    print(f"Wrote figures to {args.figures_dir}")


if __name__ == "__main__":
    try:
        main()
    except DatasetUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
