"""Diagnose why masked PSLB can outperform TOFU on image-bandit reductions."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_image_classification_experiments as image_runner
from tofu_pov import ZeroImputedOFUL, mask_image_classification_dataset
from tofu_pov.imputation import ImputationError, impute_arms
from tofu_pov.learner import TOFUPOV
from tofu_pov.baselines import MaskedPSLB


DEFAULT_RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="digits_image_d40")
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--calibration-csv", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args()


def selected_value(
    calibration: pd.DataFrame,
    *,
    scenario: str,
    p: float,
    family: str,
    column: str,
) -> float | int:
    rows = calibration[
        (calibration["scenario"] == scenario)
        & np.isclose(calibration["p"].astype(float), p)
        & (calibration["method_family"] == family)
        & (calibration["selected"].astype(int) == 1)
    ]
    if rows.empty:
        raise ValueError(f"No selected calibration row for {scenario}, p={p}, {family}.")
    values = rows[column].dropna().to_numpy()
    if values.size == 0:
        raise ValueError(f"Selected calibration rows have no value for {column}: {family}.")
    counts = Counter(values.tolist())
    value, _ = counts.most_common(1)[0]
    if column == "rank":
        return int(value)
    return float(value)


def score_rank(scores: NDArray[np.float64], optimal_action: int) -> int:
    order = np.argsort(-scores)
    return int(np.where(order == optimal_action)[0][0]) + 1


def score_margin(scores: NDArray[np.float64], optimal_action: int) -> float:
    if scores.size <= 1:
        return 0.0
    other = np.delete(scores, optimal_action)
    return float(scores[optimal_action] - np.max(other))


def projection_energy(arms: NDArray[np.float64], U: NDArray[np.float64]) -> float:
    denominator = float(np.sum(arms * arms))
    if denominator <= 0.0:
        return math.nan
    projected_norm_sq = float(np.sum((arms @ U) ** 2))
    return projected_norm_sq / denominator


def relative_error(estimate: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    denominator = float(np.linalg.norm(truth))
    if denominator <= 0.0:
        return math.nan
    return float(np.linalg.norm(estimate - truth) / denominator)


def mean_or_nan(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else math.nan


def stderr_or_nan(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    if len(clean) <= 1:
        return 0.0 if clean else math.nan
    return float(np.std(clean, ddof=1) / math.sqrt(len(clean)))


def model_update_count(policy: object) -> int | str:
    if isinstance(policy, TOFUPOV):
        return "" if policy.oful is None else int(policy.oful.n_updates)
    if isinstance(policy, MaskedPSLB):
        return int(policy.ambient_model.n_updates)
    if isinstance(policy, ZeroImputedOFUL):
        return int(policy.model.n_updates)
    return ""


def diagnostic_policy_set(
    scenario: image_runner.ImageClassificationScenario,
    *,
    p: float,
    K: int,
    seed: int,
    calibration: pd.DataFrame,
) -> list[tuple[str, Any]]:
    tofu_rank = selected_value(
        calibration,
        scenario=scenario.name,
        p=p,
        family="TOFU fixed-rank",
        column="rank",
    )
    pslb_rank = selected_value(
        calibration,
        scenario=scenario.name,
        p=p,
        family="Masked PSLB fixed-rank",
        column="rank",
    )
    oful_lambda = selected_value(
        calibration,
        scenario=scenario.name,
        p=p,
        family="Zero-imputed OFUL",
        column="lambda_reg",
    )
    oful_beta = selected_value(
        calibration,
        scenario=scenario.name,
        p=p,
        family="Zero-imputed OFUL",
        column="beta_scale",
    )
    return [
        (
            f"TOFU fixed-rank best-val (rank={tofu_rank})",
            image_runner.tofu_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=int(tofu_rank),
                adaptive=False,
            ),
        ),
        (
            f"TOFU full-history replay (rank={tofu_rank})",
            TOFUPOV(
                replace(
                    image_runner.tofu_policy(
                        scenario,
                        p=p,
                        K=K,
                        seed=seed,
                        rank=int(tofu_rank),
                        adaptive=False,
                    ).config,
                    warm_start_replay="full_history_every_epoch",
                )
            ),
        ),
        (
            f"Masked PSLB fixed-rank best-val (rank={pslb_rank})",
            image_runner.masked_pslb_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=int(pslb_rank),
                adaptive=False,
            ),
        ),
        (
            f"Zero-imputed OFUL (lambda={oful_lambda:g}, beta={oful_beta:g})",
            ZeroImputedOFUL(
                d=scenario.d,
                lambda_reg=float(oful_lambda),
                S=1.0,
                R=scenario.R,
                delta=0.05,
                beta_scale=float(oful_beta),
            ),
        ),
    ]


def diagnose_policy(
    *,
    method: str,
    policy: object,
    data: Any,
    seed: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    if hasattr(policy, "reset"):
        policy.reset(seed)

    cumulative_regret = 0.0
    per_round: list[dict[str, float | int | str]] = []
    aggregates: dict[str, list[float]] = defaultdict(list)
    post_burnin_start: int | None = None

    for t in range(1, data.T + 1):
        masked_arms = data.masked_arms[t - 1]
        masks = data.masks[t - 1]
        full_arms = data.full_arms[t - 1]
        rewards = data.rewards[t - 1]
        optimal_action = int(np.argmax(rewards))
        action = int(policy.observe(masked_arms, masks, full_arms))
        selected_reward = float(rewards[action])
        instant_regret = float(np.max(rewards) - selected_reward)
        cumulative_regret += instant_regret

        state = policy.state_dict() if hasattr(policy, "state_dict") else {}
        scores = state.get("last_scores")
        if not isinstance(scores, np.ndarray) and hasattr(policy, "last_scores"):
            scores = getattr(policy, "last_scores")
        active_rank = state.get("active_m")
        if active_rank is None and isinstance(policy, ZeroImputedOFUL):
            active_rank = policy.d
        score_rank_value = math.nan
        score_margin_value = math.nan
        if isinstance(scores, np.ndarray) and scores.shape[0] == data.K:
            score_rank_value = score_rank(scores, optimal_action)
            score_margin_value = score_margin(scores, optimal_action)

        U = state.get("U_hat")
        full_energy = math.nan
        masked_energy = math.nan
        imputation_error_all = math.nan
        imputation_error_selected = math.nan
        imputation_error_optimal = math.nan
        if isinstance(U, np.ndarray):
            full_energy = projection_energy(full_arms, U)
            masked_energy = projection_energy(masked_arms, U)
            if isinstance(policy, TOFUPOV):
                try:
                    imputed, _ = impute_arms(
                        masked_arms,
                        masks,
                        U,
                        impute_ridge=policy.config.impute_ridge,
                    )
                    imputation_error_all = relative_error(imputed, full_arms)
                    imputation_error_selected = relative_error(imputed[action], full_arms[action])
                    imputation_error_optimal = relative_error(
                        imputed[optimal_action],
                        full_arms[optimal_action],
                    )
                except ImputationError:
                    imputation_error_all = math.inf
                    imputation_error_selected = math.inf
                    imputation_error_optimal = math.inf

        pslb_fallback = state.get("last_intersection_fallback")
        feasible_count = state.get("last_intersection_feasible_count")
        sample_count = state.get("last_intersection_sample_count")

        selected_full_norm = float(np.linalg.norm(full_arms[action]))
        selected_masked_norm = float(np.linalg.norm(masked_arms[action]))
        selected_norm_ratio = (
            selected_masked_norm / selected_full_norm if selected_full_norm > 0.0 else math.nan
        )
        zero_fill_error_all = relative_error(masked_arms, full_arms)

        row = {
            "method": method,
            "t": t,
            "action": action,
            "optimal_action": optimal_action,
            "reward": selected_reward,
            "instant_regret": instant_regret,
            "cumulative_regret": cumulative_regret,
            "score_rank_optimal": score_rank_value,
            "score_margin_optimal": score_margin_value,
            "active_rank": "" if active_rank is None else int(active_rank),
            "full_projection_energy": full_energy,
            "masked_projection_energy": masked_energy,
            "tofu_imputation_error_all": imputation_error_all,
            "tofu_imputation_error_selected": imputation_error_selected,
            "tofu_imputation_error_optimal": imputation_error_optimal,
            "selected_observed_fraction": float(np.mean(masks[action])),
            "selected_masked_norm_ratio": selected_norm_ratio,
            "zero_fill_error_all": zero_fill_error_all,
            "pslb_feasible_count": "" if feasible_count is None else int(feasible_count),
            "pslb_sample_count": "" if sample_count is None else int(sample_count),
            "pslb_used_fallback": int(pslb_fallback is not None),
        }
        per_round.append(row)

        learning_phase = active_rank not in (None, "")
        if learning_phase and post_burnin_start is None:
            post_burnin_start = t
        if learning_phase:
            aggregates["score_rank_optimal"].append(score_rank_value)
            aggregates["score_margin_optimal"].append(score_margin_value)
            aggregates["full_projection_energy"].append(full_energy)
            aggregates["masked_projection_energy"].append(masked_energy)
            aggregates["tofu_imputation_error_all"].append(imputation_error_all)
            aggregates["tofu_imputation_error_selected"].append(imputation_error_selected)
            aggregates["tofu_imputation_error_optimal"].append(imputation_error_optimal)
            aggregates["selected_observed_fraction"].append(float(np.mean(masks[action])))
            aggregates["selected_masked_norm_ratio"].append(selected_norm_ratio)
            aggregates["zero_fill_error_all"].append(zero_fill_error_all)
            if feasible_count not in (None, ""):
                aggregates["pslb_feasible_count"].append(float(feasible_count))
            if sample_count not in (None, ""):
                aggregates["pslb_sample_count"].append(float(sample_count))
            aggregates["pslb_used_fallback"].append(float(pslb_fallback is not None))

        policy.update(selected_reward)

    final_state = policy.state_dict() if hasattr(policy, "state_dict") else {}
    rank_history = final_state.get("rank_history", [])
    summary = {
        "method": method,
        "final_regret": cumulative_regret,
        "mean_reward": float(np.mean([row["reward"] for row in per_round])),
        "optimal_top1_rate": float(
            np.mean([row["score_rank_optimal"] == 1 for row in per_round if np.isfinite(row["score_rank_optimal"])])
        ),
        "mean_optimal_score_rank": mean_or_nan(aggregates["score_rank_optimal"]),
        "mean_optimal_score_margin": mean_or_nan(aggregates["score_margin_optimal"]),
        "final_active_rank": per_round[-1]["active_rank"],
        "final_model_updates": model_update_count(policy),
        "rank_history": " ".join(str(item) for item in rank_history),
        "mean_full_projection_energy": mean_or_nan(aggregates["full_projection_energy"]),
        "mean_masked_projection_energy": mean_or_nan(aggregates["masked_projection_energy"]),
        "mean_tofu_imputation_error_all": mean_or_nan(aggregates["tofu_imputation_error_all"]),
        "mean_tofu_imputation_error_selected": mean_or_nan(
            aggregates["tofu_imputation_error_selected"]
        ),
        "mean_tofu_imputation_error_optimal": mean_or_nan(
            aggregates["tofu_imputation_error_optimal"]
        ),
        "mean_selected_observed_fraction": mean_or_nan(aggregates["selected_observed_fraction"]),
        "mean_selected_masked_norm_ratio": mean_or_nan(aggregates["selected_masked_norm_ratio"]),
        "mean_zero_fill_error_all": mean_or_nan(aggregates["zero_fill_error_all"]),
        "mean_pslb_feasible_count": mean_or_nan(aggregates["pslb_feasible_count"]),
        "mean_pslb_sample_count": mean_or_nan(aggregates["pslb_sample_count"]),
        "pslb_fallback_rate": mean_or_nan(aggregates["pslb_used_fallback"]),
        "post_burnin_start": "" if post_burnin_start is None else post_burnin_start,
    }
    return summary, per_round


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    scenario: str,
    p: float,
    seed: int,
    summary_rows: list[dict[str, float | int | str]],
) -> None:
    lines = [
        f"# PSLB vs TOFU Diagnostic: `{scenario}`, p={p:g}, seed={seed}\n\n",
        "This diagnostic reruns calibrated contenders on the same masked decision sequence and logs score alignment, rank/subspace behavior, PSLB intersection feasibility, and TOFU imputation error.\n\n",
        "## Summary\n\n",
    ]
    columns = [
        "method",
        "final_regret",
        "optimal_top1_rate",
        "mean_optimal_score_rank",
        "mean_optimal_score_margin",
        "final_active_rank",
        "final_model_updates",
        "mean_full_projection_energy",
        "mean_tofu_imputation_error_all",
        "pslb_fallback_rate",
    ]
    lines.append("| " + " | ".join(columns) + " |\n")
    lines.append("|" + "|".join("---" for _ in columns) + "|\n")
    for row in summary_rows:
        formatted = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                formatted.append(f"{value:.4g}" if np.isfinite(value) else "")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |\n")
    lines.append(
        "\nA low `optimal_top1_rate` means the policy's own UCB scores rarely rank the true label first. "
        "Large TOFU imputation error means the low-rank reconstruction is substantially changing the offered arms before OFUL sees them.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main() -> None:
    args = parse_args()
    scenario = image_runner.materialize_scenario(args.scenario, quick=False, horizon=args.horizon)
    calibration_csv = args.calibration_csv or args.results_dir / "image_classification_calibration.csv"
    calibration = pd.read_csv(calibration_csv)
    full_data = image_runner.load_full_dataset_for_run(
        scenario,
        seed=args.seed,
        allow_downloads=args.allow_downloads,
    )
    data = mask_image_classification_dataset(
        full_data,
        p=args.p,
        seed=image_runner.mask_seed(args.seed, args.p),
    )
    scenario = image_runner.effective_scenario(scenario, d=data.d)

    summary_rows: list[dict[str, float | int | str]] = []
    round_rows: list[dict[str, float | int | str]] = []
    for method, policy in diagnostic_policy_set(
        scenario,
        p=args.p,
        K=data.K,
        seed=args.seed,
        calibration=calibration,
    ):
        summary, per_round = diagnose_policy(method=method, policy=policy, data=data, seed=args.seed)
        summary.update(
            {
                "scenario": scenario.name,
                "p": args.p,
                "seed": args.seed,
                "T": data.T,
                "d": data.d,
                "K": data.K,
                "latent_dim": int(data.metadata.get("latent_dim", data.d)),
                "heldout_accuracy": float(data.metadata.get("heldout_accuracy", math.nan)),
            }
        )
        for row in per_round:
            row.update({"scenario": scenario.name, "p": args.p, "seed": args.seed})
        summary_rows.append(summary)
        round_rows.extend(per_round)

    stem = f"pslb_tofu_gap_{scenario.name}_p{str(args.p).replace('.', 'p')}_seed{args.seed}"
    summary_path = args.results_dir / f"{stem}_summary.csv"
    rounds_path = args.results_dir / f"{stem}_rounds.csv"
    report_path = args.results_dir / f"{stem}.md"
    write_csv(summary_path, summary_rows)
    write_csv(rounds_path, round_rows)
    write_report(
        report_path,
        scenario=scenario.name,
        p=args.p,
        seed=args.seed,
        summary_rows=summary_rows,
    )
    print(f"Wrote diagnostic summary to {summary_path}")
    print(f"Wrote per-round diagnostics to {rounds_path}")
    print(f"Wrote diagnostic report to {report_path}")


if __name__ == "__main__":
    main()
