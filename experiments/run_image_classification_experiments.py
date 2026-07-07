"""Run PSLB-style image-classification-to-bandit experiments."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from tofu_pov import (
    DatasetUnavailableError,
    FullInformationOFUL,
    MaskedPSLB,
    PSLB,
    PSLBConfig,
    RandomPolicy,
    TOFUPOV,
    TOFUPOVConfig,
    ZeroImputedOFUL,
    load_image_classification_full_dataset,
    mask_image_classification_dataset,
)
from tofu_pov.calibration import (
    candidate_label,
    mark_selected,
    oful_beta_scale_grid,
    oful_lambda_grid,
    select_fixed_rank,
    select_oful,
    select_threshold,
    threshold_grid,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRAJECTORY_COLUMNS = [
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
    "latent_dim",
    "heldout_accuracy",
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
]

SUMMARY_COLUMNS = [
    "scenario",
    "source",
    "p",
    "method",
    "rank_label",
    "configured_rank",
    "n",
    "mean_final_regret",
    "stderr_final_regret",
    "median_final_regret",
    "mean_reward",
    "mean_final_rank",
    "mean_latent_dim",
    "mean_heldout_accuracy",
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
]

RANK_SELECTION_COLUMNS = [
    "scenario",
    "source",
    "p",
    "method_family",
    "rank",
    "validation_seed",
    "final_regret",
    "representation_dim",
    "latent_dim",
    "heldout_accuracy",
]

CALIBRATION_COLUMNS = [
    "scenario",
    "source",
    "p",
    "method_family",
    "candidate_label",
    "rank",
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
    "validation_seed",
    "final_regret",
    "representation_dim",
    "latent_dim",
    "heldout_accuracy",
    "selected",
]

FAIR_METHODS = [
    "Adaptive TOFU",
    "TOFU fixed-rank best-val",
    "Zero-imputed OFUL",
    "Masked PSLB fixed-rank best-val",
    "Masked PSLB adaptive-rank",
]

REFERENCE_METHODS = [
    "Random",
    "Full-info OFUL",
    "Full-info PSLB",
]


@dataclass(frozen=True)
class ImageClassificationScenario:
    name: str
    source: str
    d: int
    T: int
    t_b: int
    max_rank: int
    fixed_rank_grid: tuple[int, ...]
    latent_dim: int | None = None
    train_fraction: float = 0.8
    max_train_examples: int = 12_000
    c_sub: float = 0.08
    R: float = 0.5
    intersection_sample_count: int = 128


SCENARIOS = {
    "digits_image_d40": ImageClassificationScenario(
        name="digits_image_d40",
        source="digits_sklearn",
        d=40,
        T=800,
        t_b=50,
        max_rank=12,
        fixed_rank_grid=(2, 4, 8, 12),
        max_train_examples=1500,
        intersection_sample_count=96,
    ),
    "mnist_openml_d300": ImageClassificationScenario(
        name="mnist_openml_d300",
        source="mnist_openml",
        d=300,
        T=1500,
        t_b=100,
        max_rank=40,
        fixed_rank_grid=(10, 20, 40),
        max_train_examples=12_000,
        intersection_sample_count=128,
    ),
    "mnist_openml_lowrank_m20_d300": ImageClassificationScenario(
        name="mnist_openml_lowrank_m20_d300",
        source="mnist_openml",
        d=300,
        T=1500,
        t_b=100,
        max_rank=40,
        fixed_rank_grid=(10, 20, 40),
        latent_dim=20,
        max_train_examples=12_000,
        intersection_sample_count=128,
    ),
}


def covariance_radius_schedule(p: float, d: int, T: int, c_sub: float) -> Callable[[int, int], float]:
    def schedule(tau: int, n_history: int) -> float:
        del tau
        log_term = math.log(2.0 * d * max(T, 2) / 0.05)
        return float(c_sub / p * math.sqrt(max(log_term, 0.0) / max(n_history, 1)))

    return schedule


def epsilon_schedule(tau: int) -> float:
    del tau
    return 0.0


def effective_scenario(
    scenario: ImageClassificationScenario,
    *,
    d: int,
) -> ImageClassificationScenario:
    max_rank = min(scenario.max_rank, d)
    fixed_grid = tuple(rank for rank in scenario.fixed_rank_grid if rank <= max_rank)
    if not fixed_grid:
        fixed_grid = (max_rank,)
    return replace(scenario, d=d, max_rank=max_rank, fixed_rank_grid=fixed_grid)


def mask_seed(seed: int, p: float) -> int:
    return int(seed + 50_000 + round(10_000 * p))


def tofu_policy(
    scenario: ImageClassificationScenario,
    *,
    p: float,
    K: int,
    seed: int,
    rank: int,
    adaptive: bool,
    rank_threshold_constant: float = 1.0,
) -> TOFUPOV:
    return TOFUPOV(
        TOFUPOVConfig(
            d=scenario.d,
            m=rank,
            K=K,
            p=p,
            lambda_reg=1.0,
            t_b=scenario.t_b,
            T=scenario.T,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.R,
            c_b=0.0,
            c_sub=scenario.c_sub,
            impute_ridge=1e-5,
            random_seed=seed + 1000,
            epsilon_schedule=epsilon_schedule,
            covariance_radius_schedule=covariance_radius_schedule(
                p,
                scenario.d,
                scenario.T,
                scenario.c_sub,
            ),
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
            warm_start_replay="first_epoch",
            rank_selection="threshold" if adaptive else "fixed",
            min_rank=1,
            max_rank=rank if adaptive else None,
            rank_threshold_constant=rank_threshold_constant,
        )
    )


def masked_pslb_policy(
    scenario: ImageClassificationScenario,
    *,
    p: float,
    K: int,
    seed: int,
    rank: int,
    adaptive: bool,
    rank_threshold_constant: float = 1.0,
) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=rank,
            K=K,
            T=scenario.T,
            p=p,
            lambda_reg=1.0,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.R,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 2000,
            rank_selection="threshold" if adaptive else "fixed",
            min_rank=1,
            max_rank=rank if adaptive else None,
            rank_threshold_constant=rank_threshold_constant,
            covariance_radius_schedule=covariance_radius_schedule(
                p,
                scenario.d,
                scenario.T,
                scenario.c_sub,
            ),
            intersection_sample_count=scenario.intersection_sample_count,
        )
    )


def full_info_pslb_policy(
    scenario: ImageClassificationScenario,
    *,
    p: float,
    K: int,
    seed: int,
    rank: int,
) -> PSLB:
    return PSLB(
        PSLBConfig(
            d=scenario.d,
            m=rank,
            K=K,
            T=scenario.T,
            p=p,
            lambda_reg=1.0,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.R,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 3000,
            arm_source="full",
            rank_selection="fixed",
            intersection_sample_count=scenario.intersection_sample_count,
        )
    )


def active_rank(policy: object) -> int | str:
    if hasattr(policy, "state_dict"):
        value = policy.state_dict().get("active_m")
        if value is not None:
            return int(value)
    if isinstance(policy, (ZeroImputedOFUL, FullInformationOFUL)):
        return int(policy.d)
    return ""


def run_trajectory(
    scenario: ImageClassificationScenario,
    p: float,
    seed: int,
    split: str,
    method: str,
    rank_label: str,
    configured_rank: int | str,
    policy: object,
    data,
    rank_threshold_constant: float | str = "",
    lambda_reg: float | str = "",
    beta_scale: float | str = "",
) -> list[dict[str, float | int | str]]:
    env = data.as_env(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)

    rows: list[dict[str, float | int | str]] = []
    cumulative = 0.0
    heldout_accuracy = float(data.metadata.get("heldout_accuracy", np.nan))
    latent_dim = int(data.metadata.get("latent_dim", data.d))
    for t in range(1, data.T + 1):
        masked_arms, masks, full_arms = env.get_round()
        action = int(policy.observe(masked_arms, masks, full_arms))
        optimal = float(env.optimal_reward(full_arms))
        selected = float(env.reward_mean(full_arms, action))
        reward = float(env.step(action))
        policy.update(reward)
        instant = optimal - selected
        cumulative += instant
        rows.append(
            {
                "scenario": scenario.name,
                "source": scenario.source,
                "p": p,
                "seed": seed,
                "split": split,
                "method": method,
                "rank_label": rank_label,
                "configured_rank": configured_rank,
                "t": t,
                "instant_regret": instant,
                "cumulative_regret": cumulative,
                "reward": reward,
                "optimal_reward": optimal,
                "active_rank": active_rank(policy),
                "representation_dim": data.d,
                "latent_dim": latent_dim,
                "heldout_accuracy": heldout_accuracy,
                "rank_threshold_constant": rank_threshold_constant,
                "lambda_reg": lambda_reg,
                "beta_scale": beta_scale,
            }
        )
    return rows


def load_full_dataset_for_run(
    scenario: ImageClassificationScenario,
    *,
    seed: int,
    allow_downloads: bool,
):
    return load_image_classification_full_dataset(
        scenario.source,
        representation_dim=scenario.latent_dim or scenario.d,
        seed=seed,
        T=scenario.T,
        ambient_dim=scenario.d if scenario.latent_dim is not None else None,
        allow_downloads=allow_downloads,
        train_fraction=scenario.train_fraction,
        max_train_examples=scenario.max_train_examples,
    )


def select_fixed_ranks(
    scenario: ImageClassificationScenario,
    *,
    p: float,
    validation_seeds: list[int],
    allow_downloads: bool,
    adaptive_threshold_grid: tuple[float, ...],
    lambda_grid: tuple[float, ...],
    beta_scale_grid: tuple[float, ...],
) -> tuple[
    int,
    int,
    float,
    float,
    tuple[float, float],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    rank_rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    family_results: dict[str, dict[int, list[float]]] = {
        "TOFU fixed-rank": defaultdict(list),
        "Masked PSLB fixed-rank": defaultdict(list),
    }
    adaptive_results: dict[str, dict[float, list[float]]] = {
        "TOFU adaptive-rank": defaultdict(list),
        "Masked PSLB adaptive-rank": defaultdict(list),
    }
    oful_results: dict[tuple[float, float], list[float]] = defaultdict(list)

    for validation_seed in validation_seeds:
        full_data = load_full_dataset_for_run(
            scenario,
            seed=validation_seed,
            allow_downloads=allow_downloads,
        )
        data = mask_image_classification_dataset(
            full_data,
            p=p,
            seed=mask_seed(validation_seed, p),
        )
        effective = effective_scenario(scenario, d=data.d)
        for rank in effective.fixed_rank_grid:
            tofu = tofu_policy(
                effective,
                p=p,
                K=data.K,
                seed=validation_seed,
                rank=rank,
                adaptive=False,
            )
            tofu_rows = run_trajectory(
                effective,
                p,
                validation_seed,
                "validation",
                "TOFU fixed-rank",
                "candidate",
                rank,
                tofu,
                data,
            )
            tofu_final = float(tofu_rows[-1]["cumulative_regret"])
            family_results["TOFU fixed-rank"][rank].append(tofu_final)
            metadata = {
                "representation_dim": data.d,
                "latent_dim": int(data.metadata.get("latent_dim", data.d)),
                "heldout_accuracy": float(data.metadata.get("heldout_accuracy", np.nan)),
            }
            rank_rows.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "method_family": "TOFU fixed-rank",
                    "rank": rank,
                    "validation_seed": validation_seed,
                    "final_regret": tofu_final,
                    **metadata,
                }
            )
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "method_family": "TOFU fixed-rank",
                    "candidate_label": candidate_label(rank=rank),
                    "rank": rank,
                    "rank_threshold_constant": "",
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": tofu_final,
                    **metadata,
                    "selected": 0,
                }
            )

            pslb = masked_pslb_policy(
                effective,
                p=p,
                K=data.K,
                seed=validation_seed,
                rank=rank,
                adaptive=False,
            )
            pslb_rows = run_trajectory(
                effective,
                p,
                validation_seed,
                "validation",
                "Masked PSLB fixed-rank",
                "candidate",
                rank,
                pslb,
                data,
            )
            pslb_final = float(pslb_rows[-1]["cumulative_regret"])
            family_results["Masked PSLB fixed-rank"][rank].append(pslb_final)
            rank_rows.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "method_family": "Masked PSLB fixed-rank",
                    "rank": rank,
                    "validation_seed": validation_seed,
                    "final_regret": pslb_final,
                    **metadata,
                }
            )
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "method_family": "Masked PSLB fixed-rank",
                    "candidate_label": candidate_label(rank=rank),
                    "rank": rank,
                    "rank_threshold_constant": "",
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": pslb_final,
                    **metadata,
                    "selected": 0,
                }
            )

        for threshold in adaptive_threshold_grid:
            adaptive_tofu = tofu_policy(
                effective,
                p=p,
                K=data.K,
                seed=validation_seed,
                rank=effective.max_rank,
                adaptive=True,
                rank_threshold_constant=threshold,
            )
            adaptive_tofu_rows = run_trajectory(
                effective,
                p,
                validation_seed,
                "validation",
                "TOFU adaptive-rank",
                "candidate",
                f"max={effective.max_rank},threshold={threshold:g}",
                adaptive_tofu,
                data,
                rank_threshold_constant=threshold,
            )
            adaptive_tofu_final = float(adaptive_tofu_rows[-1]["cumulative_regret"])
            adaptive_results["TOFU adaptive-rank"][threshold].append(adaptive_tofu_final)
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "method_family": "TOFU adaptive-rank",
                    "candidate_label": candidate_label(rank_threshold_constant=threshold),
                    "rank": effective.max_rank,
                    "rank_threshold_constant": threshold,
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": adaptive_tofu_final,
                    **metadata,
                    "selected": 0,
                }
            )

            adaptive_pslb = masked_pslb_policy(
                effective,
                p=p,
                K=data.K,
                seed=validation_seed,
                rank=effective.max_rank,
                adaptive=True,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_rows = run_trajectory(
                effective,
                p,
                validation_seed,
                "validation",
                "Masked PSLB adaptive-rank",
                "candidate",
                f"max={effective.max_rank},threshold={threshold:g}",
                adaptive_pslb,
                data,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_final = float(adaptive_pslb_rows[-1]["cumulative_regret"])
            adaptive_results["Masked PSLB adaptive-rank"][threshold].append(adaptive_pslb_final)
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "method_family": "Masked PSLB adaptive-rank",
                    "candidate_label": candidate_label(rank_threshold_constant=threshold),
                    "rank": effective.max_rank,
                    "rank_threshold_constant": threshold,
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": adaptive_pslb_final,
                    **metadata,
                    "selected": 0,
                }
            )

        for lambda_reg in lambda_grid:
            for beta_scale in beta_scale_grid:
                oful = ZeroImputedOFUL(
                    d=effective.d,
                    lambda_reg=lambda_reg,
                    S=1.0,
                    R=effective.R,
                    delta=0.05,
                    beta_scale=beta_scale,
                )
                oful_rows = run_trajectory(
                    effective,
                    p,
                    validation_seed,
                    "validation",
                    "Zero-imputed OFUL",
                    "candidate",
                    "ambient",
                    oful,
                    data,
                    lambda_reg=lambda_reg,
                    beta_scale=beta_scale,
                )
                oful_final = float(oful_rows[-1]["cumulative_regret"])
                oful_results[(lambda_reg, beta_scale)].append(oful_final)
                calibration_rows.append(
                    {
                        "scenario": scenario.name,
                        "source": scenario.source,
                        "p": p,
                        "method_family": "Zero-imputed OFUL",
                        "candidate_label": candidate_label(lambda_reg=lambda_reg, beta_scale=beta_scale),
                        "rank": effective.d,
                        "rank_threshold_constant": "",
                        "lambda_reg": lambda_reg,
                        "beta_scale": beta_scale,
                        "validation_seed": validation_seed,
                        "final_regret": oful_final,
                        **metadata,
                        "selected": 0,
                    }
                )

    selected: dict[str, int] = {}
    for family, by_rank in family_results.items():
        selected[family] = select_fixed_rank(by_rank)
    selected_adaptive = {
        family: select_threshold(by_threshold)
        for family, by_threshold in adaptive_results.items()
    }
    selected_oful = select_oful(oful_results)
    selected_by_family: dict[str, object] = {
        **selected,
        **selected_adaptive,
        "Zero-imputed OFUL": selected_oful,
    }
    return (
        selected["TOFU fixed-rank"],
        selected["Masked PSLB fixed-rank"],
        selected_adaptive["TOFU adaptive-rank"],
        selected_adaptive["Masked PSLB adaptive-rank"],
        selected_oful,
        rank_rows,
        mark_selected(calibration_rows, selected_by_family),
    )


def reporting_policies(
    scenario: ImageClassificationScenario,
    *,
    p: float,
    K: int,
    seed: int,
    tofu_rank: int,
    pslb_rank: int,
    adaptive_threshold: float,
    adaptive_pslb_threshold: float,
    oful_lambda_reg: float,
    oful_beta_scale: float,
    include_references: bool,
) -> list[tuple[str, str, int | str, object, float | str, float | str, float | str]]:
    policies: list[tuple[str, str, int | str, object, float | str, float | str, float | str]] = [
        (
            "Adaptive TOFU",
            "adaptive",
            f"max={scenario.max_rank},threshold={adaptive_threshold:g}",
            tofu_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=scenario.max_rank,
                adaptive=True,
                rank_threshold_constant=adaptive_threshold,
            ),
            adaptive_threshold,
            "",
            "",
        ),
        (
            "TOFU fixed-rank best-val",
            "best-val",
            tofu_rank,
            tofu_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=tofu_rank,
                adaptive=False,
            ),
            "",
            "",
            "",
        ),
        (
            "Zero-imputed OFUL",
            "ambient",
            scenario.d,
            ZeroImputedOFUL(
                d=scenario.d,
                lambda_reg=oful_lambda_reg,
                S=1.0,
                R=scenario.R,
                delta=0.05,
                beta_scale=oful_beta_scale,
            ),
            "",
            oful_lambda_reg,
            oful_beta_scale,
        ),
        (
            "Masked PSLB fixed-rank best-val",
            "best-val",
            pslb_rank,
            masked_pslb_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=pslb_rank,
                adaptive=False,
            ),
            "",
            "",
            "",
        ),
        (
            "Masked PSLB adaptive-rank",
            "adaptive",
            f"max={scenario.max_rank},threshold={adaptive_pslb_threshold:g}",
            masked_pslb_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=scenario.max_rank,
                adaptive=True,
                rank_threshold_constant=adaptive_pslb_threshold,
            ),
            adaptive_pslb_threshold,
            "",
            "",
        ),
    ]
    if include_references:
        policies.extend(
            [
                ("Random", "random", "", RandomPolicy(K=K, seed=seed + 4000), "", "", ""),
                (
                    "Full-info OFUL",
                    "full-info",
                    f"d={scenario.d},lambda={oful_lambda_reg:g},beta={oful_beta_scale:g}",
                    FullInformationOFUL(
                        d=scenario.d,
                        lambda_reg=oful_lambda_reg,
                        S=1.0,
                        R=scenario.R,
                        delta=0.05,
                        beta_scale=oful_beta_scale,
                    ),
                    "",
                    oful_lambda_reg,
                    oful_beta_scale,
                ),
                (
                    "Full-info PSLB",
                    "full-info",
                    pslb_rank,
                    full_info_pslb_policy(
                        scenario,
                        p=p,
                        K=K,
                        seed=seed,
                        rank=pslb_rank,
                    ),
                    "",
                    "",
                    "",
                ),
            ]
        )
    return policies


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    final_rows_by_run: dict[tuple[str, float, int, str, str], dict[str, float | int | str]] = {}
    rewards_by_group: dict[tuple[str, str, float, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["scenario"]),
            float(row["p"]),
            int(row["seed"]),
            str(row["method"]),
            str(row["rank_label"]),
        )
        group_key = (
            str(row["scenario"]),
            str(row["source"]),
            float(row["p"]),
            str(row["method"]),
            str(row["rank_label"]),
            str(row["configured_rank"]),
        )
        rewards_by_group[group_key].append(float(row["reward"]))
        if key not in final_rows_by_run or int(row["t"]) > int(final_rows_by_run[key]["t"]):
            final_rows_by_run[key] = row

    grouped: dict[tuple[str, str, float, str, str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in final_rows_by_run.values():
        grouped[
            (
                str(row["scenario"]),
                str(row["source"]),
                float(row["p"]),
                str(row["method"]),
                str(row["rank_label"]),
                str(row["configured_rank"]),
            )
        ].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (scenario, source, p, method, rank_label, configured_rank), values in sorted(grouped.items()):
        regrets = np.array([float(row["cumulative_regret"]) for row in values])
        rewards = np.array(rewards_by_group[(scenario, source, p, method, rank_label, configured_rank)])
        rank_values = [
            int(row["active_rank"])
            for row in values
            if str(row["active_rank"]) != ""
        ]
        accuracies = np.array([float(row["heldout_accuracy"]) for row in values])
        latent_dims = np.array([float(row["latent_dim"]) for row in values])
        summary.append(
            {
                "scenario": scenario,
                "source": source,
                "p": p,
                "method": method,
                "rank_label": rank_label,
                "configured_rank": configured_rank,
                "n": len(values),
                "mean_final_regret": float(np.mean(regrets)),
                "stderr_final_regret": float(np.std(regrets, ddof=1) / np.sqrt(len(regrets)))
                if len(values) > 1
                else 0.0,
                "median_final_regret": float(np.median(regrets)),
                "mean_reward": float(np.mean(rewards)),
                "mean_final_rank": float(np.mean(rank_values)) if rank_values else "",
                "mean_latent_dim": float(np.mean(latent_dims)),
                "mean_heldout_accuracy": float(np.mean(accuracies)),
                "rank_threshold_constant": values[0].get("rank_threshold_constant", ""),
                "lambda_reg": values[0].get("lambda_reg", ""),
                "beta_scale": values[0].get("beta_scale", ""),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]], columns: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, summary: list[dict[str, float | int | str]]) -> None:
    rows = [row for row in summary if row["method"] in FAIR_METHODS]
    header = "| Scenario | p | Method | Final regret mean +/- SE | Mean final rank | Latent dim | Heldout acc. |\n"
    divider = "|---|---:|---|---:|---:|---:|---:|\n"
    lines = [header, divider]
    for row in sorted(rows, key=lambda item: (str(item["scenario"]), float(item["p"]), str(item["method"]))):
        rank = row["mean_final_rank"] if row["mean_final_rank"] != "" else "-"
        lines.append(
            "| {scenario} | {p:g} | {method} | {mean:.3f} +/- {se:.3f} | {rank} | {latent:.1f} | {acc:.3f} |\n".format(
                scenario=row["scenario"],
                p=float(row["p"]),
                method=row["method"],
                mean=float(row["mean_final_regret"]),
                se=float(row["stderr_final_regret"]),
                rank=rank,
                latent=float(row["mean_latent_dim"]),
                acc=float(row["mean_heldout_accuracy"]),
            )
        )
    path.write_text("".join(lines))


def materialize_scenario(
    name: str,
    *,
    quick: bool,
    horizon: int | None,
) -> ImageClassificationScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    scenario = SCENARIOS[name]
    if quick:
        T = horizon or min(60, scenario.T)
        t_b = min(max(8, T // 8), T - 1)
        d = min(scenario.d, 40)
        latent_dim = None if scenario.latent_dim is None else min(scenario.latent_dim, d)
        max_rank = min(scenario.max_rank, 8, d)
        fixed_grid = tuple(rank for rank in scenario.fixed_rank_grid if rank <= max_rank)
        if not fixed_grid:
            fixed_grid = (max_rank,)
        return replace(
            scenario,
            d=d,
            latent_dim=latent_dim,
            T=T,
            t_b=t_b,
            max_rank=max_rank,
            fixed_rank_grid=fixed_grid,
            max_train_examples=min(scenario.max_train_examples, 1500),
            intersection_sample_count=min(scenario.intersection_sample_count, 32),
        )
    if horizon is not None:
        t_b = min(scenario.t_b, max(1, horizon - 1))
        return replace(scenario, T=horizon, t_b=t_b)
    return scenario


def parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test experiment.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario names to run.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, default=None, help="Number of reporting seeds.")
    parser.add_argument("--validation-seeds", type=int, default=None, help="Number of validation seeds.")
    parser.add_argument("--p-values", type=parse_float_list, default=None, help="Comma-separated p values.")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow OpenML MNIST download.")
    parser.add_argument("--no-references", action="store_true", help="Skip random/full-information appendix policies.")
    return parser.parse_args()


def default_scenarios(quick: bool) -> list[str]:
    if quick:
        return ["digits_image_d40"]
    return ["digits_image_d40"]


def run_scenario(
    scenario: ImageClassificationScenario,
    *,
    p_values: list[float],
    seeds: list[int],
    validation_seeds: list[int],
    allow_downloads: bool,
    include_references: bool,
    quick: bool,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    rows: list[dict[str, float | int | str]] = []
    rank_rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    selected_by_p: dict[float, tuple[int, int, float, float, tuple[float, float]]] = {}

    for p in p_values:
        (
            tofu_rank,
            pslb_rank,
            adaptive_threshold,
            adaptive_pslb_threshold,
            oful_params,
            validation_rows,
            selected_calibration_rows,
        ) = select_fixed_ranks(
            scenario,
            p=p,
            validation_seeds=validation_seeds,
            allow_downloads=allow_downloads,
            adaptive_threshold_grid=threshold_grid(quick),
            lambda_grid=oful_lambda_grid(quick),
            beta_scale_grid=oful_beta_scale_grid(quick),
        )
        selected_by_p[p] = (
            tofu_rank,
            pslb_rank,
            adaptive_threshold,
            adaptive_pslb_threshold,
            oful_params,
        )
        rank_rows.extend(validation_rows)
        calibration_rows.extend(selected_calibration_rows)

    for p in p_values:
        tofu_rank, pslb_rank, adaptive_threshold, adaptive_pslb_threshold, oful_params = selected_by_p[p]
        oful_lambda_reg, oful_beta_scale = oful_params
        for seed in seeds:
            full_data = load_full_dataset_for_run(
                scenario,
                seed=seed,
                allow_downloads=allow_downloads,
            )
            data = mask_image_classification_dataset(
                full_data,
                p=p,
                seed=mask_seed(seed, p),
            )
            effective = effective_scenario(scenario, d=data.d)
            for (
                method,
                rank_label,
                configured_rank,
                policy,
                rank_threshold_constant,
                lambda_reg,
                beta_scale,
            ) in reporting_policies(
                effective,
                p=p,
                K=data.K,
                seed=seed,
                tofu_rank=min(tofu_rank, effective.max_rank),
                pslb_rank=min(pslb_rank, effective.max_rank),
                adaptive_threshold=adaptive_threshold,
                adaptive_pslb_threshold=adaptive_pslb_threshold,
                oful_lambda_reg=oful_lambda_reg,
                oful_beta_scale=oful_beta_scale,
                include_references=include_references,
            ):
                rows.extend(
                    run_trajectory(
                        effective,
                        p,
                        seed,
                        "report",
                        method,
                        rank_label,
                        configured_rank,
                        policy,
                        data,
                        rank_threshold_constant=rank_threshold_constant,
                        lambda_reg=lambda_reg,
                        beta_scale=beta_scale,
                    )
                )
    return rows, rank_rows, calibration_rows


def main() -> None:
    args = parse_args()
    scenario_names = args.scenarios or default_scenarios(args.quick)
    p_values = args.p_values or ([0.8, 0.4] if args.quick else [0.8, 0.6, 0.4, 0.3, 0.2])
    seed_count = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    seeds = list(range(seed_count))
    validation_seed_count = (
        args.validation_seeds if args.validation_seeds is not None else (1 if args.quick else 2)
    )
    validation_seeds = [10_000 + idx for idx in range(validation_seed_count)]
    include_references = not args.no_references

    all_rows: list[dict[str, float | int | str]] = []
    all_rank_rows: list[dict[str, float | int | str]] = []
    all_calibration_rows: list[dict[str, float | int | str]] = []
    skipped: list[str] = []

    for scenario_name in scenario_names:
        try:
            scenario = materialize_scenario(scenario_name, quick=args.quick, horizon=args.horizon)
            rows, rank_rows, calibration_rows = run_scenario(
                scenario,
                p_values=p_values,
                seeds=seeds,
                validation_seeds=validation_seeds,
                allow_downloads=args.allow_downloads,
                include_references=include_references,
                quick=args.quick,
            )
        except DatasetUnavailableError as exc:
            skipped.append(f"{scenario_name}: {exc}")
            continue
        all_rows.extend(rows)
        all_rank_rows.extend(rank_rows)
        all_calibration_rows.extend(calibration_rows)

    if not all_rows:
        message = "No image-classification experiments ran successfully."
        if skipped:
            message += " Skipped: " + "; ".join(skipped)
        raise RuntimeError(message)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(all_rows)
    write_csv(args.results_dir / "image_classification_trajectories.csv", all_rows, TRAJECTORY_COLUMNS)
    write_csv(args.results_dir / "image_classification_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(
        args.results_dir / "image_classification_rank_selection.csv",
        all_rank_rows,
        RANK_SELECTION_COLUMNS,
    )
    write_csv(
        args.results_dir / "image_classification_calibration.csv",
        all_calibration_rows,
        CALIBRATION_COLUMNS,
    )
    write_markdown_table(args.results_dir / "image_classification_table.md", summary)

    print(
        "Wrote "
        f"{len(all_rows)} trajectory rows to {args.results_dir / 'image_classification_trajectories.csv'}"
    )
    print(f"Wrote summary to {args.results_dir / 'image_classification_summary.csv'}")
    print(
        "Wrote rank-selection diagnostics to "
        f"{args.results_dir / 'image_classification_rank_selection.csv'}"
    )
    print(
        "Wrote calibration diagnostics to "
        f"{args.results_dir / 'image_classification_calibration.csv'}"
    )
    if skipped:
        print("Skipped scenarios: " + "; ".join(skipped))


if __name__ == "__main__":
    main()
