"""Run low-rank CNN image-classification-to-bandit experiments."""

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
    load_cnn_image_classification_full_dataset,
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
CACHE_DIR = ROOT / "data" / "cnn_image_cache"

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
    "warm_start_replay",
    "latent_dim",
    "ambient_dim",
    "train_accuracy",
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
    "mean_ambient_dim",
    "mean_train_accuracy",
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
    "latent_dim",
    "ambient_dim",
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
    "latent_dim",
    "ambient_dim",
    "heldout_accuracy",
    "selected",
]

SELECTIONS_COLUMNS = [
    "scenario",
    "source",
    "p",
    "tofu_full_rank",
    "tofu_first_rank",
    "pslb_rank",
    "adaptive_threshold",
    "adaptive_pslb_threshold",
    "oful_lambda_reg",
    "oful_beta_scale",
]


TRAJECTORIES_FILENAME = "cnn_image_trajectories.csv"
SUMMARY_FILENAME = "cnn_image_summary.csv"
RANK_SELECTION_FILENAME = "cnn_image_rank_selection.csv"
CALIBRATION_FILENAME = "cnn_image_calibration.csv"
SELECTIONS_FILENAME = "cnn_image_selections.csv"
TABLE_FILENAME = "cnn_image_table.md"

INCREMENTAL_OUTPUT_FILENAMES = (
    TRAJECTORIES_FILENAME,
    RANK_SELECTION_FILENAME,
    CALIBRATION_FILENAME,
    SELECTIONS_FILENAME,
)

FAIR_METHODS = [
    "TOFU full-history replay fixed-rank best-val",
    "TOFU full-history replay adaptive-rank",
    "TOFU first-epoch replay fixed-rank best-val",
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
class CNNImageScenario:
    name: str
    source: str
    d: int
    latent_dim: int
    T: int
    t_b: int
    max_rank: int
    fixed_rank_grid: tuple[int, ...]
    train_fraction: float = 0.8
    max_train_examples: int = 12_000
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    c_sub: float = 0.08
    R: float = 0.5
    intersection_sample_count: int = 128


SCENARIOS = {
    "mnist_cnn_lowrank_m20_d300": CNNImageScenario(
        name="mnist_cnn_lowrank_m20_d300",
        source="mnist_openml",
        d=300,
        latent_dim=20,
        T=800,
        t_b=100,
        max_rank=40,
        fixed_rank_grid=(10, 20, 40),
        epochs=3,
        max_train_examples=12_000,
        intersection_sample_count=128,
    ),
    "mnist_cnn_product_lowrank_m4_d100": CNNImageScenario(
        name="mnist_cnn_product_lowrank_m4_d100",
        source="mnist_openml_product",
        d=100,
        latent_dim=4,
        T=5000,
        t_b=500,
        max_rank=20,
        fixed_rank_grid=(1, 2, 4, 8, 16),
        epochs=3,
        max_train_examples=12_000,
        intersection_sample_count=128,
    ),
    "mock_cnn_lowrank_m4_d20": CNNImageScenario(
        name="mock_cnn_lowrank_m4_d20",
        source="mock_cnn",
        d=20,
        latent_dim=4,
        T=30,
        t_b=6,
        max_rank=8,
        fixed_rank_grid=(2, 4, 8),
        epochs=1,
        max_train_examples=0,
        intersection_sample_count=24,
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


def mask_seed(seed: int, p: float) -> int:
    return int(seed + 50_000 + round(10_000 * p))


def effective_scenario(scenario: CNNImageScenario, *, d: int, latent_dim: int) -> CNNImageScenario:
    max_rank = min(scenario.max_rank, d)
    fixed_grid = tuple(rank for rank in scenario.fixed_rank_grid if rank <= max_rank)
    if not fixed_grid:
        fixed_grid = (max_rank,)
    return replace(
        scenario,
        d=d,
        latent_dim=latent_dim,
        max_rank=max_rank,
        fixed_rank_grid=fixed_grid,
    )


def tofu_policy(
    scenario: CNNImageScenario,
    *,
    p: float,
    K: int,
    seed: int,
    rank: int,
    adaptive: bool,
    warm_start_replay: str,
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
            warm_start_replay=warm_start_replay,
            rank_selection="threshold" if adaptive else "fixed",
            min_rank=1,
            max_rank=rank if adaptive else None,
            rank_threshold_constant=rank_threshold_constant,
        )
    )


def masked_pslb_policy(
    scenario: CNNImageScenario,
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
    scenario: CNNImageScenario,
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


def policy_replay(policy: object) -> str:
    if hasattr(policy, "state_dict"):
        return str(policy.state_dict().get("warm_start_replay", ""))
    return ""


def load_full_dataset_for_run(
    scenario: CNNImageScenario,
    *,
    seed: int,
    allow_downloads: bool,
    cache_dir: Path,
    force_retrain: bool,
):
    return load_cnn_image_classification_full_dataset(
        scenario.source,
        latent_dim=scenario.latent_dim,
        ambient_dim=scenario.d,
        seed=seed,
        T=scenario.T,
        allow_downloads=allow_downloads,
        cache_dir=cache_dir,
        train_fraction=scenario.train_fraction,
        max_train_examples=scenario.max_train_examples,
        epochs=scenario.epochs,
        batch_size=scenario.batch_size,
        learning_rate=scenario.learning_rate,
        weight_decay=scenario.weight_decay,
        hidden_dim=scenario.hidden_dim,
        force_retrain=force_retrain,
    )


def run_trajectory(
    scenario: CNNImageScenario,
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
    latent_dim = int(data.metadata.get("latent_dim", data.d))
    ambient_dim = int(data.metadata.get("ambient_dim", data.d))
    train_accuracy = float(data.metadata.get("train_accuracy", np.nan))
    heldout_accuracy = float(data.metadata.get("heldout_accuracy", np.nan))
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
                "warm_start_replay": policy_replay(policy),
                "latent_dim": latent_dim,
                "ambient_dim": ambient_dim,
                "train_accuracy": train_accuracy,
                "heldout_accuracy": heldout_accuracy,
                "rank_threshold_constant": rank_threshold_constant,
                "lambda_reg": lambda_reg,
                "beta_scale": beta_scale,
            }
        )
    return rows


def _metadata(data) -> dict[str, float | int]:
    return {
        "latent_dim": int(data.metadata.get("latent_dim", data.d)),
        "ambient_dim": int(data.metadata.get("ambient_dim", data.d)),
        "heldout_accuracy": float(data.metadata.get("heldout_accuracy", np.nan)),
    }


def _append_rank_calibration(
    *,
    rows: list[dict[str, float | int | str]],
    rank_rows: list[dict[str, float | int | str]],
    scenario: CNNImageScenario,
    p: float,
    family: str,
    rank: int,
    validation_seed: int,
    final_regret: float,
    metadata: dict[str, float | int],
) -> None:
    rank_rows.append(
        {
            "scenario": scenario.name,
            "source": scenario.source,
            "p": p,
            "method_family": family,
            "rank": rank,
            "validation_seed": validation_seed,
            "final_regret": final_regret,
            **metadata,
        }
    )
    rows.append(
        {
            "scenario": scenario.name,
            "source": scenario.source,
            "p": p,
            "method_family": family,
            "candidate_label": candidate_label(rank=rank),
            "rank": rank,
            "rank_threshold_constant": "",
            "lambda_reg": "",
            "beta_scale": "",
            "validation_seed": validation_seed,
            "final_regret": final_regret,
            **metadata,
            "selected": 0,
        }
    )


def select_calibration(
    scenario: CNNImageScenario,
    *,
    p: float,
    validation_seeds: list[int],
    allow_downloads: bool,
    cache_dir: Path,
    force_retrain: bool,
    adaptive_threshold_grid: tuple[float, ...],
    lambda_grid: tuple[float, ...],
    beta_scale_grid: tuple[float, ...],
) -> tuple[
    int,
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
    fixed_results: dict[str, dict[int, list[float]]] = {
        "TOFU full-history replay fixed-rank": defaultdict(list),
        "TOFU first-epoch replay fixed-rank": defaultdict(list),
        "Masked PSLB fixed-rank": defaultdict(list),
    }
    adaptive_results: dict[str, dict[float, list[float]]] = {
        "TOFU full-history replay adaptive-rank": defaultdict(list),
        "Masked PSLB adaptive-rank": defaultdict(list),
    }
    oful_results: dict[tuple[float, float], list[float]] = defaultdict(list)

    for validation_seed in validation_seeds:
        full_data = load_full_dataset_for_run(
            scenario,
            seed=validation_seed,
            allow_downloads=allow_downloads,
            cache_dir=cache_dir,
            force_retrain=force_retrain,
        )
        data = mask_image_classification_dataset(
            full_data,
            p=p,
            seed=mask_seed(validation_seed, p),
        )
        effective = effective_scenario(
            scenario,
            d=data.d,
            latent_dim=int(data.metadata.get("latent_dim", scenario.latent_dim)),
        )
        metadata = _metadata(data)

        for rank in effective.fixed_rank_grid:
            fixed_specs = [
                (
                    "TOFU full-history replay fixed-rank",
                    tofu_policy(
                        effective,
                        p=p,
                        K=data.K,
                        seed=validation_seed,
                        rank=rank,
                        adaptive=False,
                        warm_start_replay="full_history_every_epoch",
                    ),
                ),
                (
                    "TOFU first-epoch replay fixed-rank",
                    tofu_policy(
                        effective,
                        p=p,
                        K=data.K,
                        seed=validation_seed,
                        rank=rank,
                        adaptive=False,
                        warm_start_replay="first_epoch",
                    ),
                ),
                (
                    "Masked PSLB fixed-rank",
                    masked_pslb_policy(
                        effective,
                        p=p,
                        K=data.K,
                        seed=validation_seed,
                        rank=rank,
                        adaptive=False,
                    ),
                ),
            ]
            for family, policy in fixed_specs:
                trajectory = run_trajectory(
                    effective,
                    p,
                    validation_seed,
                    "validation",
                    family,
                    "candidate",
                    rank,
                    policy,
                    data,
                )
                final = float(trajectory[-1]["cumulative_regret"])
                fixed_results[family][rank].append(final)
                _append_rank_calibration(
                    rows=calibration_rows,
                    rank_rows=rank_rows,
                    scenario=scenario,
                    p=p,
                    family=family,
                    rank=rank,
                    validation_seed=validation_seed,
                    final_regret=final,
                    metadata=metadata,
                )

        for threshold in adaptive_threshold_grid:
            adaptive_specs = [
                (
                    "TOFU full-history replay adaptive-rank",
                    tofu_policy(
                        effective,
                        p=p,
                        K=data.K,
                        seed=validation_seed,
                        rank=effective.max_rank,
                        adaptive=True,
                        warm_start_replay="full_history_every_epoch",
                        rank_threshold_constant=threshold,
                    ),
                ),
                (
                    "Masked PSLB adaptive-rank",
                    masked_pslb_policy(
                        effective,
                        p=p,
                        K=data.K,
                        seed=validation_seed,
                        rank=effective.max_rank,
                        adaptive=True,
                        rank_threshold_constant=threshold,
                    ),
                ),
            ]
            for family, policy in adaptive_specs:
                trajectory = run_trajectory(
                    effective,
                    p,
                    validation_seed,
                    "validation",
                    family,
                    "candidate",
                    f"max={effective.max_rank},threshold={threshold:g}",
                    policy,
                    data,
                    rank_threshold_constant=threshold,
                )
                final = float(trajectory[-1]["cumulative_regret"])
                adaptive_results[family][threshold].append(final)
                calibration_rows.append(
                    {
                        "scenario": scenario.name,
                        "source": scenario.source,
                        "p": p,
                        "method_family": family,
                        "candidate_label": candidate_label(rank_threshold_constant=threshold),
                        "rank": effective.max_rank,
                        "rank_threshold_constant": threshold,
                        "lambda_reg": "",
                        "beta_scale": "",
                        "validation_seed": validation_seed,
                        "final_regret": final,
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
                trajectory = run_trajectory(
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
                final = float(trajectory[-1]["cumulative_regret"])
                oful_results[(lambda_reg, beta_scale)].append(final)
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
                        "final_regret": final,
                        **metadata,
                        "selected": 0,
                    }
                )

    selected_fixed = {
        family: select_fixed_rank(by_rank)
        for family, by_rank in fixed_results.items()
    }
    selected_adaptive = {
        family: select_threshold(by_threshold)
        for family, by_threshold in adaptive_results.items()
    }
    selected_oful = select_oful(oful_results)
    selected_by_family: dict[str, object] = {
        **selected_fixed,
        **selected_adaptive,
        "Zero-imputed OFUL": selected_oful,
    }
    return (
        selected_fixed["TOFU full-history replay fixed-rank"],
        selected_fixed["TOFU first-epoch replay fixed-rank"],
        selected_fixed["Masked PSLB fixed-rank"],
        selected_adaptive["TOFU full-history replay adaptive-rank"],
        selected_adaptive["Masked PSLB adaptive-rank"],
        selected_oful,
        rank_rows,
        mark_selected(calibration_rows, selected_by_family),
    )


def reporting_policies(
    scenario: CNNImageScenario,
    *,
    p: float,
    K: int,
    seed: int,
    tofu_full_rank: int,
    tofu_first_rank: int,
    pslb_rank: int,
    adaptive_threshold: float,
    adaptive_pslb_threshold: float,
    oful_lambda_reg: float,
    oful_beta_scale: float,
    include_references: bool,
) -> list[tuple[str, str, int | str, object, float | str, float | str, float | str]]:
    policies: list[tuple[str, str, int | str, object, float | str, float | str, float | str]] = [
        (
            "TOFU full-history replay fixed-rank best-val",
            "best-val",
            tofu_full_rank,
            tofu_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=tofu_full_rank,
                adaptive=False,
                warm_start_replay="full_history_every_epoch",
            ),
            "",
            "",
            "",
        ),
        (
            "TOFU full-history replay adaptive-rank",
            "adaptive",
            f"max={scenario.max_rank},threshold={adaptive_threshold:g}",
            tofu_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=scenario.max_rank,
                adaptive=True,
                warm_start_replay="full_history_every_epoch",
                rank_threshold_constant=adaptive_threshold,
            ),
            adaptive_threshold,
            "",
            "",
        ),
        (
            "TOFU first-epoch replay fixed-rank best-val",
            "best-val",
            tofu_first_rank,
            tofu_policy(
                scenario,
                p=p,
                K=K,
                seed=seed,
                rank=tofu_first_rank,
                adaptive=False,
                warm_start_replay="first_epoch",
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
        latent_dims = np.array([float(row["latent_dim"]) for row in values])
        ambient_dims = np.array([float(row["ambient_dim"]) for row in values])
        train_acc = np.array([float(row["train_accuracy"]) for row in values])
        heldout_acc = np.array([float(row["heldout_accuracy"]) for row in values])
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
                "mean_ambient_dim": float(np.mean(ambient_dims)),
                "mean_train_accuracy": float(np.mean(train_acc)),
                "mean_heldout_accuracy": float(np.mean(heldout_acc)),
                "rank_threshold_constant": values[0].get("rank_threshold_constant", ""),
                "lambda_reg": values[0].get("lambda_reg", ""),
                "beta_scale": values[0].get("beta_scale", ""),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]], columns: list[str]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, float | int | str]], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_completed_trajectory_counts(path: Path) -> dict[tuple[str, float, int, str, str], int]:
    counts: dict[tuple[str, float, int, str, str], int] = defaultdict(int)
    for row in read_csv_rows(path):
        if row.get("split") != "report":
            continue
        key = (
            row["scenario"],
            float(row["p"]),
            int(row["seed"]),
            row["method"],
            row["rank_label"],
        )
        counts[key] += 1
    return counts


def load_existing_selections(path: Path) -> dict[
    tuple[str, float],
    tuple[int, int, int, float, float, tuple[float, float]],
]:
    selections: dict[
        tuple[str, float],
        tuple[int, int, int, float, float, tuple[float, float]],
    ] = {}
    for row in read_csv_rows(path):
        key = (row["scenario"], float(row["p"]))
        selections[key] = (
            int(row["tofu_full_rank"]),
            int(row["tofu_first_rank"]),
            int(row["pslb_rank"]),
            float(row["adaptive_threshold"]),
            float(row["adaptive_pslb_threshold"]),
            (float(row["oful_lambda_reg"]), float(row["oful_beta_scale"])),
        )
    return selections


def write_markdown_table(path: Path, summary: list[dict[str, float | int | str]]) -> None:
    rows = [row for row in summary if row["method"] in FAIR_METHODS]
    lines = [
        "| Scenario | p | Method | Final regret mean +/- SE | Mean final rank | m | d | Heldout acc. |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in sorted(rows, key=lambda item: (str(item["scenario"]), float(item["p"]), str(item["method"]))):
        rank = row["mean_final_rank"] if row["mean_final_rank"] != "" else "-"
        lines.append(
            "| {scenario} | {p:g} | {method} | {mean:.3f} +/- {se:.3f} | {rank} | {latent:.1f} | {ambient:.1f} | {acc:.3f} |\n".format(
                scenario=row["scenario"],
                p=float(row["p"]),
                method=row["method"],
                mean=float(row["mean_final_regret"]),
                se=float(row["stderr_final_regret"]),
                rank=rank,
                latent=float(row["mean_latent_dim"]),
                ambient=float(row["mean_ambient_dim"]),
                acc=float(row["mean_heldout_accuracy"]),
            )
        )
    path.write_text("".join(lines))


def materialize_scenario(
    name: str,
    *,
    quick: bool,
    horizon: int | None,
    epochs: int | None,
    max_train_examples: int | None,
) -> CNNImageScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    scenario = SCENARIOS[name]
    if quick:
        T = horizon or min(30, scenario.T)
        t_b = min(max(4, T // 5), T - 1)
        if scenario.source == "mock_cnn":
            return replace(scenario, T=T, t_b=t_b)
        d = min(scenario.d, 60)
        latent_dim = min(scenario.latent_dim, 8, d)
        max_rank = min(scenario.max_rank, 8, d)
        fixed_grid = tuple(rank for rank in scenario.fixed_rank_grid if rank <= max_rank)
        if not fixed_grid:
            fixed_grid = (max(1, min(latent_dim, max_rank)),)
        scenario = replace(
            scenario,
            d=d,
            latent_dim=latent_dim,
            T=T,
            t_b=t_b,
            max_rank=max_rank,
            fixed_rank_grid=fixed_grid,
            epochs=epochs or 1,
            max_train_examples=max_train_examples or min(scenario.max_train_examples, 800),
            intersection_sample_count=min(scenario.intersection_sample_count, 32),
        )
    else:
        if horizon is not None:
            scenario = replace(scenario, T=horizon, t_b=min(scenario.t_b, max(1, horizon - 1)))
        if epochs is not None:
            scenario = replace(scenario, epochs=epochs)
        if max_train_examples is not None:
            scenario = replace(scenario, max_train_examples=max_train_examples)
    return scenario


def parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test experiment.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario names to run.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--seeds", type=int, default=None, help="Number of reporting seeds.")
    parser.add_argument("--validation-seeds", type=int, default=None, help="Number of validation seeds.")
    parser.add_argument("--p-values", type=parse_float_list, default=None, help="Comma-separated p values.")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon.")
    parser.add_argument("--epochs", type=int, default=None, help="Override CNN training epochs.")
    parser.add_argument("--max-train-examples", type=int, default=None, help="Override CNN train subset size.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow OpenML MNIST download.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached CNN features.")
    parser.add_argument("--no-references", action="store_true", help="Skip random/full-information appendix policies.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse incremental CSVs in --results-dir: skip already-completed "
            "(scenario, p, seed, method) trajectories and per-p calibration. "
            "Use the same scenario/horizon/epochs args as the original run."
        ),
    )
    return parser.parse_args()


def default_scenarios(quick: bool) -> list[str]:
    if quick:
        return ["mock_cnn_lowrank_m4_d20"]
    return ["mnist_cnn_lowrank_m20_d300"]


def clean_partial_trajectories(
    path: Path,
    scenario_T_by_name: dict[str, int],
) -> dict[tuple[str, float, int, str, str], int]:
    """Drop rows for any (scenario,p,seed,method,rank_label) key with 0<count<T.

    Returns the post-cleanup completed counts (keys with count >= T).
    """
    rows = read_csv_rows(path)
    if not rows:
        return defaultdict(int)

    grouped: dict[tuple[str, float, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["scenario"],
            float(row["p"]),
            int(row["seed"]),
            row["method"],
            row["rank_label"],
        )
        grouped[key].append(row)

    keep_rows: list[dict[str, str]] = []
    completed: dict[tuple[str, float, int, str, str], int] = defaultdict(int)
    dropped_keys: list[tuple[str, float, int, str, str]] = []
    for key, key_rows in grouped.items():
        scenario_name = key[0]
        expected_T = scenario_T_by_name.get(scenario_name)
        count = len(key_rows)
        if expected_T is not None and count < expected_T:
            dropped_keys.append(key)
            continue
        keep_rows.extend(key_rows)
        completed[key] = count

    if dropped_keys:
        print(
            f"[resume] dropping {len(dropped_keys)} partial trajectory key(s) from {path.name}",
            flush=True,
        )
        for key in dropped_keys:
            print(f"  dropped: {key}", flush=True)
        if keep_rows:
            write_csv(path, keep_rows, TRAJECTORY_COLUMNS)
        else:
            path.unlink()
    return completed


def run_scenario(
    scenario: CNNImageScenario,
    *,
    p_values: list[float],
    seeds: list[int],
    validation_seeds: list[int],
    allow_downloads: bool,
    cache_dir: Path,
    force_retrain: bool,
    include_references: bool,
    quick: bool,
    results_dir: Path,
    completed_trajectory_counts: dict[tuple[str, float, int, str, str], int],
    existing_selections: dict[tuple[str, float], tuple[int, int, int, float, float, tuple[float, float]]],
) -> None:
    trajectories_path = results_dir / TRAJECTORIES_FILENAME
    rank_path = results_dir / RANK_SELECTION_FILENAME
    calibration_path = results_dir / CALIBRATION_FILENAME
    selections_path = results_dir / SELECTIONS_FILENAME

    selected_by_p: dict[float, tuple[int, int, int, float, float, tuple[float, float]]] = {}

    print(f"[scenario] {scenario.name} (T={scenario.T}, p_values={p_values})", flush=True)

    for p in p_values:
        sel_key = (scenario.name, p)
        if sel_key in existing_selections:
            print(f"  [calibration] p={p:g} -> resume from selections", flush=True)
            selected_by_p[p] = existing_selections[sel_key]
            continue

        print(f"  [calibration] p={p:g} (validation_seeds={validation_seeds})", flush=True)
        (
            tofu_full_rank,
            tofu_first_rank,
            pslb_rank,
            adaptive_threshold,
            adaptive_pslb_threshold,
            oful_params,
            validation_rows,
            selected_calibration_rows,
        ) = select_calibration(
            scenario,
            p=p,
            validation_seeds=validation_seeds,
            allow_downloads=allow_downloads,
            cache_dir=cache_dir,
            force_retrain=force_retrain,
            adaptive_threshold_grid=threshold_grid(quick),
            lambda_grid=oful_lambda_grid(quick),
            beta_scale_grid=oful_beta_scale_grid(quick),
        )
        selection = (
            tofu_full_rank,
            tofu_first_rank,
            pslb_rank,
            adaptive_threshold,
            adaptive_pslb_threshold,
            oful_params,
        )
        selected_by_p[p] = selection
        existing_selections[sel_key] = selection

        append_csv(rank_path, validation_rows, RANK_SELECTION_COLUMNS)
        append_csv(calibration_path, selected_calibration_rows, CALIBRATION_COLUMNS)
        append_csv(
            selections_path,
            [
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "tofu_full_rank": tofu_full_rank,
                    "tofu_first_rank": tofu_first_rank,
                    "pslb_rank": pslb_rank,
                    "adaptive_threshold": adaptive_threshold,
                    "adaptive_pslb_threshold": adaptive_pslb_threshold,
                    "oful_lambda_reg": oful_params[0],
                    "oful_beta_scale": oful_params[1],
                }
            ],
            SELECTIONS_COLUMNS,
        )
        print(
            f"    selected: tofu_full={tofu_full_rank}, tofu_first={tofu_first_rank}, "
            f"pslb={pslb_rank}, threshold_tofu={adaptive_threshold:g}, "
            f"threshold_pslb={adaptive_pslb_threshold:g}, "
            f"oful_lambda={oful_params[0]:g}, oful_beta={oful_params[1]:g}",
            flush=True,
        )

    for p in p_values:
        (
            tofu_full_rank,
            tofu_first_rank,
            pslb_rank,
            adaptive_threshold,
            adaptive_pslb_threshold,
            oful_params,
        ) = selected_by_p[p]
        oful_lambda_reg, oful_beta_scale = oful_params
        for seed in seeds:
            print(f"  [report] p={p:g} seed={seed}", flush=True)
            full_data = load_full_dataset_for_run(
                scenario,
                seed=seed,
                allow_downloads=allow_downloads,
                cache_dir=cache_dir,
                force_retrain=force_retrain,
            )
            data = mask_image_classification_dataset(
                full_data,
                p=p,
                seed=mask_seed(seed, p),
            )
            effective = effective_scenario(
                scenario,
                d=data.d,
                latent_dim=int(data.metadata.get("latent_dim", scenario.latent_dim)),
            )
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
                tofu_full_rank=min(tofu_full_rank, effective.max_rank),
                tofu_first_rank=min(tofu_first_rank, effective.max_rank),
                pslb_rank=min(pslb_rank, effective.max_rank),
                adaptive_threshold=adaptive_threshold,
                adaptive_pslb_threshold=adaptive_pslb_threshold,
                oful_lambda_reg=oful_lambda_reg,
                oful_beta_scale=oful_beta_scale,
                include_references=include_references,
            ):
                key = (scenario.name, p, seed, method, rank_label)
                if completed_trajectory_counts.get(key, 0) >= effective.T:
                    print(f"    [skip] {method} (already complete)", flush=True)
                    continue
                if key in completed_trajectory_counts:
                    print(
                        f"    [warn] partial rows for {method} ("
                        f"{completed_trajectory_counts[key]}/{effective.T}); rerunning method",
                        flush=True,
                    )
                print(f"    [run] {method}", flush=True)
                trajectory = run_trajectory(
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
                append_csv(trajectories_path, trajectory, TRAJECTORY_COLUMNS)
                completed_trajectory_counts[key] = effective.T


def main() -> None:
    args = parse_args()
    scenario_names = args.scenarios or default_scenarios(args.quick)
    p_values = args.p_values or ([0.2] if args.quick else [0.3, 0.2, 0.15, 0.1, 0.075])
    seed_count = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    seeds = list(range(seed_count))
    validation_seed_count = (
        args.validation_seeds if args.validation_seeds is not None else (1 if args.quick else 2)
    )
    validation_seeds = [10_000 + idx for idx in range(validation_seed_count)]
    include_references = not args.no_references

    args.results_dir.mkdir(parents=True, exist_ok=True)

    materialized: dict[str, CNNImageScenario] = {}
    skipped: list[str] = []
    for scenario_name in scenario_names:
        try:
            materialized[scenario_name] = materialize_scenario(
                scenario_name,
                quick=args.quick,
                horizon=args.horizon,
                epochs=args.epochs,
                max_train_examples=args.max_train_examples,
            )
        except ValueError as exc:
            skipped.append(f"{scenario_name}: {exc}")

    existing_files = [
        args.results_dir / name
        for name in INCREMENTAL_OUTPUT_FILENAMES
        if (args.results_dir / name).exists()
    ]
    if existing_files and not args.resume:
        names = ", ".join(p.name for p in existing_files)
        raise SystemExit(
            f"Refusing to overwrite existing CSVs in {args.results_dir}: {names}. "
            f"Pass --resume to continue, or remove the directory to start fresh."
        )

    scenario_T_by_name = {name: scenario.T for name, scenario in materialized.items()}
    if args.resume:
        completed_trajectory_counts = clean_partial_trajectories(
            args.results_dir / TRAJECTORIES_FILENAME,
            scenario_T_by_name,
        )
        existing_selections = load_existing_selections(args.results_dir / SELECTIONS_FILENAME)
        if completed_trajectory_counts:
            print(
                f"[resume] {len(completed_trajectory_counts)} completed trajectory key(s) "
                f"loaded from {TRAJECTORIES_FILENAME}",
                flush=True,
            )
        if existing_selections:
            print(
                f"[resume] {len(existing_selections)} cached calibration selection(s) "
                f"loaded from {SELECTIONS_FILENAME}",
                flush=True,
            )
    else:
        completed_trajectory_counts = defaultdict(int)
        existing_selections = {}

    for scenario_name in scenario_names:
        if scenario_name not in materialized:
            continue
        scenario = materialized[scenario_name]
        try:
            run_scenario(
                scenario,
                p_values=p_values,
                seeds=seeds,
                validation_seeds=validation_seeds,
                allow_downloads=args.allow_downloads,
                cache_dir=args.cache_dir,
                force_retrain=args.force_retrain,
                include_references=include_references,
                quick=args.quick,
                results_dir=args.results_dir,
                completed_trajectory_counts=completed_trajectory_counts,
                existing_selections=existing_selections,
            )
        except DatasetUnavailableError as exc:
            skipped.append(f"{scenario_name}: {exc}")
            continue

    trajectories_path = args.results_dir / TRAJECTORIES_FILENAME
    final_rows = read_csv_rows(trajectories_path)
    if not final_rows:
        message = "No CNN image experiments ran successfully."
        if skipped:
            message += " Skipped: " + "; ".join(skipped)
        raise RuntimeError(message)

    summary = summarize(final_rows)
    write_csv(args.results_dir / SUMMARY_FILENAME, summary, SUMMARY_COLUMNS)
    write_markdown_table(args.results_dir / TABLE_FILENAME, summary)

    print(f"Wrote {len(final_rows)} trajectory rows to {trajectories_path}", flush=True)
    print(f"Wrote summary to {args.results_dir / SUMMARY_FILENAME}", flush=True)
    print(
        f"Wrote rank-selection diagnostics to {args.results_dir / RANK_SELECTION_FILENAME}",
        flush=True,
    )
    print(
        f"Wrote calibration diagnostics to {args.results_dir / CALIBRATION_FILENAME}",
        flush=True,
    )
    if skipped:
        print("Skipped scenarios: " + "; ".join(skipped), flush=True)


if __name__ == "__main__":
    main()
