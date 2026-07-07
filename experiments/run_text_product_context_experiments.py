"""Run product-context text classification bandit experiments."""

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
    load_text_product_context_full_dataset,
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
RESULTS_DIR = ROOT / "results" / "text_product_context"
CACHE_DIR = ROOT / "data" / "text_product_cache"

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

TRAJECTORIES_FILENAME = "text_product_trajectories.csv"
SUMMARY_FILENAME = "text_product_summary.csv"
CALIBRATION_FILENAME = "text_product_calibration.csv"
RANK_SELECTION_FILENAME = "text_product_rank_selection.csv"
SELECTIONS_FILENAME = "text_product_selections.csv"
TABLE_FILENAME = "text_product_table.md"

FAIR_METHODS = [
    "TOFU full-history replay fixed-rank best-val",
    "TOFU full-history replay adaptive-rank",
    "TOFU first-epoch replay fixed-rank best-val",
    "Zero-imputed OFUL",
    "Masked PSLB fixed-rank best-val",
    "Masked PSLB adaptive-rank",
]

PILOT_METHODS = [
    "TOFU full-history replay fixed-rank best-val",
    "TOFU full-history replay adaptive-rank",
    "Zero-imputed OFUL",
    "Masked PSLB fixed-rank best-val",
]


@dataclass(frozen=True)
class TextProductScenario:
    name: str
    source: str
    d: int
    latent_dim: int
    T: int
    t_b: int
    max_rank: int
    fixed_rank_grid: tuple[int, ...]
    train_fraction: float = 0.8
    max_features: int = 20_000
    min_df: int = 2
    logistic_C: float = 1.0
    max_iter: int = 1_000
    lift_type: str = "dense_random"
    nuisance_mode: str = "raw"
    nuisance_scale: float = 0.0
    nuisance_spectral_ratio: float = 0.0
    c_sub: float = 0.08
    R: float = 0.5
    intersection_sample_count: int = 128


SCENARIOS = {
    "text20news4_product_m20_d200": TextProductScenario(
        name="text20news4_product_m20_d200",
        source="20newsgroups4",
        d=200,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        intersection_sample_count=128,
    ),
    "text20news4_product_m20_d500": TextProductScenario(
        name="text20news4_product_m20_d500",
        source="20newsgroups4",
        d=500,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        intersection_sample_count=128,
    ),
    "text20news4_product_grouped_m20_d500": TextProductScenario(
        name="text20news4_product_grouped_m20_d500",
        source="20newsgroups4",
        d=500,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        lift_type="grouped",
        intersection_sample_count=128,
    ),
    "text20news4_product_spiked_s025_m20_d1000": TextProductScenario(
        name="text20news4_product_spiked_s025_m20_d1000",
        source="20newsgroups4",
        d=1000,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_scale=0.25,
        intersection_sample_count=128,
    ),
    "text20news4_product_spiked_s05_m20_d1000": TextProductScenario(
        name="text20news4_product_spiked_s05_m20_d1000",
        source="20newsgroups4",
        d=1000,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_scale=0.5,
        intersection_sample_count=128,
    ),
    "text20news4_product_spiked_s10_m20_d1000": TextProductScenario(
        name="text20news4_product_spiked_s10_m20_d1000",
        source="20newsgroups4",
        d=1000,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_scale=1.0,
        intersection_sample_count=128,
    ),
    "text20news4_product_spectral_tail_r025_m20_d1000": TextProductScenario(
        name="text20news4_product_spectral_tail_r025_m20_d1000",
        source="20newsgroups4",
        d=1000,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.25,
        intersection_sample_count=128,
    ),
    "text20news4_product_spectral_tail_r025_m20_d800": TextProductScenario(
        name="text20news4_product_spectral_tail_r025_m20_d800",
        source="20newsgroups4",
        d=800,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.25,
        intersection_sample_count=128,
    ),
    "text20news4_product_spectral_tail_r025_m20_d600": TextProductScenario(
        name="text20news4_product_spectral_tail_r025_m20_d600",
        source="20newsgroups4",
        d=600,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.25,
        intersection_sample_count=128,
    ),
    "text20news4_product_spectral_tail_r025_m20_d400": TextProductScenario(
        name="text20news4_product_spectral_tail_r025_m20_d400",
        source="20newsgroups4",
        d=400,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.25,
        intersection_sample_count=128,
    ),
    "text20news4_product_spectral_tail_r05_m20_d1000": TextProductScenario(
        name="text20news4_product_spectral_tail_r05_m20_d1000",
        source="20newsgroups4",
        d=1000,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.5,
        intersection_sample_count=128,
    ),
    "text20news4_product_spectral_tail_r075_m20_d1000": TextProductScenario(
        name="text20news4_product_spectral_tail_r075_m20_d1000",
        source="20newsgroups4",
        d=1000,
        latent_dim=20,
        T=4000,
        t_b=400,
        max_rank=40,
        fixed_rank_grid=(5, 10, 20, 40),
        max_features=20_000,
        nuisance_mode="spectral_tail",
        nuisance_spectral_ratio=0.75,
        intersection_sample_count=128,
    ),
    "text20news20_product_m40_d300": TextProductScenario(
        name="text20news20_product_m40_d300",
        source="20newsgroups20",
        d=300,
        latent_dim=40,
        T=5000,
        t_b=500,
        max_rank=80,
        fixed_rank_grid=(10, 20, 40, 80),
        max_features=30_000,
        intersection_sample_count=128,
    ),
    "text20news20_product_grouped_m40_d300": TextProductScenario(
        name="text20news20_product_grouped_m40_d300",
        source="20newsgroups20",
        d=300,
        latent_dim=40,
        T=5000,
        t_b=500,
        max_rank=80,
        fixed_rank_grid=(10, 20, 40, 80),
        max_features=30_000,
        lift_type="grouped",
        intersection_sample_count=128,
    ),
    "mock_text_product_m4_d20": TextProductScenario(
        name="mock_text_product_m4_d20",
        source="mock_text",
        d=20,
        latent_dim=4,
        T=30,
        t_b=6,
        max_rank=8,
        fixed_rank_grid=(2, 4, 8),
        max_features=100,
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
    return int(seed + 70_000 + round(10_000 * p))


def effective_scenario(
    scenario: TextProductScenario,
    *,
    d: int,
    latent_dim: int,
) -> TextProductScenario:
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
    scenario: TextProductScenario,
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
    scenario: TextProductScenario,
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
    scenario: TextProductScenario,
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
    if isinstance(policy, (ZeroImputedOFUL, FullInformationOFUL)):
        return policy.d
    if hasattr(policy, "state_dict"):
        state = policy.state_dict()
        value = state.get("active_m")
        if value is not None:
            return int(value)
    return ""


def policy_replay(policy: object) -> str:
    if hasattr(policy, "state_dict"):
        return str(policy.state_dict().get("warm_start_replay", ""))
    return ""


def load_full_dataset_for_run(
    scenario: TextProductScenario,
    *,
    seed: int,
    allow_downloads: bool,
    cache_dir: Path,
    force_rebuild: bool,
):
    return load_text_product_context_full_dataset(
        scenario.source,
        latent_dim=scenario.latent_dim,
        ambient_dim=scenario.d,
        seed=seed,
        T=scenario.T,
        allow_downloads=allow_downloads,
        cache_dir=cache_dir,
        train_fraction=scenario.train_fraction,
        max_features=scenario.max_features,
        min_df=scenario.min_df,
        logistic_C=scenario.logistic_C,
        max_iter=scenario.max_iter,
        lift_type=scenario.lift_type,
        nuisance_mode=scenario.nuisance_mode,
        nuisance_scale=scenario.nuisance_scale,
        nuisance_spectral_ratio=scenario.nuisance_spectral_ratio,
        force_rebuild=force_rebuild,
    )


def run_final_regret(
    scenario: TextProductScenario,
    *,
    p: float,
    seed: int,
    policy: object,
    data,
) -> float:
    env = data.as_env(seed=seed + 9000)
    cumulative = 0.0
    for _ in range(1, data.T + 1):
        masked_arms, masks, full_arms = env.get_round()
        action = int(policy.observe(masked_arms, masks, full_arms))
        optimal = float(env.optimal_reward(full_arms))
        selected = float(env.reward_mean(full_arms, action))
        reward = float(env.step(action))
        policy.update(reward)
        cumulative += optimal - selected
    return cumulative


def run_trajectory(
    scenario: TextProductScenario,
    *,
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
    env = data.as_env(seed=seed + 9000)
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
    scenario: TextProductScenario,
    p: float,
    family: str,
    rank: int,
    validation_seed: int,
    final_regret: float,
    metadata: dict[str, float | int],
) -> None:
    base = {
        "scenario": scenario.name,
        "source": scenario.source,
        "p": p,
        "method_family": family,
        "rank": rank,
        "validation_seed": validation_seed,
        "final_regret": final_regret,
        **metadata,
    }
    rank_rows.append(base)
    rows.append(
        {
            **base,
            "candidate_label": candidate_label(rank=rank),
            "rank_threshold_constant": "",
            "lambda_reg": "",
            "beta_scale": "",
            "selected": 0,
        }
    )


def select_calibration(
    scenario: TextProductScenario,
    *,
    p: float,
    validation_seeds: list[int],
    allow_downloads: bool,
    cache_dir: Path,
    force_rebuild: bool,
    adaptive_threshold_grid: tuple[float, ...],
    lambda_grid: tuple[float, ...],
    beta_scale_grid: tuple[float, ...],
):
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
            force_rebuild=force_rebuild,
        )
        data = mask_image_classification_dataset(full_data, p=p, seed=mask_seed(validation_seed, p))
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
                final = run_final_regret(
                    effective,
                    p=p,
                    seed=validation_seed,
                    policy=policy,
                    data=data,
                )
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
                final = run_final_regret(
                    effective,
                    p=p,
                    seed=validation_seed,
                    policy=policy,
                    data=data,
                )
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
                final = run_final_regret(
                    effective,
                    p=p,
                    seed=validation_seed,
                    policy=oful,
                    data=data,
                )
                oful_results[(lambda_reg, beta_scale)].append(final)
                calibration_rows.append(
                    {
                        "scenario": scenario.name,
                        "source": scenario.source,
                        "p": p,
                        "method_family": "Zero-imputed OFUL",
                        "candidate_label": candidate_label(
                            lambda_reg=lambda_reg,
                            beta_scale=beta_scale,
                        ),
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

    selected_fixed = {family: select_fixed_rank(by_rank) for family, by_rank in fixed_results.items()}
    selected_adaptive = {
        family: select_threshold(by_threshold) for family, by_threshold in adaptive_results.items()
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
    scenario: TextProductScenario,
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
):
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
        rank_values = [int(row["active_rank"]) for row in values if str(row["active_rank"]) != ""]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, summary: list[dict[str, float | int | str]]) -> None:
    lines = [
        "| Scenario | p | Method | Final regret mean +/- SE | Mean final rank | m | d | Heldout acc. |\n",
        "|---|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        rank = row["mean_final_rank"]
        rank_text = "" if rank == "" else f"{float(rank):.1f}"
        lines.append(
            "| {scenario} | {p:g} | {method} | {mean:.3f} +/- {se:.3f} | {rank} | {latent:.1f} | {ambient:.1f} | {acc:.3f} |\n".format(
                scenario=row["scenario"],
                p=float(row["p"]),
                method=row["method"],
                mean=float(row["mean_final_regret"]),
                se=float(row["stderr_final_regret"]),
                rank=rank_text,
                latent=float(row["mean_latent_dim"]),
                ambient=float(row["mean_ambient_dim"]),
                acc=float(row["mean_heldout_accuracy"]),
            )
        )
    path.write_text("".join(lines))


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_scenario_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def materialize_scenario(
    name: str,
    *,
    quick: bool,
    horizon: int | None,
) -> TextProductScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    scenario = SCENARIOS[name]
    if quick:
        T = horizon or min(30, scenario.T)
        t_b = min(max(4, T // 5), T - 1)
        if scenario.source == "mock_text":
            return replace(scenario, T=T, t_b=t_b)
        d = min(scenario.d, 60)
        latent_dim = min(scenario.latent_dim, 8, d)
        max_rank = min(scenario.max_rank, 8, d)
        fixed_grid = tuple(rank for rank in scenario.fixed_rank_grid if rank <= max_rank)
        if not fixed_grid:
            fixed_grid = (max(1, min(latent_dim, max_rank)),)
        return replace(
            scenario,
            d=d,
            latent_dim=latent_dim,
            T=T,
            t_b=t_b,
            max_rank=max_rank,
            fixed_rank_grid=fixed_grid,
            max_features=min(scenario.max_features, 2_000),
            intersection_sample_count=min(scenario.intersection_sample_count, 32),
        )
    if horizon is not None:
        scenario = replace(scenario, T=horizon, t_b=min(scenario.t_b, max(1, horizon - 1)))
    return scenario


def default_fixed_rank(scenario: TextProductScenario, explicit_rank: int | None) -> int:
    if explicit_rank is not None:
        return min(max(1, explicit_rank), scenario.max_rank)
    target = min(scenario.latent_dim, scenario.max_rank)
    grid = sorted(rank for rank in scenario.fixed_rank_grid if rank <= scenario.max_rank)
    if not grid:
        return target
    for rank in grid:
        if rank >= target:
            return rank
    return grid[-1]


def method_filter_from_args(args: argparse.Namespace) -> set[str] | None:
    if args.method_set == "all":
        return None
    if args.method_set == "pilot":
        return set(PILOT_METHODS)
    raise ValueError(f"Unknown method set: {args.method_set}")


def run_experiments(args: argparse.Namespace) -> None:
    scenario_names = args.scenarios or (["mock_text_product_m4_d20"] if args.quick else ["text20news4_product_m20_d200"])
    p_values = args.p_values or ([0.4] if args.quick else [0.8, 0.6, 0.4, 0.3, 0.2])
    seed_count = args.seeds if args.seeds is not None else (1 if args.quick else 5)
    validation_count = args.validation_seeds if args.validation_seeds is not None else (1 if args.quick else 2)
    seeds = list(range(seed_count))
    validation_seeds = [10_000 + idx for idx in range(validation_count)]

    all_rows: list[dict[str, float | int | str]] = []
    all_summary: list[dict[str, float | int | str]] = []
    all_calibration: list[dict[str, float | int | str]] = []
    all_rank_rows: list[dict[str, float | int | str]] = []
    all_selections: list[dict[str, float | int | str]] = []
    method_filter = method_filter_from_args(args)

    for name in scenario_names:
        scenario = materialize_scenario(name, quick=args.quick, horizon=args.horizon)
        for p in p_values:
            if args.skip_calibration:
                print(f"Using default pilot selections for {scenario.name} at p={p:g}...")
                default_rank = default_fixed_rank(scenario, args.default_rank)
                tofu_full_rank = default_rank
                tofu_first_rank = default_rank
                pslb_rank = default_rank
                adaptive_threshold = args.default_adaptive_threshold
                adaptive_pslb_threshold = args.default_adaptive_pslb_threshold
                oful_lambda_reg = args.default_oful_lambda
                oful_beta_scale = args.default_oful_beta_scale
            else:
                print(f"Calibrating {scenario.name} at p={p:g}...")
                (
                    tofu_full_rank,
                    tofu_first_rank,
                    pslb_rank,
                    adaptive_threshold,
                    adaptive_pslb_threshold,
                    (oful_lambda_reg, oful_beta_scale),
                    rank_rows,
                    calibration_rows,
                ) = select_calibration(
                    scenario,
                    p=p,
                    validation_seeds=validation_seeds,
                    allow_downloads=args.allow_downloads,
                    cache_dir=args.cache_dir,
                    force_rebuild=args.force_rebuild,
                    adaptive_threshold_grid=threshold_grid(args.quick),
                    lambda_grid=oful_lambda_grid(args.quick),
                    beta_scale_grid=oful_beta_scale_grid(args.quick),
                )
                all_rank_rows.extend(rank_rows)
                all_calibration.extend(calibration_rows)
            all_selections.append(
                {
                    "scenario": scenario.name,
                    "source": scenario.source,
                    "p": p,
                    "tofu_full_rank": tofu_full_rank,
                    "tofu_first_rank": tofu_first_rank,
                    "pslb_rank": pslb_rank,
                    "adaptive_threshold": adaptive_threshold,
                    "adaptive_pslb_threshold": adaptive_pslb_threshold,
                    "oful_lambda_reg": oful_lambda_reg,
                    "oful_beta_scale": oful_beta_scale,
                }
            )
            print(
                "  selected ranks: "
                f"tofu_full={tofu_full_rank}, tofu_first={tofu_first_rank}, "
                f"pslb={pslb_rank}, threshold_tofu={adaptive_threshold:g}, "
                f"threshold_pslb={adaptive_pslb_threshold:g}, "
                f"oful=(lambda={oful_lambda_reg:g}, beta={oful_beta_scale:g})"
            )

            for seed in seeds:
                full_data = load_full_dataset_for_run(
                    scenario,
                    seed=seed,
                    allow_downloads=args.allow_downloads,
                    cache_dir=args.cache_dir,
                    force_rebuild=args.force_rebuild,
                )
                data = mask_image_classification_dataset(full_data, p=p, seed=mask_seed(seed, p))
                effective = effective_scenario(
                    scenario,
                    d=data.d,
                    latent_dim=int(data.metadata.get("latent_dim", scenario.latent_dim)),
                )
                policies = reporting_policies(
                    effective,
                    p=p,
                    K=data.K,
                    seed=seed,
                    tofu_full_rank=tofu_full_rank,
                    tofu_first_rank=tofu_first_rank,
                    pslb_rank=pslb_rank,
                    adaptive_threshold=adaptive_threshold,
                    adaptive_pslb_threshold=adaptive_pslb_threshold,
                    oful_lambda_reg=oful_lambda_reg,
                    oful_beta_scale=oful_beta_scale,
                    include_references=not args.no_references,
                )
                if method_filter is not None:
                    policies = [policy_spec for policy_spec in policies if policy_spec[0] in method_filter]
                for method, rank_label, configured_rank, policy, rank_threshold_constant, lambda_reg, beta_scale in policies:
                    print(f"  running {scenario.name} p={p:g} seed={seed} method={method}")
                    all_rows.extend(
                        run_trajectory(
                            effective,
                            p=p,
                            seed=seed,
                            split="report",
                            method=method,
                            rank_label=rank_label,
                            configured_rank=configured_rank,
                            policy=policy,
                            data=data,
                            rank_threshold_constant=rank_threshold_constant,
                            lambda_reg=lambda_reg,
                            beta_scale=beta_scale,
                        )
                    )

    all_summary = summarize(all_rows)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / TRAJECTORIES_FILENAME, all_rows, TRAJECTORY_COLUMNS)
    write_csv(args.results_dir / SUMMARY_FILENAME, all_summary, SUMMARY_COLUMNS)
    write_csv(args.results_dir / CALIBRATION_FILENAME, all_calibration, CALIBRATION_COLUMNS)
    write_csv(args.results_dir / RANK_SELECTION_FILENAME, all_rank_rows, RANK_SELECTION_COLUMNS)
    write_csv(args.results_dir / SELECTIONS_FILENAME, all_selections, SELECTIONS_COLUMNS)
    write_table(args.results_dir / TABLE_FILENAME, all_summary)
    print(f"Wrote trajectories to {args.results_dir / TRAJECTORIES_FILENAME}")
    print(f"Wrote summary to {args.results_dir / SUMMARY_FILENAME}")
    print(f"Wrote table to {args.results_dir / TABLE_FILENAME}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test experiment.")
    parser.add_argument("--scenarios", type=parse_scenario_list, default=None)
    parser.add_argument("--p-values", type=parse_float_list, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--validation-seeds", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-references", action="store_true")
    parser.add_argument(
        "--method-set",
        choices=("all", "pilot"),
        default="all",
        help="Run all reporting methods or a lean pilot subset.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Use deterministic default selections instead of validation calibration.",
    )
    parser.add_argument(
        "--default-rank",
        type=int,
        default=None,
        help="Fixed rank to use with --skip-calibration; defaults to the scenario latent dimension.",
    )
    parser.add_argument(
        "--default-adaptive-threshold",
        type=float,
        default=0.05,
        help="Adaptive TOFU rank threshold used with --skip-calibration.",
    )
    parser.add_argument(
        "--default-adaptive-pslb-threshold",
        type=float,
        default=0.03,
        help="Adaptive masked PSLB rank threshold used with --skip-calibration.",
    )
    parser.add_argument(
        "--default-oful-lambda",
        type=float,
        default=0.1,
        help="Zero-imputed OFUL lambda used with --skip-calibration.",
    )
    parser.add_argument(
        "--default-oful-beta-scale",
        type=float,
        default=0.25,
        help="Zero-imputed OFUL beta scale used with --skip-calibration.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_experiments(args)
    except DatasetUnavailableError as exc:
        if args.quick:
            print(f"Dataset unavailable in quick mode: {exc}")
            raise
        raise


if __name__ == "__main__":
    main()
