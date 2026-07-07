"""Run MovieLens recommendation-bandit experiments."""

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
    load_movielens_bandit_dataset,
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
DATA_DIR = ROOT / "data"

TRAJECTORY_COLUMNS = [
    "scenario",
    "feature_mode",
    "p",
    "seed",
    "method",
    "rank_label",
    "configured_rank",
    "t",
    "instant_regret",
    "cumulative_regret",
    "reward",
    "optimal_reward",
    "active_rank",
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
]

SUMMARY_COLUMNS = [
    "scenario",
    "feature_mode",
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
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
]

RANK_SELECTION_COLUMNS = [
    "scenario",
    "feature_mode",
    "p",
    "method_family",
    "rank",
    "validation_seed",
    "final_regret",
]

CALIBRATION_COLUMNS = [
    "scenario",
    "feature_mode",
    "p",
    "method_family",
    "candidate_label",
    "rank",
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
    "validation_seed",
    "final_regret",
    "selected",
]


@dataclass(frozen=True)
class MovieLensScenario:
    name: str = "movielens100k_hybrid"
    feature_mode: str = "hybrid"
    slate_mode: str = "random"
    d: int = 80
    K: int = 8
    T: int = 1500
    t_b: int = 80
    max_rank: int = 12
    fixed_rank_grid: tuple[int, ...] = (4, 8, 12)
    mf_rank: int = 8
    R: float = 0.25
    c_sub: float = 0.08
    adaptive_c_sub_grid: tuple[float, ...] = (0.04, 0.08, 0.12, 0.2)


SCENARIOS = {
    "movielens100k_mf": MovieLensScenario(
        name="movielens100k_mf",
        feature_mode="mf",
        d=60,
        max_rank=10,
        fixed_rank_grid=(4, 8, 10),
        mf_rank=8,
    ),
    "movielens100k_mf_product": MovieLensScenario(
        name="movielens100k_mf_product",
        feature_mode="mf_product",
        slate_mode="contrastive",
        d=200,
        K=15,
        T=2000,
        t_b=120,
        max_rank=12,
        fixed_rank_grid=(4, 8, 12),
        mf_rank=8,
        c_sub=0.08,
        adaptive_c_sub_grid=(0.02, 0.04, 0.08, 0.12, 0.2),
    ),
    "movielens100k_side_info": MovieLensScenario(
        name="movielens100k_side_info",
        feature_mode="side_info",
        d=80,
        max_rank=12,
        fixed_rank_grid=(4, 8, 12),
        mf_rank=8,
    ),
    "movielens100k_hybrid": MovieLensScenario(),
    "synthetic_movielens_mf": MovieLensScenario(
        name="synthetic_movielens_mf",
        feature_mode="mf",
        d=40,
        K=5,
        T=200,
        t_b=25,
        max_rank=8,
        fixed_rank_grid=(2, 4, 8),
        mf_rank=4,
    ),
}

FAIR_METHODS = [
    "Adaptive TOFU",
    "TOFU fixed-rank best-val",
    "Zero-imputed OFUL",
    "Masked PSLB fixed-rank best-val",
    "Masked PSLB adaptive-rank",
]


def covariance_radius_schedule(p: float, d: int, T: int, c_sub: float) -> Callable[[int, int], float]:
    def schedule(tau: int, n_history: int) -> float:
        del tau
        log_term = math.log(2.0 * d * max(T, 2) / 0.05)
        return float(c_sub / p * math.sqrt(max(log_term, 0.0) / max(n_history, 1)))

    return schedule


def epsilon_schedule(tau: int) -> float:
    del tau
    return 0.0


def tofu_policy(
    scenario: MovieLensScenario,
    *,
    p: float,
    seed: int,
    rank: int,
    adaptive: bool,
    rank_threshold_constant: float = 1.0,
) -> TOFUPOV:
    return TOFUPOV(
        TOFUPOVConfig(
            d=scenario.d,
            m=rank,
            K=scenario.K,
            p=p,
            lambda_reg=1.0,
            t_b=scenario.t_b,
            T=scenario.T,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.R,
            c_b=0.0,
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
    scenario: MovieLensScenario,
    *,
    p: float,
    seed: int,
    rank: int,
    adaptive: bool,
    rank_threshold_constant: float = 1.0,
) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=rank,
            K=scenario.K,
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
            covariance_radius_schedule=covariance_radius_schedule(
                p,
                scenario.d,
                scenario.T,
                scenario.c_sub,
            ),
            intersection_sample_count=128,
            rank_threshold_constant=rank_threshold_constant,
        )
    )


def full_info_pslb_policy(
    scenario: MovieLensScenario,
    *,
    p: float,
    seed: int,
    rank: int,
) -> PSLB:
    return PSLB(
        PSLBConfig(
            d=scenario.d,
            m=rank,
            K=scenario.K,
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
            intersection_sample_count=128,
        )
    )


def policies(
    scenario: MovieLensScenario,
    *,
    p: float,
    seed: int,
    tofu_rank: int,
    pslb_rank: int,
    adaptive_threshold: float,
    adaptive_pslb_threshold: float,
    oful_lambda_reg: float,
    oful_beta_scale: float,
    include_references: bool,
) -> list[tuple[str, str, int | str, object, float | str, float | str, float | str]]:
    items: list[tuple[str, str, int | str, object, float | str, float | str, float | str]] = [
        (
            "Adaptive TOFU",
            "adaptive",
            f"max={scenario.max_rank},threshold={adaptive_threshold:g}",
            tofu_policy(
                scenario,
                p=p,
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
            tofu_policy(scenario, p=p, seed=seed, rank=tofu_rank, adaptive=False),
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
            masked_pslb_policy(scenario, p=p, seed=seed, rank=pslb_rank, adaptive=False),
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
        items.extend(
            [
                ("Random", "random", "", RandomPolicy(K=scenario.K, seed=seed + 4000), "", "", ""),
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
                    full_info_pslb_policy(scenario, p=p, seed=seed, rank=pslb_rank),
                    "",
                    "",
                    "",
                ),
            ]
        )
    return items


def active_rank(policy: object) -> int | str:
    if hasattr(policy, "state_dict"):
        value = policy.state_dict().get("active_m")
        if value is not None:
            return int(value)
    if isinstance(policy, (ZeroImputedOFUL, FullInformationOFUL)):
        return policy.d
    return ""


def load_dataset(
    scenario: MovieLensScenario,
    *,
    p: float,
    seed: int,
    data_dir: Path,
    allow_downloads: bool,
    synthetic: bool,
):
    return load_movielens_bandit_dataset(
        p=p,
        seed=seed,
        T=scenario.T,
        K=scenario.K,
        d=scenario.d,
        rank=scenario.mf_rank,
        feature_mode=scenario.feature_mode,
        slate_mode=scenario.slate_mode,
        data_dir=data_dir,
        allow_downloads=allow_downloads,
        synthetic=synthetic,
    )


def select_fixed_ranks(
    scenario: MovieLensScenario,
    *,
    p: float,
    validation_seeds: list[int],
    data_dir: Path,
    allow_downloads: bool,
    synthetic: bool,
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
        data = load_dataset(
            scenario,
            p=p,
            seed=validation_seed,
            data_dir=data_dir,
            allow_downloads=allow_downloads,
            synthetic=synthetic,
        )
        for rank in scenario.fixed_rank_grid:
            rank = min(rank, scenario.d)
            tofu = tofu_policy(scenario, p=p, seed=validation_seed, rank=rank, adaptive=False)
            tofu_rows = run_trajectory(
                scenario,
                p=p,
                seed=validation_seed,
                method="TOFU fixed-rank",
                rank_label="candidate",
                configured_rank=rank,
                policy=tofu,
                data=data,
            )
            tofu_final = float(tofu_rows[-1]["cumulative_regret"])
            family_results["TOFU fixed-rank"][rank].append(tofu_final)
            rank_rows.append(
                {
                    "scenario": scenario.name,
                    "feature_mode": scenario.feature_mode,
                    "p": p,
                    "method_family": "TOFU fixed-rank",
                    "rank": rank,
                    "validation_seed": validation_seed,
                    "final_regret": tofu_final,
                }
            )
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "feature_mode": scenario.feature_mode,
                    "p": p,
                    "method_family": "TOFU fixed-rank",
                    "candidate_label": candidate_label(rank=rank),
                    "rank": rank,
                    "rank_threshold_constant": "",
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": tofu_final,
                    "selected": 0,
                }
            )

            pslb = masked_pslb_policy(
                scenario,
                p=p,
                seed=validation_seed,
                rank=rank,
                adaptive=False,
            )
            pslb_rows = run_trajectory(
                scenario,
                p=p,
                seed=validation_seed,
                method="Masked PSLB fixed-rank",
                rank_label="candidate",
                configured_rank=rank,
                policy=pslb,
                data=data,
            )
            pslb_final = float(pslb_rows[-1]["cumulative_regret"])
            family_results["Masked PSLB fixed-rank"][rank].append(pslb_final)
            rank_rows.append(
                {
                    "scenario": scenario.name,
                    "feature_mode": scenario.feature_mode,
                    "p": p,
                    "method_family": "Masked PSLB fixed-rank",
                    "rank": rank,
                    "validation_seed": validation_seed,
                    "final_regret": pslb_final,
                }
            )
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "feature_mode": scenario.feature_mode,
                    "p": p,
                    "method_family": "Masked PSLB fixed-rank",
                    "candidate_label": candidate_label(rank=rank),
                    "rank": rank,
                    "rank_threshold_constant": "",
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": pslb_final,
                    "selected": 0,
                }
            )

        for threshold in adaptive_threshold_grid:
            adaptive_tofu = tofu_policy(
                scenario,
                p=p,
                seed=validation_seed,
                rank=scenario.max_rank,
                adaptive=True,
                rank_threshold_constant=threshold,
            )
            adaptive_tofu_rows = run_trajectory(
                scenario,
                p=p,
                seed=validation_seed,
                method="TOFU adaptive-rank",
                rank_label="candidate",
                configured_rank=f"max={scenario.max_rank},threshold={threshold:g}",
                policy=adaptive_tofu,
                data=data,
                rank_threshold_constant=threshold,
            )
            adaptive_tofu_final = float(adaptive_tofu_rows[-1]["cumulative_regret"])
            adaptive_results["TOFU adaptive-rank"][threshold].append(adaptive_tofu_final)
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "feature_mode": scenario.feature_mode,
                    "p": p,
                    "method_family": "TOFU adaptive-rank",
                    "candidate_label": candidate_label(rank_threshold_constant=threshold),
                    "rank": scenario.max_rank,
                    "rank_threshold_constant": threshold,
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": adaptive_tofu_final,
                    "selected": 0,
                }
            )

            adaptive_pslb = masked_pslb_policy(
                scenario,
                p=p,
                seed=validation_seed,
                rank=scenario.max_rank,
                adaptive=True,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_rows = run_trajectory(
                scenario,
                p=p,
                seed=validation_seed,
                method="Masked PSLB adaptive-rank",
                rank_label="candidate",
                configured_rank=f"max={scenario.max_rank},threshold={threshold:g}",
                policy=adaptive_pslb,
                data=data,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_final = float(adaptive_pslb_rows[-1]["cumulative_regret"])
            adaptive_results["Masked PSLB adaptive-rank"][threshold].append(adaptive_pslb_final)
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "feature_mode": scenario.feature_mode,
                    "p": p,
                    "method_family": "Masked PSLB adaptive-rank",
                    "candidate_label": candidate_label(rank_threshold_constant=threshold),
                    "rank": scenario.max_rank,
                    "rank_threshold_constant": threshold,
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": adaptive_pslb_final,
                    "selected": 0,
                }
            )

        for lambda_reg in lambda_grid:
            for beta_scale in beta_scale_grid:
                oful = ZeroImputedOFUL(
                    d=scenario.d,
                    lambda_reg=lambda_reg,
                    S=1.0,
                    R=scenario.R,
                    delta=0.05,
                    beta_scale=beta_scale,
                )
                oful_rows = run_trajectory(
                    scenario,
                    p=p,
                    seed=validation_seed,
                    method="Zero-imputed OFUL",
                    rank_label="candidate",
                    configured_rank="ambient",
                    policy=oful,
                    data=data,
                    lambda_reg=lambda_reg,
                    beta_scale=beta_scale,
                )
                oful_final = float(oful_rows[-1]["cumulative_regret"])
                oful_results[(lambda_reg, beta_scale)].append(oful_final)
                calibration_rows.append(
                    {
                        "scenario": scenario.name,
                        "feature_mode": scenario.feature_mode,
                        "p": p,
                        "method_family": "Zero-imputed OFUL",
                        "candidate_label": candidate_label(lambda_reg=lambda_reg, beta_scale=beta_scale),
                        "rank": scenario.d,
                        "rank_threshold_constant": "",
                        "lambda_reg": lambda_reg,
                        "beta_scale": beta_scale,
                        "validation_seed": validation_seed,
                        "final_regret": oful_final,
                        "selected": 0,
                    }
                )

    selected: dict[str, int] = {}
    for family, by_rank in family_results.items():
        selected[family] = select_fixed_rank(by_rank)
    selected_adaptive: dict[str, float] = {
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


def run_trajectory(
    scenario: MovieLensScenario,
    *,
    p: float,
    seed: int,
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
    for t in range(1, scenario.T + 1):
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
                "feature_mode": scenario.feature_mode,
                "p": p,
                "seed": seed,
                "method": method,
                "rank_label": rank_label,
                "configured_rank": configured_rank,
                "t": t,
                "instant_regret": instant,
                "cumulative_regret": cumulative,
                "reward": reward,
                "optimal_reward": optimal,
                "active_rank": active_rank(policy),
                "rank_threshold_constant": rank_threshold_constant,
                "lambda_reg": lambda_reg,
                "beta_scale": beta_scale,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    final_by_run: dict[tuple[str, float, int, str], dict[str, float | int | str]] = {}
    rewards_by_group: dict[tuple[str, str, float, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["scenario"]), float(row["p"]), int(row["seed"]), str(row["method"]))
        group_key = (
            str(row["scenario"]),
            str(row["feature_mode"]),
            float(row["p"]),
            str(row["method"]),
            str(row["rank_label"]),
            str(row["configured_rank"]),
        )
        rewards_by_group[group_key].append(float(row["reward"]))
        if key not in final_by_run or int(row["t"]) > int(final_by_run[key]["t"]):
            final_by_run[key] = row

    grouped: dict[tuple[str, str, float, str, str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in final_by_run.values():
        grouped[
            (
                str(row["scenario"]),
                str(row["feature_mode"]),
                float(row["p"]),
                str(row["method"]),
                str(row["rank_label"]),
                str(row["configured_rank"]),
            )
        ].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (scenario_name, feature_mode, p, method, rank_label, configured_rank), values in sorted(grouped.items()):
        regrets = np.array([float(row["cumulative_regret"]) for row in values])
        rewards = np.array(
            rewards_by_group[(scenario_name, feature_mode, p, method, rank_label, configured_rank)]
        )
        rank_values = [
            int(row["active_rank"])
            for row in values
            if str(row["active_rank"]) != ""
        ]
        summary.append(
            {
                "scenario": scenario_name,
                "feature_mode": feature_mode,
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


def materialize_scenario(name: str, *, quick: bool, horizon: int | None) -> MovieLensScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown MovieLens scenario: {name}")
    scenario = SCENARIOS[name]
    if quick:
        T = horizon or min(60, scenario.T)
        t_b = min(max(8, T // 8), T - 1)
        return replace(scenario, T=T, t_b=t_b)
    if horizon is not None:
        return replace(scenario, T=horizon, t_b=min(scenario.t_b, max(1, horizon - 1)))
    return scenario


def parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run network-free synthetic MovieLens smoke test.")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--validation-seeds", type=int, default=None)
    parser.add_argument("--p-values", type=parse_float_list, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic MovieLens-like ratings.")
    parser.add_argument("--no-references", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = args.scenarios or (
        ["synthetic_movielens_mf"] if args.quick else ["movielens100k_hybrid"]
    )
    p_values = args.p_values or ([0.8, 0.4] if args.quick else [0.8, 0.6, 0.4, 0.3, 0.2])
    seeds = list(range(args.seeds if args.seeds is not None else (2 if args.quick else 10)))
    validation_seed_count = (
        args.validation_seeds if args.validation_seeds is not None else (1 if args.quick else 3)
    )
    validation_seeds = [10_000 + idx for idx in range(validation_seed_count)]
    synthetic = args.synthetic or args.quick
    include_references = not args.no_references

    rows: list[dict[str, float | int | str]] = []
    rank_rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    skipped: list[str] = []
    for scenario_name in scenario_names:
        scenario = materialize_scenario(scenario_name, quick=args.quick, horizon=args.horizon)
        for p in p_values:
            try:
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
                    data_dir=args.data_dir,
                    allow_downloads=args.allow_downloads,
                    synthetic=synthetic,
                    adaptive_threshold_grid=threshold_grid(args.quick),
                    lambda_grid=oful_lambda_grid(args.quick),
                    beta_scale_grid=oful_beta_scale_grid(args.quick),
                )
            except DatasetUnavailableError as exc:
                skipped.append(f"{scenario.name}: {exc}")
                continue
            rank_rows.extend(validation_rows)
            calibration_rows.extend(selected_calibration_rows)
            oful_lambda_reg, oful_beta_scale = oful_params
            for seed in seeds:
                try:
                    data = load_dataset(
                        scenario,
                        p=p,
                        seed=seed,
                        data_dir=args.data_dir,
                        allow_downloads=args.allow_downloads,
                        synthetic=synthetic,
                    )
                except DatasetUnavailableError as exc:
                    skipped.append(f"{scenario.name}: {exc}")
                    continue
                for (
                    method,
                    rank_label,
                    configured_rank,
                    policy,
                    rank_threshold_constant,
                    lambda_reg,
                    beta_scale,
                ) in policies(
                    scenario,
                    p=p,
                    seed=seed,
                    tofu_rank=tofu_rank,
                    pslb_rank=pslb_rank,
                    adaptive_threshold=adaptive_threshold,
                    adaptive_pslb_threshold=adaptive_pslb_threshold,
                    oful_lambda_reg=oful_lambda_reg,
                    oful_beta_scale=oful_beta_scale,
                    include_references=include_references,
                ):
                    rows.extend(
                        run_trajectory(
                            scenario,
                            p=p,
                            seed=seed,
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

    if not rows:
        message = "No MovieLens experiments ran successfully."
        if skipped:
            message += " Skipped: " + "; ".join(sorted(set(skipped)))
        raise RuntimeError(message)
    summary = summarize(rows)
    write_csv(args.results_dir / "movielens_trajectories.csv", rows, TRAJECTORY_COLUMNS)
    write_csv(args.results_dir / "movielens_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(args.results_dir / "movielens_rank_selection.csv", rank_rows, RANK_SELECTION_COLUMNS)
    write_csv(args.results_dir / "movielens_calibration.csv", calibration_rows, CALIBRATION_COLUMNS)
    print(f"Wrote {len(rows)} trajectory rows to {args.results_dir / 'movielens_trajectories.csv'}")
    print(f"Wrote summary to {args.results_dir / 'movielens_summary.csv'}")
    print(f"Wrote rank-selection diagnostics to {args.results_dir / 'movielens_rank_selection.csv'}")
    print(f"Wrote calibration diagnostics to {args.results_dir / 'movielens_calibration.csv'}")
    if skipped:
        print("Skipped datasets: " + "; ".join(sorted(set(skipped))))


if __name__ == "__main__":
    main()
