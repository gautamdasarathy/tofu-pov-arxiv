"""Run real-world-style classification-to-bandit experiments."""

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
    load_real_world_bandit_dataset,
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
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
]

SUMMARY_COLUMNS = [
    "dataset",
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
    "dataset",
    "p",
    "method_family",
    "rank",
    "validation_seed",
    "final_regret",
]

CALIBRATION_COLUMNS = [
    "dataset",
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
class RealWorldScenario:
    name: str
    d: int
    T: int
    t_b: int
    max_rank: int
    fixed_rank_grid: tuple[int, ...]
    base_feature_dim: int | None = None


SCENARIOS = {
    "toy_classification": RealWorldScenario(
        name="toy_classification",
        d=24,
        T=400,
        t_b=30,
        max_rank=8,
        fixed_rank_grid=(2, 4, 6, 8),
    ),
    "digits_sklearn": RealWorldScenario(
        name="digits_sklearn",
        d=40,
        T=1000,
        t_b=50,
        max_rank=12,
        fixed_rank_grid=(2, 4, 8, 12),
    ),
    "20newsgroups": RealWorldScenario(
        name="20newsgroups",
        d=100,
        T=2500,
        t_b=100,
        max_rank=20,
        fixed_rank_grid=(5, 10, 15, 20),
        base_feature_dim=40,
    ),
    "covertype": RealWorldScenario(
        name="covertype",
        d=80,
        T=3000,
        t_b=100,
        max_rank=20,
        fixed_rank_grid=(5, 10, 15, 20),
    ),
}

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


def covariance_radius_schedule(p: float, d: int, T: int) -> Callable[[int, int], float]:
    def schedule(tau: int, n_history: int) -> float:
        del tau
        log_term = math.log(2.0 * d * max(T, 2) / 0.05)
        return float(0.08 / p * math.sqrt(max(log_term, 0.0) / max(n_history, 1)))

    return schedule


def epsilon_schedule(tau: int) -> float:
    del tau
    return 0.0


def tofu_policy(
    scenario: RealWorldScenario,
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
            R=0.5,
            c_b=0.0,
            impute_ridge=1e-5,
            random_seed=seed + 1000,
            epsilon_schedule=epsilon_schedule,
            covariance_radius_schedule=covariance_radius_schedule(p, scenario.d, scenario.T),
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
    scenario: RealWorldScenario,
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
            R=0.5,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 2000,
            rank_selection="threshold" if adaptive else "fixed",
            min_rank=1,
            max_rank=rank if adaptive else None,
            rank_threshold_constant=rank_threshold_constant,
            covariance_radius_schedule=covariance_radius_schedule(p, scenario.d, scenario.T),
        )
    )


def full_info_pslb_policy(
    scenario: RealWorldScenario,
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
            R=0.5,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 3000,
            arm_source="full",
            rank_selection="fixed",
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
    dataset_name: str,
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
                "dataset": dataset_name,
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
                "rank_threshold_constant": rank_threshold_constant,
                "lambda_reg": lambda_reg,
                "beta_scale": beta_scale,
            }
        )
    return rows


def load_dataset_for_run(
    scenario: RealWorldScenario,
    *,
    p: float,
    seed: int,
    allow_downloads: bool,
):
    return load_real_world_bandit_dataset(
        scenario.name,
        p=p,
        seed=seed,
        T=scenario.T,
        d=scenario.d,
        base_feature_dim=scenario.base_feature_dim,
        allow_downloads=allow_downloads,
    )


def select_fixed_ranks(
    scenario: RealWorldScenario,
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
        data = load_dataset_for_run(
            scenario,
            p=p,
            seed=validation_seed,
            allow_downloads=allow_downloads,
        )
        for rank in scenario.fixed_rank_grid:
            rank = min(rank, scenario.d)
            tofu = tofu_policy(
                scenario,
                p=p,
                K=data.K,
                seed=validation_seed,
                rank=rank,
                adaptive=False,
            )
            tofu_rows = run_trajectory(
                scenario.name,
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
            rank_rows.append(
                {
                    "dataset": scenario.name,
                    "p": p,
                    "method_family": "TOFU fixed-rank",
                    "rank": rank,
                    "validation_seed": validation_seed,
                    "final_regret": tofu_final,
                }
            )
            calibration_rows.append(
                {
                    "dataset": scenario.name,
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
                K=data.K,
                seed=validation_seed,
                rank=rank,
                adaptive=False,
            )
            pslb_rows = run_trajectory(
                scenario.name,
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
                    "dataset": scenario.name,
                    "p": p,
                    "method_family": "Masked PSLB fixed-rank",
                    "rank": rank,
                    "validation_seed": validation_seed,
                    "final_regret": pslb_final,
                }
            )
            calibration_rows.append(
                {
                    "dataset": scenario.name,
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
                K=data.K,
                seed=validation_seed,
                rank=scenario.max_rank,
                adaptive=True,
                rank_threshold_constant=threshold,
            )
            adaptive_tofu_rows = run_trajectory(
                scenario.name,
                p,
                validation_seed,
                "validation",
                "TOFU adaptive-rank",
                "candidate",
                f"max={scenario.max_rank},threshold={threshold:g}",
                adaptive_tofu,
                data,
                rank_threshold_constant=threshold,
            )
            adaptive_tofu_final = float(adaptive_tofu_rows[-1]["cumulative_regret"])
            adaptive_results["TOFU adaptive-rank"][threshold].append(adaptive_tofu_final)
            calibration_rows.append(
                {
                    "dataset": scenario.name,
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
                K=data.K,
                seed=validation_seed,
                rank=scenario.max_rank,
                adaptive=True,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_rows = run_trajectory(
                scenario.name,
                p,
                validation_seed,
                "validation",
                "Masked PSLB adaptive-rank",
                "candidate",
                f"max={scenario.max_rank},threshold={threshold:g}",
                adaptive_pslb,
                data,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_final = float(adaptive_pslb_rows[-1]["cumulative_regret"])
            adaptive_results["Masked PSLB adaptive-rank"][threshold].append(adaptive_pslb_final)
            calibration_rows.append(
                {
                    "dataset": scenario.name,
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
                    R=0.5,
                    delta=0.05,
                    beta_scale=beta_scale,
                )
                oful_rows = run_trajectory(
                    scenario.name,
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
                        "dataset": scenario.name,
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
    scenario: RealWorldScenario,
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
                R=0.5,
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
            scenario.max_rank,
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
                        R=0.5,
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
    rewards_by_group: dict[tuple[str, float, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            float(row["p"]),
            int(row["seed"]),
            str(row["method"]),
            str(row["rank_label"]),
        )
        group_key = (
            str(row["dataset"]),
            float(row["p"]),
            str(row["method"]),
            str(row["rank_label"]),
            str(row["configured_rank"]),
        )
        rewards_by_group[group_key].append(float(row["reward"]))
        if key not in final_rows_by_run or int(row["t"]) > int(final_rows_by_run[key]["t"]):
            final_rows_by_run[key] = row

    grouped: dict[tuple[str, float, str, str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in final_rows_by_run.values():
        grouped[
            (
                str(row["dataset"]),
                float(row["p"]),
                str(row["method"]),
                str(row["rank_label"]),
                str(row["configured_rank"]),
            )
        ].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (dataset, p, method, rank_label, configured_rank), values in sorted(grouped.items()):
        regrets = np.array([float(row["cumulative_regret"]) for row in values])
        rewards = np.array(rewards_by_group[(dataset, p, method, rank_label, configured_rank)])
        rank_values = [
            int(row["active_rank"])
            for row in values
            if str(row["active_rank"]) != ""
        ]
        summary.append(
            {
                "dataset": dataset,
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
    header = "| Dataset | p | Method | Final regret mean +/- SE | Mean final rank |\n"
    divider = "|---|---:|---|---:|---:|\n"
    lines = [header, divider]
    for row in sorted(rows, key=lambda item: (str(item["dataset"]), float(item["p"]), str(item["method"]))):
        rank = row["mean_final_rank"] if row["mean_final_rank"] != "" else "-"
        lines.append(
            "| {dataset} | {p:g} | {method} | {mean:.3f} +/- {se:.3f} | {rank} |\n".format(
                dataset=row["dataset"],
                p=float(row["p"]),
                method=row["method"],
                mean=float(row["mean_final_regret"]),
                se=float(row["stderr_final_regret"]),
                rank=rank,
            )
        )
    path.write_text("".join(lines))


def materialize_scenario(name: str, *, quick: bool, horizon: int | None) -> RealWorldScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    scenario = SCENARIOS[name]
    if quick:
        T = horizon or min(80, scenario.T)
        t_b = min(max(8, T // 8), T - 1)
        max_rank = min(scenario.max_rank, 8, scenario.d)
        fixed_grid = tuple(rank for rank in scenario.fixed_rank_grid if rank <= max_rank)
        if not fixed_grid:
            fixed_grid = (max_rank,)
        return replace(scenario, T=T, t_b=t_b, max_rank=max_rank, fixed_rank_grid=fixed_grid)
    if horizon is not None:
        t_b = min(scenario.t_b, max(1, horizon - 1))
        return replace(scenario, T=horizon, t_b=t_b)
    return scenario


def parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test experiment.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset names to run.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, default=None, help="Number of reporting seeds.")
    parser.add_argument("--validation-seeds", type=int, default=None, help="Number of validation seeds.")
    parser.add_argument("--p-values", type=parse_float_list, default=None, help="Comma-separated p values.")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow sklearn fetchers to download data.")
    parser.add_argument("--no-references", action="store_true", help="Skip random/full-information appendix policies.")
    return parser.parse_args()


def default_datasets(quick: bool) -> list[str]:
    if quick:
        return ["digits_sklearn"]
    return ["digits_sklearn", "20newsgroups", "covertype"]


def run_dataset(
    scenario: RealWorldScenario,
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
            data = load_dataset_for_run(
                scenario,
                p=p,
                seed=seed,
                allow_downloads=allow_downloads,
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
                scenario,
                p=p,
                K=data.K,
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
                        scenario.name,
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
    dataset_names = args.datasets or default_datasets(args.quick)
    p_values = args.p_values or ([0.8, 0.4] if args.quick else [0.8, 0.6, 0.4, 0.3, 0.2])
    seed_count = args.seeds if args.seeds is not None else (2 if args.quick else 10)
    seeds = list(range(seed_count))
    validation_seed_count = (
        args.validation_seeds if args.validation_seeds is not None else (1 if args.quick else 3)
    )
    validation_seeds = [10_000 + idx for idx in range(validation_seed_count)]
    include_references = not args.no_references

    all_rows: list[dict[str, float | int | str]] = []
    all_rank_rows: list[dict[str, float | int | str]] = []
    all_calibration_rows: list[dict[str, float | int | str]] = []
    skipped: list[str] = []

    for dataset_name in dataset_names:
        try:
            scenario = materialize_scenario(dataset_name, quick=args.quick, horizon=args.horizon)
            rows, rank_rows, calibration_rows = run_dataset(
                scenario,
                p_values=p_values,
                seeds=seeds,
                validation_seeds=validation_seeds,
                allow_downloads=args.allow_downloads,
                include_references=include_references,
                quick=args.quick,
            )
        except DatasetUnavailableError as exc:
            if args.quick and args.datasets is None and dataset_name == "digits_sklearn":
                print(f"Skipping digits_sklearn ({exc}); falling back to toy_classification.")
                scenario = materialize_scenario(
                    "toy_classification",
                    quick=True,
                    horizon=args.horizon,
                )
                rows, rank_rows, calibration_rows = run_dataset(
                    scenario,
                    p_values=p_values,
                    seeds=seeds,
                    validation_seeds=validation_seeds,
                    allow_downloads=False,
                    include_references=include_references,
                    quick=args.quick,
                )
            else:
                skipped.append(f"{dataset_name}: {exc}")
                continue
        all_rows.extend(rows)
        all_rank_rows.extend(rank_rows)
        all_calibration_rows.extend(calibration_rows)

    if not all_rows:
        message = "No real-world experiments ran successfully."
        if skipped:
            message += " Skipped: " + "; ".join(skipped)
        raise RuntimeError(message)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(all_rows)
    write_csv(args.results_dir / "real_world_trajectories.csv", all_rows, TRAJECTORY_COLUMNS)
    write_csv(args.results_dir / "real_world_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(args.results_dir / "real_world_rank_selection.csv", all_rank_rows, RANK_SELECTION_COLUMNS)
    write_csv(args.results_dir / "real_world_calibration.csv", all_calibration_rows, CALIBRATION_COLUMNS)
    write_markdown_table(args.results_dir / "real_world_table.md", summary)

    print(f"Wrote {len(all_rows)} trajectory rows to {args.results_dir / 'real_world_trajectories.csv'}")
    print(f"Wrote summary to {args.results_dir / 'real_world_summary.csv'}")
    print(f"Wrote rank-selection diagnostics to {args.results_dir / 'real_world_rank_selection.csv'}")
    print(f"Wrote calibration diagnostics to {args.results_dir / 'real_world_calibration.csv'}")
    if skipped:
        print("Skipped datasets: " + "; ".join(skipped))


if __name__ == "__main__":
    main()
