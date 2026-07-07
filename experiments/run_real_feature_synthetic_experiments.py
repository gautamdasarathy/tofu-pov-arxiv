"""Run low-rank synthetic-reward experiments using real covariates."""

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
    MaskedPSLB,
    OracleSubspaceOFUL,
    PSLBConfig,
    RandomPolicy,
    TOFUPOV,
    TOFUPOVConfig,
    ZeroImputedOFUL,
    load_real_feature_synthetic_dataset,
)
from tofu_pov.calibration import (
    candidate_label,
    mark_selected,
    oful_beta_scale_grid,
    oful_lambda_grid,
    select_oful,
    select_threshold,
    threshold_grid,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRAJECTORY_COLUMNS = [
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
    "rank_threshold_constant",
    "lambda_reg",
    "beta_scale",
]

SUMMARY_COLUMNS = [
    "scenario",
    "source_dataset",
    "p",
    "method",
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

CALIBRATION_COLUMNS = [
    "scenario",
    "source_dataset",
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
class RealFeatureSyntheticScenario:
    name: str = "digits_lr_d80_m3"
    source_dataset: str = "digits_sklearn"
    d: int = 80
    true_m: int = 3
    max_rank: int = 10
    K: int = 8
    T: int = 600
    t_b: int = 40
    noise_std: float = 0.05
    perturbation_std: float = 0.0
    base_feature_dim: int | None = None
    c_sub: float = 0.12
    impute_ridge: float = 1e-5


SCENARIOS = {
    "digits_lr_d80_m3": RealFeatureSyntheticScenario(),
    "digits_lr_d160_m3": RealFeatureSyntheticScenario(
        name="digits_lr_d160_m3",
        d=160,
        true_m=3,
        max_rank=10,
        T=600,
        t_b=40,
    ),
    "digits_lr_d160_m5": RealFeatureSyntheticScenario(
        name="digits_lr_d160_m5",
        d=160,
        true_m=5,
        max_rank=12,
        T=700,
        t_b=50,
    ),
    "toy_lr_d80_m3": RealFeatureSyntheticScenario(
        name="toy_lr_d80_m3",
        source_dataset="toy_classification",
        d=80,
        true_m=3,
        max_rank=10,
        T=300,
        t_b=30,
    ),
}

METHOD_ORDER = [
    "Adaptive TOFU",
    "Known-rank TOFU",
    "Zero-imputed OFUL",
    "Masked PSLB known-rank",
    "Masked PSLB adaptive-rank",
    "Random",
    "Oracle-subspace OFUL",
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


def known_rank_tofu(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
) -> TOFUPOV:
    return TOFUPOV(
        TOFUPOVConfig(
            d=scenario.d,
            m=scenario.true_m,
            K=scenario.K,
            p=p,
            lambda_reg=1.0,
            t_b=scenario.t_b,
            T=scenario.T,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.noise_std,
            c_b=0.0,
            impute_ridge=scenario.impute_ridge,
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
        )
    )


def adaptive_tofu(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
    rank_threshold_constant: float = 1.0,
) -> TOFUPOV:
    return TOFUPOV(
        TOFUPOVConfig(
            d=scenario.d,
            m=scenario.max_rank,
            K=scenario.K,
            p=p,
            lambda_reg=1.0,
            t_b=scenario.t_b,
            T=scenario.T,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.noise_std,
            c_b=0.0,
            impute_ridge=scenario.impute_ridge,
            random_seed=seed + 2000,
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
            rank_selection="threshold",
            min_rank=1,
            max_rank=scenario.max_rank,
            rank_threshold_constant=rank_threshold_constant,
        )
    )


def masked_pslb_known_rank(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=scenario.true_m,
            K=scenario.K,
            T=scenario.T,
            p=p,
            lambda_reg=1.0,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.noise_std,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 3000,
            rank_selection="fixed",
            intersection_sample_count=128,
        )
    )


def masked_pslb_adaptive(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
    rank_threshold_constant: float = 1.0,
) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=scenario.max_rank,
            K=scenario.K,
            T=scenario.T,
            p=p,
            lambda_reg=1.0,
            delta=0.05,
            L=1.0,
            S=1.0,
            R=scenario.noise_std,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 4000,
            rank_selection="threshold",
            min_rank=1,
            max_rank=scenario.max_rank,
            rank_threshold_constant=rank_threshold_constant,
            covariance_radius_schedule=covariance_radius_schedule(
                p,
                scenario.d,
                scenario.T,
                scenario.c_sub,
            ),
            intersection_sample_count=128,
        )
    )


def zero_imputed_oful(
    scenario: RealFeatureSyntheticScenario,
    *,
    lambda_reg: float = 1.0,
    beta_scale: float = 1.0,
) -> ZeroImputedOFUL:
    return ZeroImputedOFUL(
        d=scenario.d,
        lambda_reg=lambda_reg,
        S=1.0,
        R=scenario.noise_std,
        delta=0.05,
        beta_scale=beta_scale,
    )


def policies(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
    U: np.ndarray,
    include_references: bool,
    adaptive_threshold: float,
    adaptive_pslb_threshold: float,
    oful_lambda_reg: float,
    oful_beta_scale: float,
) -> list[tuple[str, object, float | str, float | str, float | str]]:
    items: list[tuple[str, object, float | str, float | str, float | str]] = [
        ("Adaptive TOFU", adaptive_tofu(scenario, p=p, seed=seed, rank_threshold_constant=adaptive_threshold), adaptive_threshold, "", ""),
        ("Known-rank TOFU", known_rank_tofu(scenario, p=p, seed=seed), "", "", ""),
        ("Zero-imputed OFUL", zero_imputed_oful(scenario, lambda_reg=oful_lambda_reg, beta_scale=oful_beta_scale), "", oful_lambda_reg, oful_beta_scale),
        ("Masked PSLB known-rank", masked_pslb_known_rank(scenario, p=p, seed=seed), "", "", ""),
        ("Masked PSLB adaptive-rank", masked_pslb_adaptive(scenario, p=p, seed=seed, rank_threshold_constant=adaptive_pslb_threshold), adaptive_pslb_threshold, "", ""),
        ("Random", RandomPolicy(K=scenario.K, seed=seed + 5000), "", "", ""),
    ]
    if include_references:
        items.append(
            (
                "Oracle-subspace OFUL",
                OracleSubspaceOFUL(
                    U=U,
                    lambda_reg=oful_lambda_reg,
                    S=1.0,
                    R=scenario.noise_std,
                    delta=0.05,
                    beta_scale=oful_beta_scale,
                ),
                "",
                oful_lambda_reg,
                oful_beta_scale,
            )
        )
    return items


def active_rank(policy: object) -> int | str:
    if hasattr(policy, "state_dict"):
        value = policy.state_dict().get("active_m")
        if value is not None:
            return int(value)
    if isinstance(policy, ZeroImputedOFUL):
        return policy.d
    if isinstance(policy, OracleSubspaceOFUL):
        return policy.U.shape[1]
    return ""


def load_dataset(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
    allow_downloads: bool,
):
    return load_real_feature_synthetic_dataset(
        scenario.source_dataset,
        p=p,
        seed=seed,
        T=scenario.T,
        K=scenario.K,
        d=scenario.d,
        m=scenario.true_m,
        perturbation_std=scenario.perturbation_std,
        allow_downloads=allow_downloads,
        base_feature_dim=scenario.base_feature_dim,
    )


def run_trajectory(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    seed: int,
    method: str,
    policy: object,
    data,
    rank_threshold_constant: float | str = "",
    lambda_reg: float | str = "",
    beta_scale: float | str = "",
) -> list[dict[str, float | int | str]]:
    env = data.as_env(reward_noise_std=scenario.noise_std, seed=seed)
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
                "source_dataset": scenario.source_dataset,
                "p": p,
                "seed": seed,
                "method": method,
                "t": t,
                "instant_regret": instant,
                "cumulative_regret": cumulative,
                "reward": reward,
                "active_rank": active_rank(policy),
                "rank_threshold_constant": rank_threshold_constant,
                "lambda_reg": lambda_reg,
                "beta_scale": beta_scale,
            }
        )
    return rows


def select_calibration(
    scenario: RealFeatureSyntheticScenario,
    *,
    p: float,
    validation_seeds: list[int],
    allow_downloads: bool,
    adaptive_threshold_grid: tuple[float, ...],
    lambda_grid: tuple[float, ...],
    beta_scale_grid: tuple[float, ...],
) -> tuple[float, float, tuple[float, float], list[dict[str, float | int | str]]]:
    adaptive_results: dict[str, dict[float, list[float]]] = {
        "TOFU adaptive-rank": defaultdict(list),
        "Masked PSLB adaptive-rank": defaultdict(list),
    }
    oful_results: dict[tuple[float, float], list[float]] = defaultdict(list)
    calibration_rows: list[dict[str, float | int | str]] = []

    for validation_seed in validation_seeds:
        data = load_dataset(scenario, p=p, seed=validation_seed, allow_downloads=allow_downloads)
        for threshold in adaptive_threshold_grid:
            adaptive_policy = adaptive_tofu(
                scenario,
                p=p,
                seed=validation_seed,
                rank_threshold_constant=threshold,
            )
            adaptive_rows = run_trajectory(
                scenario,
                p=p,
                seed=validation_seed,
                method="TOFU adaptive-rank",
                policy=adaptive_policy,
                data=data,
                rank_threshold_constant=threshold,
            )
            adaptive_final = float(adaptive_rows[-1]["cumulative_regret"])
            adaptive_results["TOFU adaptive-rank"][threshold].append(adaptive_final)
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "source_dataset": scenario.source_dataset,
                    "p": p,
                    "method_family": "TOFU adaptive-rank",
                    "candidate_label": candidate_label(rank_threshold_constant=threshold),
                    "rank": scenario.max_rank,
                    "rank_threshold_constant": threshold,
                    "lambda_reg": "",
                    "beta_scale": "",
                    "validation_seed": validation_seed,
                    "final_regret": adaptive_final,
                    "selected": 0,
                }
            )

            adaptive_pslb_policy = masked_pslb_adaptive(
                scenario,
                p=p,
                seed=validation_seed,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_rows = run_trajectory(
                scenario,
                p=p,
                seed=validation_seed,
                method="Masked PSLB adaptive-rank",
                policy=adaptive_pslb_policy,
                data=data,
                rank_threshold_constant=threshold,
            )
            adaptive_pslb_final = float(adaptive_pslb_rows[-1]["cumulative_regret"])
            adaptive_results["Masked PSLB adaptive-rank"][threshold].append(adaptive_pslb_final)
            calibration_rows.append(
                {
                    "scenario": scenario.name,
                    "source_dataset": scenario.source_dataset,
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
                oful = zero_imputed_oful(
                    scenario,
                    lambda_reg=lambda_reg,
                    beta_scale=beta_scale,
                )
                oful_rows = run_trajectory(
                    scenario,
                    p=p,
                    seed=validation_seed,
                    method="Zero-imputed OFUL",
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
                        "source_dataset": scenario.source_dataset,
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

    selected_adaptive = {
        family: select_threshold(results)
        for family, results in adaptive_results.items()
    }
    selected_oful = select_oful(oful_results)
    selected_by_family: dict[str, object] = {
        **selected_adaptive,
        "Zero-imputed OFUL": selected_oful,
    }
    return (
        selected_adaptive["TOFU adaptive-rank"],
        selected_adaptive["Masked PSLB adaptive-rank"],
        selected_oful,
        mark_selected(calibration_rows, selected_by_family),
    )


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    final_by_run: dict[tuple[str, float, int, str], dict[str, float | int | str]] = {}
    rewards_by_group: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["scenario"]), float(row["p"]), int(row["seed"]), str(row["method"]))
        group_key = (
            str(row["scenario"]),
            str(row["source_dataset"]),
            float(row["p"]),
            str(row["method"]),
        )
        rewards_by_group[group_key].append(float(row["reward"]))
        if key not in final_by_run or int(row["t"]) > int(final_by_run[key]["t"]):
            final_by_run[key] = row

    grouped: dict[tuple[str, str, float, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in final_by_run.values():
        grouped[
            (
                str(row["scenario"]),
                str(row["source_dataset"]),
                float(row["p"]),
                str(row["method"]),
            )
        ].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (scenario_name, source_dataset, p, method), values in sorted(grouped.items()):
        regrets = np.array([float(row["cumulative_regret"]) for row in values])
        rewards = np.array(rewards_by_group[(scenario_name, source_dataset, p, method)])
        rank_values = [
            int(row["active_rank"])
            for row in values
            if str(row["active_rank"]) != ""
        ]
        summary.append(
            {
                "scenario": scenario_name,
                "source_dataset": source_dataset,
                "p": p,
                "method": method,
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


def materialize_scenario(name: str, *, quick: bool, horizon: int | None) -> RealFeatureSyntheticScenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    scenario = SCENARIOS[name]
    if quick:
        T = horizon or min(80, scenario.T)
        t_b = min(max(8, T // 8), T - 1)
        return replace(scenario, T=T, t_b=t_b, d=min(scenario.d, 40), max_rank=min(scenario.max_rank, 8))
    if horizon is not None:
        return replace(scenario, T=horizon, t_b=min(scenario.t_b, max(1, horizon - 1)))
    return scenario


def parse_float_list(values: str) -> list[float]:
    return [float(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test experiment.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario names to run.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--validation-seeds", type=int, default=None)
    parser.add_argument("--p-values", type=parse_float_list, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--no-references", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_names = args.scenarios or (["toy_lr_d80_m3"] if args.quick else ["digits_lr_d80_m3"])
    p_values = args.p_values or ([0.8, 0.4] if args.quick else [0.8, 0.6, 0.4, 0.3, 0.2])
    seeds = list(range(args.seeds if args.seeds is not None else (2 if args.quick else 10)))
    validation_seed_count = (
        args.validation_seeds if args.validation_seeds is not None else (1 if args.quick else 3)
    )
    validation_seeds = [10_000 + idx for idx in range(validation_seed_count)]
    include_references = not args.no_references

    rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    skipped: list[str] = []
    for scenario_name in scenario_names:
        scenario = materialize_scenario(scenario_name, quick=args.quick, horizon=args.horizon)
        for p in p_values:
            try:
                (
                    adaptive_threshold,
                    adaptive_pslb_threshold,
                    oful_params,
                    selected_calibration_rows,
                ) = select_calibration(
                    scenario,
                    p=p,
                    validation_seeds=validation_seeds,
                    allow_downloads=args.allow_downloads,
                    adaptive_threshold_grid=threshold_grid(args.quick),
                    lambda_grid=oful_lambda_grid(args.quick),
                    beta_scale_grid=oful_beta_scale_grid(args.quick),
                )
            except DatasetUnavailableError as exc:
                skipped.append(f"{scenario.name}: {exc}")
                continue
            calibration_rows.extend(selected_calibration_rows)
            oful_lambda_reg, oful_beta_scale = oful_params
            for seed in seeds:
                try:
                    data = load_dataset(scenario, p=p, seed=seed, allow_downloads=args.allow_downloads)
                except DatasetUnavailableError as exc:
                    skipped.append(f"{scenario.name}: {exc}")
                    continue
                for method, policy, rank_threshold_constant, lambda_reg, beta_scale in policies(
                    scenario,
                    p=p,
                    seed=seed,
                    U=data.U,
                    include_references=include_references,
                    adaptive_threshold=adaptive_threshold,
                    adaptive_pslb_threshold=adaptive_pslb_threshold,
                    oful_lambda_reg=oful_lambda_reg,
                    oful_beta_scale=oful_beta_scale,
                ):
                    rows.extend(
                        run_trajectory(
                            scenario,
                            p=p,
                            seed=seed,
                            method=method,
                            policy=policy,
                            data=data,
                            rank_threshold_constant=rank_threshold_constant,
                            lambda_reg=lambda_reg,
                            beta_scale=beta_scale,
                        )
                    )

    if not rows:
        message = "No real-feature synthetic experiments ran successfully."
        if skipped:
            message += " Skipped: " + "; ".join(skipped)
        raise RuntimeError(message)

    summary = summarize(rows)
    write_csv(args.results_dir / "real_feature_synthetic_trajectories.csv", rows, TRAJECTORY_COLUMNS)
    write_csv(args.results_dir / "real_feature_synthetic_summary.csv", summary, SUMMARY_COLUMNS)
    write_csv(
        args.results_dir / "real_feature_synthetic_calibration.csv",
        calibration_rows,
        CALIBRATION_COLUMNS,
    )
    print(
        "Wrote "
        f"{len(rows)} trajectory rows to "
        f"{args.results_dir / 'real_feature_synthetic_trajectories.csv'}"
    )
    print(f"Wrote summary to {args.results_dir / 'real_feature_synthetic_summary.csv'}")
    print(
        "Wrote calibration diagnostics to "
        f"{args.results_dir / 'real_feature_synthetic_calibration.csv'}"
    )
    if skipped:
        print("Skipped datasets: " + "; ".join(sorted(set(skipped))))


if __name__ == "__main__":
    main()
