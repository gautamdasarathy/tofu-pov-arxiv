"""Generate trajectory data for the paper's main synthetic plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tofu_pov import (
    MaskedPSLB,
    PSLBConfig,
    RandomPolicy,
    SyntheticLowRankBanditEnv,
    TOFUPOV,
    TOFUPOVConfig,
    ZeroImputedOFUL,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

TRAJECTORY_COLUMNS = [
    "scenario",
    "p",
    "seed",
    "method",
    "t",
    "instant_regret",
    "cumulative_regret",
    "reward",
    "active_rank",
]


@dataclass(frozen=True)
class MainPlotScenario:
    name: str = "main_d30_m3"
    d: int = 30
    true_m: int = 3
    max_rank: int = 8
    K: int = 8
    T: int = 400
    t_b: int = 30
    noise_std: float = 0.05
    L: float = 8.0
    c_sub: float = 0.22
    impute_ridge: float = 1e-5


def make_env(scenario: MainPlotScenario, p: float, seed: int) -> SyntheticLowRankBanditEnv:
    return SyntheticLowRankBanditEnv(
        d=scenario.d,
        m=scenario.true_m,
        K=scenario.K,
        p=p,
        T=scenario.T,
        noise_std=scenario.noise_std,
        seed=seed,
    )


def known_rank_tofu(scenario: MainPlotScenario, p: float, seed: int) -> TOFUPOV:
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
            L=scenario.L,
            S=1.0,
            R=0.05,
            lambda_1=1.5,
            lambda_m=0.5,
            M=1.0,
            c_b=0.0,
            impute_ridge=scenario.impute_ridge,
            random_seed=seed + 1000,
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
            warm_start_replay="first_epoch",
        )
    )


def adaptive_tofu(scenario: MainPlotScenario, p: float, seed: int) -> TOFUPOV:
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
            L=scenario.L,
            S=1.0,
            R=0.05,
            lambda_1=1.5,
            lambda_m=0.5,
            M=1.0,
            c_sub=scenario.c_sub,
            c_b=0.0,
            impute_ridge=scenario.impute_ridge,
            random_seed=seed + 2000,
            burnin_policy="zero_oful",
            warm_start_from_burnin=True,
            warm_start_replay="first_epoch",
            rank_selection="threshold",
            min_rank=1,
            max_rank=scenario.max_rank,
            rank_threshold_constant=1.0,
        )
    )


def masked_pslb_adaptive(scenario: MainPlotScenario, p: float, seed: int) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=scenario.max_rank,
            K=scenario.K,
            T=scenario.T,
            p=p,
            lambda_reg=1.0,
            delta=0.05,
            L=scenario.L,
            S=1.0,
            R=0.05,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 3000,
            rank_selection="threshold",
            min_rank=1,
            max_rank=scenario.max_rank,
            rank_threshold_constant=1.0,
            covariance_radius_schedule=lambda t, n_history: 0.06,
        )
    )


def masked_pslb_known_rank(scenario: MainPlotScenario, p: float, seed: int) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=scenario.true_m,
            K=scenario.K,
            T=scenario.T,
            p=p,
            lambda_reg=1.0,
            delta=0.05,
            L=scenario.L,
            S=1.0,
            R=0.05,
            warmup_rounds=scenario.t_b,
            warmup_policy="zero_oful",
            random_seed=seed + 3500,
            rank_selection="fixed",
        )
    )


def zero_imputed_oful(scenario: MainPlotScenario, seed: int) -> ZeroImputedOFUL:
    return ZeroImputedOFUL(
        d=scenario.d,
        lambda_reg=1.0,
        S=1.0,
        R=0.05,
        delta=0.05,
    )


def policies(
    scenario: MainPlotScenario,
    p: float,
    seed: int,
) -> list[tuple[str, object]]:
    return [
        ("Adaptive TOFU", adaptive_tofu(scenario, p, seed)),
        ("Known-rank TOFU", known_rank_tofu(scenario, p, seed)),
        ("Zero-imputed OFUL", zero_imputed_oful(scenario, seed)),
        ("Masked PSLB known-rank", masked_pslb_known_rank(scenario, p, seed)),
        ("Masked PSLB adaptive-rank", masked_pslb_adaptive(scenario, p, seed)),
        ("Random", RandomPolicy(K=scenario.K, seed=seed + 4000)),
    ]


def active_rank(policy: object) -> int | str:
    if hasattr(policy, "state_dict"):
        state = policy.state_dict()
        value = state.get("active_m")
        if value is not None:
            return int(value)
    if isinstance(policy, ZeroImputedOFUL):
        return policy.d
    return ""


def run_trajectory(
    scenario: MainPlotScenario,
    p: float,
    seed: int,
    method: str,
    policy: object,
) -> list[dict[str, float | int | str]]:
    env = make_env(scenario, p, seed)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    env.reset(seed)

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
                "p": p,
                "seed": seed,
                "method": method,
                "t": t,
                "instant_regret": instant,
                "cumulative_regret": cumulative,
                "reward": reward,
                "active_rank": active_rank(policy),
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    final_t = max(int(row["t"]) for row in rows)
    final_rows = [row for row in rows if int(row["t"]) == final_t]
    grouped: dict[tuple[str, float, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in final_rows:
        grouped[(str(row["scenario"]), float(row["p"]), str(row["method"]))].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (scenario_name, p, method), values in sorted(grouped.items()):
        regrets = np.array([float(row["cumulative_regret"]) for row in values])
        rank_values = [
            int(row["active_rank"])
            for row in values
            if str(row["active_rank"]) != ""
        ]
        summary.append(
            {
                "scenario": scenario_name,
                "p": p,
                "method": method,
                "n": len(values),
                "mean_final_regret": float(np.mean(regrets)),
                "stderr_final_regret": float(np.std(regrets, ddof=1) / np.sqrt(len(regrets)))
                if len(values) > 1
                else 0.0,
                "median_final_regret": float(np.median(regrets)),
                "mean_final_rank": float(np.mean(rank_values)) if rank_values else "",
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]], columns: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke-test experiment.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, default=None, help="Number of seeds for non-quick runs.")
    parser.add_argument("--horizon", type=int, default=None, help="Override non-quick horizon.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        scenario = MainPlotScenario(T=60, t_b=10)
    else:
        default = MainPlotScenario()
        scenario = MainPlotScenario(T=args.horizon or default.T)
    p_values = [0.8, 0.4, 0.2] if args.quick else [0.8, 0.6, 0.4, 0.3, 0.2]
    seeds = list(range(2 if args.quick else (args.seeds or 20)))

    rows: list[dict[str, float | int | str]] = []
    for p in p_values:
        for seed in seeds:
            for method, policy in policies(scenario, p, seed):
                rows.extend(run_trajectory(scenario, p, seed, method, policy))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "main_plot_trajectories.csv", rows, TRAJECTORY_COLUMNS)
    write_csv(args.results_dir / "main_plot_summary.csv", summarize(rows))

    print(f"Wrote {len(rows)} trajectory rows to {args.results_dir / 'main_plot_trajectories.csv'}")
    print(f"Wrote summary to {args.results_dir / 'main_plot_summary.csv'}")


if __name__ == "__main__":
    main()
