"""Run focused rank-misspecification and adaptive-rank experiments.

This script intentionally excludes full-information references. Those are
covered by `run_synthetic_benchmarks.py`; here the question is how fixed-rank
partial-observation methods behave when the rank is under/over-specified, and
whether the threshold-rank rule helps.

Outputs:

- `results/rank_adaptation.csv`
- `results/rank_adaptation_summary.md`
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tofu_pov import (
    MaskedPSLB,
    PSLBConfig,
    SyntheticLowRankBanditEnv,
    TOFUPOV,
    TOFUPOVConfig,
    ZeroImputedOFUL,
    run_bandit,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class RankScenario:
    name: str
    d: int
    true_m: int
    max_rank: int
    K: int
    T: int
    p: float
    t_b: int
    L: float
    tofu_radius: float
    masked_pslb_radius: float
    noise_std: float = 0.05
    impute_ridge: float = 1e-5


def make_env(scenario: RankScenario, seed: int) -> SyntheticLowRankBanditEnv:
    return SyntheticLowRankBanditEnv(
        d=scenario.d,
        m=scenario.true_m,
        K=scenario.K,
        p=scenario.p,
        T=scenario.T,
        noise_std=scenario.noise_std,
        seed=seed,
    )


def tofu_config(
    scenario: RankScenario,
    seed: int,
    rank: int,
    adaptive: bool,
) -> TOFUPOVConfig:
    return TOFUPOVConfig(
        d=scenario.d,
        m=rank,
        K=scenario.K,
        p=scenario.p,
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
        rank_selection="threshold" if adaptive else "fixed",
        min_rank=1,
        max_rank=scenario.max_rank if adaptive else None,
        rank_threshold_constant=1.0,
        covariance_radius_schedule=(
            (lambda tau_e, n_history, radius=scenario.tofu_radius: radius)
            if adaptive
            else None
        ),
    )


def masked_pslb_config(
    scenario: RankScenario,
    seed: int,
    rank: int,
    adaptive: bool,
) -> PSLBConfig:
    return PSLBConfig(
        d=scenario.d,
        m=rank,
        K=scenario.K,
        T=scenario.T,
        p=scenario.p,
        lambda_reg=1.0,
        delta=0.05,
        L=scenario.L,
        S=1.0,
        R=0.05,
        warmup_rounds=scenario.t_b,
        warmup_policy="zero_oful",
        random_seed=seed + 2000,
        rank_selection="threshold" if adaptive else "fixed",
        min_rank=1,
        max_rank=scenario.max_rank if adaptive else None,
        rank_threshold_constant=1.0,
        covariance_radius_schedule=(
            (lambda t, n_history, radius=scenario.masked_pslb_radius: radius)
            if adaptive
            else None
        ),
    )


def rank_grid(scenario: RankScenario) -> list[tuple[str, int]]:
    under = max(1, scenario.true_m - 1)
    over = min(scenario.max_rank, 2 * scenario.true_m)
    return [
        ("under", under),
        ("true", scenario.true_m),
        ("over", over),
    ]


def run_policy(
    scenario: RankScenario,
    seed: int,
    method: str,
    rank_label: str,
    configured_rank: int,
    adaptive: bool,
    factory: Callable[[], object],
) -> dict[str, float | int | str]:
    env = make_env(scenario, seed)
    policy = factory()
    result = run_bandit(policy, env, seed=seed, policy_seed=seed)
    state = policy.state_dict() if hasattr(policy, "state_dict") else {}
    rank_history = state.get("rank_history", [])
    rank_times = state.get("rank_times", [])
    final_regret = float(result.cumulative_regret[-1])
    final_rank = int(rank_history[-1]) if rank_history else configured_rank
    correct_rank_rate = (
        float(np.mean(np.asarray(rank_history, dtype=int) == scenario.true_m))
        if rank_history
        else float(configured_rank == scenario.true_m)
    )
    return {
        "scenario": scenario.name,
        "seed": seed,
        "method": method,
        "rank_label": rank_label,
        "configured_rank": configured_rank,
        "adaptive": int(adaptive),
        "true_m": scenario.true_m,
        "max_rank": scenario.max_rank,
        "p": scenario.p,
        "d": scenario.d,
        "K": scenario.K,
        "T": scenario.T,
        "t_b": scenario.t_b,
        "final_regret": final_regret,
        "avg_regret": final_regret / scenario.T,
        "final_rank": final_rank,
        "correct_rank_rate": correct_rank_rate,
        "rank_history": " ".join(str(int(rank)) for rank in rank_history),
        "rank_times": " ".join(str(int(time)) for time in rank_times),
    }


def run_scenario(scenario: RankScenario, seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        rows.append(
            run_policy(
                scenario,
                seed,
                "Zero-imputed OFUL",
                "ambient",
                scenario.d,
                False,
                lambda scenario=scenario: ZeroImputedOFUL(
                    d=scenario.d,
                    lambda_reg=1.0,
                    S=1.0,
                    R=0.05,
                    delta=0.05,
                ),
            )
        )

        for rank_label, rank in rank_grid(scenario):
            rows.append(
                run_policy(
                    scenario,
                    seed,
                    "TOFU fixed-rank",
                    rank_label,
                    rank,
                    False,
                    lambda scenario=scenario, seed=seed, rank=rank: TOFUPOV(
                        tofu_config(scenario, seed, rank=rank, adaptive=False)
                    ),
                )
            )
            rows.append(
                run_policy(
                    scenario,
                    seed,
                    "Masked PSLB fixed-rank",
                    rank_label,
                    rank,
                    False,
                    lambda scenario=scenario, seed=seed, rank=rank: MaskedPSLB(
                        masked_pslb_config(scenario, seed, rank=rank, adaptive=False)
                    ),
                )
            )

        rows.append(
            run_policy(
                scenario,
                seed,
                "TOFU adaptive-rank",
                "adaptive",
                scenario.max_rank,
                True,
                lambda scenario=scenario, seed=seed: TOFUPOV(
                    tofu_config(scenario, seed, rank=scenario.max_rank, adaptive=True)
                ),
            )
        )
        rows.append(
            run_policy(
                scenario,
                seed,
                "Masked PSLB adaptive-rank",
                "adaptive",
                scenario.max_rank,
                True,
                lambda scenario=scenario, seed=seed: MaskedPSLB(
                    masked_pslb_config(scenario, seed, rank=scenario.max_rank, adaptive=True)
                ),
            )
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scenario"]), str(row["method"]), str(row["rank_label"]))].append(row)

    summary: list[dict[str, float | int | str]] = []
    for (scenario, method, rank_label), values in sorted(grouped.items()):
        regrets = np.array([float(row["final_regret"]) for row in values])
        final_ranks = np.array([int(row["final_rank"]) for row in values])
        correct_rates = np.array([float(row["correct_rank_rate"]) for row in values])
        first = values[0]
        summary.append(
            {
                "scenario": scenario,
                "method": method,
                "rank_label": rank_label,
                "configured_rank": first["configured_rank"],
                "true_m": first["true_m"],
                "p": first["p"],
                "n": len(values),
                "mean_final_regret": float(np.mean(regrets)),
                "stderr_final_regret": float(np.std(regrets, ddof=1) / np.sqrt(len(regrets)))
                if len(values) > 1
                else 0.0,
                "median_final_regret": float(np.median(regrets)),
                "mean_final_rank": float(np.mean(final_ranks)),
                "mean_correct_rank_rate": float(np.mean(correct_rates)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, float | int | str]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(summary_rows: list[dict[str, float | int | str]]) -> None:
    columns = [
        "scenario",
        "method",
        "rank_label",
        "configured_rank",
        "true_m",
        "n",
        "mean_final_regret",
        "stderr_final_regret",
        "mean_final_rank",
        "mean_correct_rank_rate",
    ]
    headline_lines = []
    for scenario in sorted({str(row["scenario"]) for row in summary_rows}):
        subset = [row for row in summary_rows if row["scenario"] == scenario]
        best = min(subset, key=lambda row: float(row["mean_final_regret"]))
        tofu_adaptive = next(row for row in subset if row["method"] == "TOFU adaptive-rank")
        pslb_adaptive = next(row for row in subset if row["method"] == "Masked PSLB adaptive-rank")
        headline_lines.append(
            f"- `{scenario}`: best method is `{best['method']} ({best['rank_label']})` "
            f"with regret {best['mean_final_regret']:.2f}. Adaptive TOFU regret is "
            f"{tofu_adaptive['mean_final_regret']:.2f} with mean final rank "
            f"{tofu_adaptive['mean_final_rank']:.2f}; adaptive masked PSLB regret is "
            f"{pslb_adaptive['mean_final_regret']:.2f} with mean final rank "
            f"{pslb_adaptive['mean_final_rank']:.2f}."
        )

    text = f"""# Rank Adaptation Synthetic Results

Generated by `experiments/run_rank_adaptation_experiments.py`.

## Setup

- Only partial-observation methods are included here.
- Fixed-rank TOFU and masked PSLB are evaluated with under-specified, true, and
  over-specified ranks.
- Adaptive-rank TOFU and adaptive masked PSLB use the same threshold-style rank
  rule, but each applies it to its own covariance estimate: TOFU uses corrected
  masked covariance; masked PSLB uses zero-imputed masked-arm covariance.
- TOFU variants use zero-imputed OFUL burn-in plus first-epoch warm start,
  matching the theory-clean warm-start version.

## Headline

{chr(10).join(headline_lines)}

## Summary Table

{markdown_table(summary_rows, columns)}

## Reading The Table

- `mean_final_rank` is the final selected rank averaged over seeds. For fixed
  methods, this equals the configured rank.
- `mean_correct_rank_rate` is the fraction of rank selections equal to the true
  rank, averaged over seeds. For fixed methods, it is either `0` or `1`.
- Full-information references are intentionally omitted; see
  `results/synthetic_summary.md` for those.
"""
    (RESULTS_DIR / "rank_adaptation_summary.md").write_text(text)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    seeds = list(range(10))
    scenarios = [
        RankScenario(
            name="dense_d12_m3_p08",
            d=12,
            true_m=3,
            max_rank=6,
            K=8,
            T=300,
            p=0.8,
            t_b=30,
            L=6.0,
            tofu_radius=0.08,
            masked_pslb_radius=0.08,
        ),
        RankScenario(
            name="sparse_d30_m3_p03",
            d=30,
            true_m=3,
            max_rank=8,
            K=8,
            T=350,
            p=0.3,
            t_b=30,
            L=8.0,
            tofu_radius=0.12,
            masked_pslb_radius=0.06,
        ),
        RankScenario(
            name="sparse_d50_m5_p02",
            d=50,
            true_m=5,
            max_rank=10,
            K=10,
            T=400,
            p=0.2,
            t_b=50,
            L=8.0,
            tofu_radius=0.16,
            masked_pslb_radius=0.05,
        ),
    ]

    rows: list[dict[str, float | int | str]] = []
    for scenario in scenarios:
        rows.extend(run_scenario(scenario, seeds))

    summary_rows = summarize(rows)
    write_csv(RESULTS_DIR / "rank_adaptation.csv", rows)
    write_csv(RESULTS_DIR / "rank_adaptation_summary.csv", summary_rows)
    write_summary(summary_rows)
    print((RESULTS_DIR / "rank_adaptation_summary.md").read_text())


if __name__ == "__main__":
    main()
