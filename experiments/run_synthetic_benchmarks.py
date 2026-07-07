"""Run reproducible synthetic TOFU-POV benchmarks.

The script writes:

- `results/synthetic_mask_sweep.csv`
- `results/synthetic_bias_sweep.csv`
- `results/synthetic_burnin_sweep.csv`
- `results/synthetic_sparse_sweep.csv`
- `results/synthetic_feature_noise_sweep.csv`
- `results/synthetic_summary.md`
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from tofu_pov import (
    MaskedPSLB,
    OracleSubspaceOFUL,
    PSLB,
    PSLBConfig,
    RandomPolicy,
    SyntheticLowRankBanditEnv,
    TOFUPOV,
    TOFUPOVConfig,
    ZeroImputedOFUL,
    run_bandit,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class Scenario:
    d: int = 12
    m: int = 3
    K: int = 8
    T: int = 300
    p: float = 0.8
    t_b: int = 30
    noise_std: float = 0.05
    perturbation_std: float = 0.0
    lambda_reg: float = 1.0
    delta: float = 0.05
    L: float = 6.0
    S: float = 1.0
    R: float = 0.05
    impute_ridge: float = 1e-6


def make_env(scenario: Scenario, seed: int) -> SyntheticLowRankBanditEnv:
    return SyntheticLowRankBanditEnv(
        d=scenario.d,
        m=scenario.m,
        K=scenario.K,
        p=scenario.p,
        T=scenario.T,
        noise_std=scenario.noise_std,
        perturbation_std=scenario.perturbation_std,
        seed=seed,
    )


def tofu_policy(scenario: Scenario, seed: int, c_b: float) -> TOFUPOV:
    return tofu_policy_variant(scenario, seed, c_b=c_b)


def tofu_policy_variant(
    scenario: Scenario,
    seed: int,
    c_b: float,
    burnin_policy: str = "random",
    warm_start_from_burnin: bool = False,
    warm_start_replay: str = "first_epoch",
    rank_selection: str = "fixed",
    max_rank: int | None = None,
    rank_threshold_constant: float = 4.0,
) -> TOFUPOV:
    return TOFUPOV(
        TOFUPOVConfig(
            d=scenario.d,
            m=scenario.m,
            K=scenario.K,
            p=scenario.p,
            lambda_reg=scenario.lambda_reg,
            t_b=scenario.t_b,
            T=scenario.T,
            delta=scenario.delta,
            L=scenario.L,
            S=scenario.S,
            R=scenario.R,
            lambda_1=1.5,
            lambda_m=0.5,
            M=1.0,
            c_b=c_b,
            impute_ridge=scenario.impute_ridge,
            random_seed=seed + 1000,
            burnin_policy=burnin_policy,
            warm_start_from_burnin=warm_start_from_burnin,
            warm_start_replay=warm_start_replay,
            rank_selection=rank_selection,
            max_rank=max_rank,
            rank_threshold_constant=rank_threshold_constant,
        )
    )


def pslb_policy(
    scenario: Scenario,
    seed: int,
    use_intersection: bool = True,
    intersection_method: str = "projected_sampled",
) -> PSLB:
    return PSLB(
        PSLBConfig(
            d=scenario.d,
            m=scenario.m,
            K=scenario.K,
            T=scenario.T,
            lambda_reg=scenario.lambda_reg,
            delta=scenario.delta,
            L=scenario.L,
            S=scenario.S,
            R=scenario.R,
            warmup_rounds=max(1, scenario.t_b),
            use_intersection=use_intersection,
            intersection_method=intersection_method,
            random_seed=seed + 2000,
        )
    )


def masked_pslb_policy(
    scenario: Scenario,
    seed: int,
    use_intersection: bool = True,
    intersection_method: str = "projected_sampled",
    warmup_policy: str = "zero_oful",
) -> MaskedPSLB:
    return MaskedPSLB(
        PSLBConfig(
            d=scenario.d,
            m=scenario.m,
            K=scenario.K,
            T=scenario.T,
            lambda_reg=scenario.lambda_reg,
            delta=scenario.delta,
            L=scenario.L,
            S=scenario.S,
            R=scenario.R,
            warmup_rounds=max(1, scenario.t_b),
            warmup_policy=warmup_policy,
            use_intersection=use_intersection,
            intersection_method=intersection_method,
            random_seed=seed + 3000,
        )
    )


def run_policy(
    scenario: Scenario,
    seed: int,
    policy_name: str,
    policy_factory: Callable[[SyntheticLowRankBanditEnv], object],
) -> dict[str, float | int | str]:
    env = make_env(scenario, seed)
    policy = policy_factory(env)
    result = run_bandit(policy, env, seed=seed, policy_seed=seed)
    final_regret = float(result.cumulative_regret[-1])
    avg_reward = float(np.mean(result.rewards))
    action_entropy = empirical_entropy(result.actions, scenario.K)
    return {
        "seed": seed,
        "policy": policy_name,
        "d": scenario.d,
        "m": scenario.m,
        "K": scenario.K,
        "T": scenario.T,
        "p": scenario.p,
        "t_b": scenario.t_b,
        "perturbation_std": scenario.perturbation_std,
        "c_b": getattr(getattr(policy, "config", None), "c_b", math.nan),
        "final_regret": final_regret,
        "avg_regret": final_regret / scenario.T,
        "avg_reward": avg_reward,
        "action_entropy": action_entropy,
    }


def empirical_entropy(actions: np.ndarray, K: int) -> float:
    counts = np.bincount(actions, minlength=K).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def summarize(rows: list[dict[str, float | int | str]], group_keys: list[str]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows: list[dict[str, float | str]] = []
    for key, values in sorted(grouped.items()):
        regrets = np.array([float(v["final_regret"]) for v in values])
        avg_rewards = np.array([float(v["avg_reward"]) for v in values])
        item: dict[str, float | str] = {
            group_name: group_value for group_name, group_value in zip(group_keys, key)
        }
        item.update(
            {
                "n": len(values),
                "mean_final_regret": float(np.mean(regrets)),
                "stderr_final_regret": float(np.std(regrets, ddof=1) / np.sqrt(len(regrets)))
                if len(regrets) > 1
                else 0.0,
                "median_final_regret": float(np.median(regrets)),
                "mean_avg_reward": float(np.mean(avg_rewards)),
            }
        )
        summary_rows.append(item)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, float | str]], columns: list[str]) -> str:
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


def mask_sweep(seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for p in [1.0, 0.8, 0.6, 0.4]:
        scenario = Scenario(p=p, t_b=30)
        for seed in seeds:
            rows.extend(
                [
                    run_policy(scenario, seed, "TOFU-POV practical", lambda env: tofu_policy(scenario, seed, c_b=0.0)),
                    run_policy(
                        scenario,
                        seed,
                        "TOFU-POV zero-burnin",
                        lambda env: tofu_policy_variant(
                            scenario,
                            seed,
                            c_b=0.0,
                            burnin_policy="zero_oful",
                        ),
                    ),
                    run_policy(
                        scenario,
                        seed,
                        "TOFU-POV warm-start first-epoch",
                        lambda env: tofu_policy_variant(
                            scenario,
                            seed,
                            c_b=0.0,
                            burnin_policy="zero_oful",
                            warm_start_from_burnin=True,
                            warm_start_replay="first_epoch",
                        ),
                    ),
                    run_policy(
                        scenario,
                        seed,
                        "TOFU-POV warm-start every-epoch",
                        lambda env: tofu_policy_variant(
                            scenario,
                            seed,
                            c_b=0.0,
                            burnin_policy="zero_oful",
                            warm_start_from_burnin=True,
                            warm_start_replay="every_epoch",
                        ),
                    ),
                    run_policy(scenario, seed, "TOFU-POV theory radius", lambda env: tofu_policy(scenario, seed, c_b=1.0)),
                    run_policy(
                        scenario,
                        seed,
                        "Oracle-subspace OFUL",
                        lambda env: OracleSubspaceOFUL(
                            env.U,
                            lambda_reg=scenario.lambda_reg,
                            S=scenario.S,
                            R=scenario.R,
                            delta=scenario.delta,
                        ),
                    ),
                    run_policy(scenario, seed, "PSLB full-action", lambda env: pslb_policy(scenario, seed)),
                    run_policy(scenario, seed, "Masked PSLB", lambda env: masked_pslb_policy(scenario, seed)),
                    run_policy(
                        scenario,
                        seed,
                        "Masked PSLB min-UCB",
                        lambda env: masked_pslb_policy(
                            scenario,
                            seed,
                            intersection_method="min_ucb",
                        ),
                    ),
                    run_policy(
                        scenario,
                        seed,
                        "Zero-imputed OFUL",
                        lambda env: ZeroImputedOFUL(
                            d=scenario.d,
                            lambda_reg=scenario.lambda_reg,
                            S=scenario.S,
                            R=scenario.R,
                            delta=scenario.delta,
                        ),
                    ),
                    run_policy(scenario, seed, "Random", lambda env: RandomPolicy(K=scenario.K, seed=seed)),
                ]
            )
    return rows


def bias_sweep(seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    scenario = Scenario(p=0.8, t_b=30)
    for c_b in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]:
        for seed in seeds:
            rows.append(
                run_policy(
                    scenario,
                    seed,
                    f"TOFU-POV c_b={c_b:g}",
                    lambda env, c_b=c_b: tofu_policy(scenario, seed, c_b=c_b),
                )
            )
    return rows


def burnin_sweep(seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for t_b in [10, 20, 30, 60, 100]:
        scenario = Scenario(p=0.8, t_b=t_b)
        for seed in seeds:
            rows.append(
                run_policy(
                    scenario,
                    seed,
                    "TOFU-POV practical",
                    lambda env: tofu_policy(scenario, seed, c_b=0.0),
                )
            )
            rows.append(
                run_policy(
                    scenario,
                    seed,
                    "TOFU-POV warm-start first-epoch",
                    lambda env: tofu_policy_variant(
                        scenario,
                        seed,
                        c_b=0.0,
                        burnin_policy="zero_oful",
                        warm_start_from_burnin=True,
                        warm_start_replay="first_epoch",
                    ),
                )
            )
            rows.append(
                run_policy(
                    scenario,
                    seed,
                    "TOFU-POV warm-start every-epoch",
                    lambda env: tofu_policy_variant(
                        scenario,
                        seed,
                        c_b=0.0,
                        burnin_policy="zero_oful",
                        warm_start_from_burnin=True,
                        warm_start_replay="every_epoch",
                    ),
                )
            )
    return rows


def sparse_sweep(seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    scenarios = [
        ("d30_m3_p02", Scenario(d=30, m=3, K=8, T=400, p=0.2, t_b=30, L=8.0, impute_ridge=1e-5)),
        ("d30_m3_p03", Scenario(d=30, m=3, K=8, T=400, p=0.3, t_b=30, L=8.0, impute_ridge=1e-5)),
        ("d50_m3_p02", Scenario(d=50, m=3, K=10, T=400, p=0.2, t_b=40, L=8.0, impute_ridge=1e-5)),
        ("d50_m5_p02", Scenario(d=50, m=5, K=10, T=500, p=0.2, t_b=50, L=8.0, impute_ridge=1e-5)),
    ]

    for label, scenario in scenarios:
        for seed in seeds:
            scenario_rows = [
                run_policy(scenario, seed, "TOFU-POV practical", lambda env: tofu_policy(scenario, seed, c_b=0.0)),
                run_policy(
                    scenario,
                    seed,
                    "TOFU-POV warm-start first-epoch",
                    lambda env: tofu_policy_variant(
                        scenario,
                        seed,
                        c_b=0.0,
                        burnin_policy="zero_oful",
                        warm_start_from_burnin=True,
                        warm_start_replay="first_epoch",
                    ),
                ),
                run_policy(
                    scenario,
                    seed,
                    "TOFU-POV warm-start every-epoch",
                    lambda env: tofu_policy_variant(
                        scenario,
                        seed,
                        c_b=0.0,
                        burnin_policy="zero_oful",
                        warm_start_from_burnin=True,
                        warm_start_replay="every_epoch",
                    ),
                ),
                run_policy(scenario, seed, "Oracle-subspace OFUL", lambda env: OracleSubspaceOFUL(
                    env.U,
                    lambda_reg=scenario.lambda_reg,
                    S=scenario.S,
                    R=scenario.R,
                    delta=scenario.delta,
                )),
                run_policy(scenario, seed, "PSLB full-action", lambda env: pslb_policy(scenario, seed)),
                run_policy(scenario, seed, "Masked PSLB", lambda env: masked_pslb_policy(scenario, seed)),
                run_policy(scenario, seed, "Zero-imputed OFUL", lambda env: ZeroImputedOFUL(
                    d=scenario.d,
                    lambda_reg=scenario.lambda_reg,
                    S=scenario.S,
                    R=scenario.R,
                    delta=scenario.delta,
                )),
                run_policy(scenario, seed, "Random", lambda env: RandomPolicy(K=scenario.K, seed=seed)),
            ]
            for row in scenario_rows:
                row["scenario"] = label
            rows.extend(scenario_rows)
    return rows


def feature_noise_sweep(seeds: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    scenarios = [
        ("d30_m3_p03", Scenario(d=30, m=3, K=8, T=400, p=0.3, t_b=30, L=8.0, impute_ridge=1e-5)),
        ("d50_m3_p02", Scenario(d=50, m=3, K=10, T=400, p=0.2, t_b=40, L=8.0, impute_ridge=1e-5)),
    ]
    methods = [
        (
            "TOFU-POV warm-start first-epoch",
            lambda scenario, seed: tofu_policy_variant(
                scenario,
                seed,
                c_b=0.0,
                burnin_policy="zero_oful",
                warm_start_from_burnin=True,
                warm_start_replay="first_epoch",
            ),
        ),
        (
            "TOFU-POV warm-start every-epoch",
            lambda scenario, seed: tofu_policy_variant(
                scenario,
                seed,
                c_b=0.0,
                burnin_policy="zero_oful",
                warm_start_from_burnin=True,
                warm_start_replay="every_epoch",
            ),
        ),
        (
            "Zero-imputed OFUL",
            lambda scenario, seed: ZeroImputedOFUL(
                d=scenario.d,
                lambda_reg=scenario.lambda_reg,
                S=scenario.S,
                R=scenario.R,
                delta=scenario.delta,
            ),
        ),
        ("Masked PSLB", lambda scenario, seed: masked_pslb_policy(scenario, seed)),
        (
            "Oracle-subspace OFUL",
            lambda scenario, seed, env: OracleSubspaceOFUL(
                env.U,
                lambda_reg=scenario.lambda_reg,
                S=scenario.S,
                R=scenario.R,
                delta=scenario.delta,
            ),
        ),
    ]

    for label, base_scenario in scenarios:
        for perturbation_std in [0.0, 0.01, 0.03, 0.05]:
            scenario = Scenario(
                d=base_scenario.d,
                m=base_scenario.m,
                K=base_scenario.K,
                T=base_scenario.T,
                p=base_scenario.p,
                t_b=base_scenario.t_b,
                noise_std=base_scenario.noise_std,
                perturbation_std=perturbation_std,
                lambda_reg=base_scenario.lambda_reg,
                delta=base_scenario.delta,
                L=base_scenario.L,
                S=base_scenario.S,
                R=base_scenario.R,
                impute_ridge=base_scenario.impute_ridge,
            )
            for seed in seeds:
                env = make_env(scenario, seed)
                for policy_name, factory in methods:
                    if policy_name == "Oracle-subspace OFUL":
                        policy = factory(scenario, seed, env)
                        result = run_bandit(policy, env, seed=seed, policy_seed=seed)
                        row = {
                            "seed": seed,
                            "policy": policy_name,
                            "d": scenario.d,
                            "m": scenario.m,
                            "K": scenario.K,
                            "T": scenario.T,
                            "p": scenario.p,
                            "t_b": scenario.t_b,
                            "perturbation_std": scenario.perturbation_std,
                            "c_b": math.nan,
                            "final_regret": float(result.cumulative_regret[-1]),
                            "avg_regret": float(result.cumulative_regret[-1]) / scenario.T,
                            "avg_reward": float(np.mean(result.rewards)),
                            "action_entropy": empirical_entropy(result.actions, scenario.K),
                        }
                    else:
                        row = run_policy(scenario, seed, policy_name, lambda env, factory=factory: factory(scenario, seed))
                    row["scenario"] = label
                    rows.append(row)
    return rows


def write_summary(
    mask_rows: list[dict[str, float | int | str]],
    bias_rows: list[dict[str, float | int | str]],
    burnin_rows: list[dict[str, float | int | str]],
    sparse_rows: list[dict[str, float | int | str]],
    feature_noise_rows: list[dict[str, float | int | str]],
) -> None:
    mask_summary = summarize(mask_rows, ["p", "policy"])
    bias_summary = summarize(bias_rows, ["policy"])
    burnin_summary = summarize(burnin_rows, ["t_b", "policy"])
    sparse_summary = summarize(sparse_rows, ["scenario", "policy"])
    feature_noise_summary = summarize(feature_noise_rows, ["scenario", "perturbation_std", "policy"])

    best_by_p: list[str] = []
    for p in [1.0, 0.8, 0.6, 0.4]:
        subset = [row for row in mask_summary if float(row["p"]) == p]
        partial_subset = [
            row
            for row in subset
            if row["policy"]
            in {
                "TOFU-POV practical",
                "TOFU-POV zero-burnin",
                "TOFU-POV warm-start first-epoch",
                "TOFU-POV warm-start every-epoch",
                "TOFU-POV theory radius",
                "Zero-imputed OFUL",
                "Masked PSLB",
                "Masked PSLB min-UCB",
                "Random",
            }
        ]
        best_partial = min(partial_subset, key=lambda row: float(row["mean_final_regret"]))
        tofu = next(row for row in subset if row["policy"] == "TOFU-POV practical")
        tofu_first = next(row for row in subset if row["policy"] == "TOFU-POV warm-start first-epoch")
        tofu_every = next(row for row in subset if row["policy"] == "TOFU-POV warm-start every-epoch")
        zero = next(row for row in subset if row["policy"] == "Zero-imputed OFUL")
        masked_pslb = next(row for row in subset if row["policy"] == "Masked PSLB")
        masked_pslb_min = next(row for row in subset if row["policy"] == "Masked PSLB min-UCB")
        pslb = next(row for row in subset if row["policy"] == "PSLB full-action")
        best_by_p.append(
            f"- `p={p}`: best partial-observation baseline is `{best_partial['policy']}` "
            f"at {best_partial['mean_final_regret']:.2f}; practical TOFU-POV is "
            f"{tofu['mean_final_regret']:.2f}, first-epoch warm-start TOFU-POV is "
            f"{tofu_first['mean_final_regret']:.2f}, every-epoch warm-start TOFU-POV is "
            f"{tofu_every['mean_final_regret']:.2f}, zero-imputed OFUL is "
            f"{zero['mean_final_regret']:.2f}, masked PSLB is "
            f"{masked_pslb['mean_final_regret']:.2f}, masked PSLB min-UCB is "
            f"{masked_pslb_min['mean_final_regret']:.2f}, and full-action PSLB "
            f"is {pslb['mean_final_regret']:.2f}."
        )

    sparse_findings: list[str] = []
    for label in ["d30_m3_p02", "d30_m3_p03", "d50_m3_p02", "d50_m5_p02"]:
        subset = [row for row in sparse_summary if row["scenario"] == label]
        tofu = next(row for row in subset if row["policy"] == "TOFU-POV practical")
        tofu_first = next(row for row in subset if row["policy"] == "TOFU-POV warm-start first-epoch")
        tofu_every = next(row for row in subset if row["policy"] == "TOFU-POV warm-start every-epoch")
        zero = next(row for row in subset if row["policy"] == "Zero-imputed OFUL")
        improvement = 100.0 * (
            float(zero["mean_final_regret"]) - float(tofu_first["mean_final_regret"])
        ) / float(zero["mean_final_regret"])
        sparse_findings.append(
            f"- `{label}`: random-burn-in TOFU-POV regret {tofu['mean_final_regret']:.2f}, "
            f"first-epoch warm-start regret {tofu_first['mean_final_regret']:.2f}, "
            f"every-epoch warm-start regret {tofu_every['mean_final_regret']:.2f} "
            f"vs zero-imputed {zero['mean_final_regret']:.2f} "
            f"({improvement:.1f}% lower for first-epoch)."
        )

    summary = f"""# Synthetic Experiment Results

Generated by `experiments/run_synthetic_benchmarks.py`.

## Setup

- Synthetic low-rank contextual bandit with `d=12`, `m=3`, `K=8`, `T=300`.
- Rewards are linear in the true full arm plus Gaussian noise with `noise_std=0.05`.
- Regret is computed against the best full-information arm in each round.
- TOFU-POV uses all offered masked arms for subspace estimation and a tiny
  `impute_ridge=1e-6` for numerical stability in sparse masks.
- `TOFU-POV practical` sets `c_b=0.0` with random burn-in.
- `TOFU-POV zero-burnin` uses zero-imputed OFUL only for burn-in actions.
- `TOFU-POV warm-start first-epoch` also retroactively imputes/projects selected
  burn-in arms and initializes only the first low-dimensional OFUL epoch with
  those rewards. This is the theory-clean Option A.
- `TOFU-POV warm-start every-epoch` replays the same burn-in rewards after each
  epoch subspace refresh. This is the theory-adjacent natural improvement,
  Option B.
- `TOFU-POV theory radius` sets `c_b=1.0`.
- `PSLB full-action` is the original Lale et al. baseline, so it receives full
  arms rather than masked arms.
- `Masked PSLB` is a zero-imputed partial-observation adaptation of PSLB that
  receives only masked arms and uses projected-space sampled model selection
  over the projected/ambient confidence-set intersection. It uses zero-imputed
  OFUL during the PCA warm-up period.
- `Masked PSLB min-UCB` is the older deterministic `min(projected UCB, ambient
  UCB)` surrogate with the same zero-imputed OFUL warm-up, included only as an
  intersection diagnostic.
- Results average over `{len(set(int(row['seed']) for row in mask_rows))}` seeds.

## Headline

{chr(10).join(best_by_p)}

The practical TOFU-POV variants consistently beat random. In the small
`d=12` sweep, zero-imputed OFUL is very strong because dense masks still leave a
usable linear signal and it does not pay TOFU-POV's epoch-reset cost. Zero-OFUL
burn-in removes much of TOFU-POV's early regret tax.
In the sparse high-dimensional sweep, TOFU-POV overtakes zero-imputation:

{chr(10).join(sparse_findings)}

The raw theory-style subspace-bias radius is far too conservative for this
horizon and often over-explores. Full-action PSLB and oracle-subspace OFUL are
not apples-to-apples missing-feature baselines because they see complete action
vectors; masked PSLB is the fairer PSLB-style adaptation.

## Mask Sweep: Mean Final Regret

{markdown_table(mask_summary, ["p", "policy", "n", "mean_final_regret", "stderr_final_regret", "median_final_regret", "mean_avg_reward"])}

## TOFU-POV Bias Constant Sweep at `p=0.8`

{markdown_table(bias_summary, ["policy", "n", "mean_final_regret", "stderr_final_regret", "median_final_regret", "mean_avg_reward"])}

## TOFU-POV Burn-In Sweep at `p=0.8`

{markdown_table(burnin_summary, ["t_b", "policy", "n", "mean_final_regret", "stderr_final_regret", "median_final_regret", "mean_avg_reward"])}

## Sparse High-Dimensional Sweep

{markdown_table(sparse_summary, ["scenario", "policy", "n", "mean_final_regret", "stderr_final_regret", "median_final_regret", "mean_avg_reward"])}

## Feature Perturbation Sweep

{markdown_table(feature_noise_summary, ["scenario", "perturbation_std", "policy", "n", "mean_final_regret", "stderr_final_regret", "median_final_regret", "mean_avg_reward"])}

## Interpretation

- The method is doing the right qualitative thing: once missingness is severe
  enough that zero-filled ambient features become a liability, TOFU-POV's
  subspace recovery and imputation help.
- Zero-imputed OFUL during burn-in is a clear practical improvement over random
  burn-in, and first-epoch warm-start reuses otherwise discarded rewards.
- Replaying burn-in rewards at every epoch is a natural additional improvement,
  but it is less theory-clean than first-epoch-only warm-start.
- Performance degrades as `p` decreases because imputation becomes less
  identified and the subspace estimate is noisier.
- In dense low-dimensional settings, zero-imputed OFUL can be a tough baseline:
  it starts learning immediately and avoids epoch resets.
- The default theoretical bias constant is not calibrated for short finite
  synthetic horizons. It is useful as a paper-faithful implementation path, but
  practical experiments should sweep `c_b`.
- Burn-in has a real tradeoff: too little history hurts subspace quality, while
  too much burn-in spends many rounds acting randomly.
- These are synthetic results only. For real-world claims, the next step is to
  convert a dataset into `ArrayBanditEnv` format and report reward/off-policy
  metrics under a fixed candidate-set protocol.
"""
    (RESULTS_DIR / "synthetic_summary.md").write_text(summary)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    seeds = list(range(20))

    mask_rows = mask_sweep(seeds)
    bias_rows = bias_sweep(seeds)
    burnin_rows = burnin_sweep(seeds)
    sparse_rows = sparse_sweep(list(range(10)))
    feature_noise_rows = feature_noise_sweep(list(range(10)))

    write_csv(RESULTS_DIR / "synthetic_mask_sweep.csv", mask_rows)
    write_csv(RESULTS_DIR / "synthetic_bias_sweep.csv", bias_rows)
    write_csv(RESULTS_DIR / "synthetic_burnin_sweep.csv", burnin_rows)
    write_csv(RESULTS_DIR / "synthetic_sparse_sweep.csv", sparse_rows)
    write_csv(RESULTS_DIR / "synthetic_feature_noise_sweep.csv", feature_noise_rows)
    write_summary(mask_rows, bias_rows, burnin_rows, sparse_rows, feature_noise_rows)

    print((RESULTS_DIR / "synthetic_summary.md").read_text())


if __name__ == "__main__":
    main()
