"""Experiment runners for online bandit policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.envs import BanditEnv


@dataclass
class ExperimentResult:
    """Outputs from a single online bandit run."""

    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]
    regrets: NDArray[np.float64]
    cumulative_regret: NDArray[np.float64]
    metadata: dict[str, Any]


def run_bandit(
    policy: Any,
    env: BanditEnv,
    T: int | None = None,
    seed: int | None = None,
    policy_seed: int | None = None,
    reset_policy: bool = True,
) -> ExperimentResult:
    """Run an online policy against an environment."""

    env.reset(seed)
    if reset_policy and hasattr(policy, "reset"):
        policy.reset(policy_seed)

    horizon = T if T is not None else getattr(env, "T")
    actions = np.empty(horizon, dtype=int)
    rewards = np.empty(horizon, dtype=float)
    regrets = np.empty(horizon, dtype=float)

    for t in range(horizon):
        masked_arms, masks, full_arms = env.get_round()
        action = int(policy.observe(masked_arms, masks, full_arms))
        optimal = float(env.optimal_reward(full_arms))
        selected = _selected_expected_reward(env, full_arms, action)
        reward = float(env.step(action))
        policy.update(reward)

        actions[t] = action
        rewards[t] = reward
        regrets[t] = optimal - selected

    return ExperimentResult(
        actions=actions,
        rewards=rewards,
        regrets=regrets,
        cumulative_regret=np.cumsum(regrets),
        metadata={
            "T": horizon,
            "seed": seed,
            "policy_seed": policy_seed,
            "policy": type(policy).__name__,
            "env": type(env).__name__,
        },
    )


def compare_policies(
    policy_factories: Mapping[str, Callable[[int], Any]],
    env_factory: Callable[[int], BanditEnv],
    seeds: Sequence[int],
    T: int | None = None,
) -> dict[str, list[ExperimentResult]]:
    """Run several policies over matched environment seeds."""

    results: dict[str, list[ExperimentResult]] = {name: [] for name in policy_factories}
    for seed in seeds:
        for name, make_policy in policy_factories.items():
            env = env_factory(seed)
            policy = make_policy(seed)
            results[name].append(run_bandit(policy, env, T=T, seed=seed, policy_seed=seed))
    return results


def _selected_expected_reward(env: BanditEnv, full_arms: NDArray[np.float64], action: int) -> float:
    reward_mean = getattr(env, "reward_mean", None)
    if callable(reward_mean):
        return float(reward_mean(full_arms, action))
    return float("nan")
