"""Configuration for the TOFU-POV learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

EpsilonSchedule = Callable[[int], float]
CovarianceRadiusSchedule = Callable[[int, int], float]
BurnInPolicy = Literal["random", "zero_oful"]
WarmStartReplay = Literal["first_epoch", "every_epoch", "full_history_every_epoch"]
RankSelection = Literal["fixed", "threshold"]


@dataclass(frozen=True)
class TOFUPOVConfig:
    """Hyperparameters for epoch-wise TOFU-POV.

    The default subspace-error schedule follows the expression in the plan and
    requires `lambda_1`, `lambda_m`, and `M`. If those constants are unavailable
    for a real dataset, provide `epsilon_schedule(tau_e)` instead.

    With `rank_selection="threshold"`, `m` is treated as the default maximum
    rank and the active rank is re-estimated at each epoch boundary.
    """

    d: int
    m: int
    K: int
    p: float
    lambda_reg: float
    t_b: int
    T: int
    delta: float
    L: float
    S: float
    R: float
    lambda_1: float | None = None
    lambda_m: float | None = None
    M: float | None = None
    c_sub: float = 1.0
    c_b: float = 1.0
    impute_ridge: float = 0.0
    random_seed: int | None = None
    epsilon_schedule: EpsilonSchedule | None = None
    covariance_radius_schedule: CovarianceRadiusSchedule | None = None
    burnin_policy: BurnInPolicy = "random"
    warm_start_from_burnin: bool = False
    warm_start_replay: WarmStartReplay = "first_epoch"
    rank_selection: RankSelection = "fixed"
    min_rank: int = 1
    max_rank: int | None = None
    rank_threshold_constant: float = 4.0

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be positive.")
        if self.m <= 0:
            raise ValueError("m must be positive.")
        if self.m > self.d:
            raise ValueError("m must be no larger than d.")
        if self.K <= 0:
            raise ValueError("K must be positive.")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("p must be in (0, 1].")
        if self.lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive.")
        if self.t_b < 1:
            raise ValueError("t_b must be at least 1 so the first epoch has history.")
        if self.T < self.t_b + 1:
            raise ValueError("T must be at least t_b + 1.")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1).")
        if self.L < 0.0 or self.S < 0.0 or self.R < 0.0:
            raise ValueError("L, S, and R must be nonnegative.")
        if self.c_sub < 0.0 or self.c_b < 0.0:
            raise ValueError("c_sub and c_b must be nonnegative.")
        if self.impute_ridge < 0.0:
            raise ValueError("impute_ridge must be nonnegative.")
        if self.burnin_policy not in {"random", "zero_oful"}:
            raise ValueError("burnin_policy must be 'random' or 'zero_oful'.")
        if self.warm_start_replay not in {
            "first_epoch",
            "every_epoch",
            "full_history_every_epoch",
        }:
            raise ValueError(
                "warm_start_replay must be 'first_epoch', 'every_epoch', "
                "or 'full_history_every_epoch'."
            )
        if self.rank_selection not in {"fixed", "threshold"}:
            raise ValueError("rank_selection must be 'fixed' or 'threshold'.")
        if self.min_rank <= 0:
            raise ValueError("min_rank must be positive.")
        if self.max_rank is not None and self.max_rank <= 0:
            raise ValueError("max_rank must be positive when provided.")
        effective_max_rank = self.m if self.max_rank is None else self.max_rank
        if effective_max_rank > self.d:
            raise ValueError("max_rank must be no larger than d.")
        if self.min_rank > effective_max_rank:
            raise ValueError("min_rank must be no larger than max_rank.")
        if self.rank_threshold_constant <= 0.0:
            raise ValueError("rank_threshold_constant must be positive.")
        if (
            self.rank_selection == "threshold"
            and self.covariance_radius_schedule is None
            and (self.lambda_1 is None or self.M is None)
        ):
            raise ValueError(
                "Threshold rank selection requires lambda_1 and M, or "
                "covariance_radius_schedule(tau_e, n_history)."
            )

        has_constants = (
            self.lambda_1 is not None and self.lambda_m is not None and self.M is not None
        )
        if not has_constants and self.epsilon_schedule is None:
            raise ValueError(
                "Provide lambda_1, lambda_m, and M, or provide epsilon_schedule(tau_e)."
            )
        if self.lambda_1 is not None and self.lambda_1 <= 0.0:
            raise ValueError("lambda_1 must be positive when provided.")
        if self.lambda_m is not None and self.lambda_m <= 0.0:
            raise ValueError("lambda_m must be positive when provided.")
        if self.M is not None and self.M <= 0.0:
            raise ValueError("M must be positive when provided.")
