"""Epoch-wise TOFU-POV learner."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tofu_pov.config import TOFUPOVConfig
from tofu_pov.imputation import ImputationError, impute_arm, impute_arms
from tofu_pov.oful import OFULModel, oful_confidence_radius
from tofu_pov.subspace import (
    corrected_covariance,
    sorted_eigendecomposition,
    threshold_rank,
)


class TOFUPOV:
    """Epoch-wise TOFU-POV learner.

    Rounds are tracked internally with one-indexed algorithm time: `observe`
    prepares the decision for round `t = completed_updates + 1`, and `update`
    completes that round.
    """

    def __init__(self, config: TOFUPOVConfig):
        self.config = config
        self._base_seed = config.random_seed
        self.rng = np.random.default_rng(self._base_seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self._base_seed
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.masked_history: list[NDArray[np.float64]] = []
        self.mask_history: list[NDArray[np.bool_]] = []
        self.burnin_selected_arms: list[NDArray[np.float64]] = []
        self.burnin_selected_masks: list[NDArray[np.bool_]] = []
        self.burnin_rewards: list[float] = []
        self.selected_arms: list[NDArray[np.float64]] = []
        self.selected_masks: list[NDArray[np.bool_]] = []
        self.selected_rewards: list[float] = []
        self.burnin_oful = OFULModel(self.config.d, self.config.lambda_reg)
        self.U_hat: NDArray[np.float64] | None = None
        self.eigenvalues: NDArray[np.float64] | None = None
        self.all_eigenvalues: NDArray[np.float64] | None = None
        self.active_m: int | None = None
        self.rank_history: list[int] = []
        self.rank_times: list[int] = []
        self.rank_threshold_e: float | None = None
        self.covariance_radius_e: float | None = None
        self.oful: OFULModel | None = None
        self.epoch_index = -1
        self._warm_started_epoch_count = 0
        self._full_history_replayed_epoch_count = 0
        self.epoch_start: int | None = None
        self.epoch_starts: list[int] = []
        self.next_epoch_start = self.config.t_b + 1
        self.epsilon_e = 0.0
        self.b_e = 0.0
        self.last_scores: NDArray[np.float64] | None = None
        self.last_features: NDArray[np.float64] | None = None
        self._pending_round: int | None = None
        self._pending_action: int | None = None
        self._pending_feature: NDArray[np.float64] | None = None
        self._pending_masked_arm: NDArray[np.float64] | None = None
        self._pending_mask: NDArray[np.bool_] | None = None
        self._pending_learning_update = False
        self._pending_burnin_update = False

    def observe(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
        full_arms: NDArray[np.float64] | None = None,
    ) -> int:
        """Return the selected arm index for the next round."""

        del full_arms
        if self._pending_round is not None:
            raise RuntimeError("Call update(reward) before observing the next round.")

        X, M = self._validate_round(masked_arms, masks)
        round_t = self.t + 1
        self.new_epoch_if_needed(round_t)

        # Current offered arms are available for future subspace estimates, not
        # for the subspace frozen at the start of this round.
        self.masked_history.append(X.copy())
        self.mask_history.append(M.copy())

        if round_t <= self.config.t_b:
            if self.config.burnin_policy == "zero_oful":
                beta = oful_confidence_radius(
                    self.burnin_oful,
                    S=self.config.S,
                    R=self.config.R,
                    delta=self.config.delta,
                )
                action, scores = self.burnin_oful.select(X, beta)
                self.last_scores = scores
            else:
                action = int(self.rng.integers(self.config.K))
                self.last_scores = None
            self._set_pending(
                round_t,
                action,
                None,
                masked_arm=X[action].copy(),
                mask=M[action].copy(),
                learning_update=False,
                burnin_update=True,
            )
            self.last_features = None
            return action

        if self.U_hat is None or self.oful is None:
            raise RuntimeError("No frozen subspace is available after burn-in.")

        try:
            imputed, _ = impute_arms(
                X,
                M,
                self.U_hat,
                impute_ridge=self.config.impute_ridge,
            )
        except ImputationError as exc:
            raise ImputationError(f"Failed to impute arms at round {round_t}: {exc}") from exc

        features = imputed @ self.U_hat
        beta = self._beta(round_t)
        action, scores = self.oful.select(features, beta)
        self.last_scores = scores
        self.last_features = features
        self._set_pending(
            round_t,
            action,
            features[action].copy(),
            masked_arm=X[action].copy(),
            mask=M[action].copy(),
            learning_update=True,
            burnin_update=False,
        )
        return action

    def update(self, reward: float) -> None:
        """Update the within-epoch OFUL state with the selected reward."""

        if self._pending_round is None or self._pending_action is None:
            raise RuntimeError("Call observe(masked_arms, masks) before update(reward).")
        value = float(reward)
        if self._pending_learning_update:
            if self.oful is None or self._pending_feature is None:
                raise RuntimeError("Missing OFUL state for a learning update.")
            self.oful.update(self._pending_feature, value)
        if self._pending_masked_arm is None:
            raise RuntimeError("Missing selected arm.")
        if self._pending_mask is None:
            raise RuntimeError("Missing selected mask.")
        self.selected_arms.append(self._pending_masked_arm.copy())
        self.selected_masks.append(self._pending_mask.copy())
        self.selected_rewards.append(value)
        if self._pending_burnin_update:
            self.burnin_selected_arms.append(self._pending_masked_arm.copy())
            self.burnin_selected_masks.append(self._pending_mask.copy())
            self.burnin_rewards.append(value)
            if self.config.burnin_policy == "zero_oful":
                self.burnin_oful.update(self._pending_masked_arm, value)

        self.t = self._pending_round
        self._pending_round = None
        self._pending_action = None
        self._pending_feature = None
        self._pending_masked_arm = None
        self._pending_mask = None
        self._pending_learning_update = False
        self._pending_burnin_update = False

    def new_epoch_if_needed(self, t: int) -> None:
        """Freeze a new subspace and reset OFUL at doubling epoch starts."""

        if t <= self.config.t_b:
            return

        while t >= self.next_epoch_start:
            self._start_new_epoch(self.next_epoch_start)
            self.next_epoch_start *= 2

    def state_dict(self) -> dict[str, Any]:
        """Return a debugging snapshot without mutating learner state."""

        return {
            "t": self.t,
            "history_size": sum(batch.shape[0] for batch in self.masked_history),
            "burnin_policy": self.config.burnin_policy,
            "warm_start_from_burnin": self.config.warm_start_from_burnin,
            "warm_start_replay": self.config.warm_start_replay,
            "warm_started_epoch_count": self._warm_started_epoch_count,
            "full_history_replayed_epoch_count": self._full_history_replayed_epoch_count,
            "selected_updates": len(self.selected_rewards),
            "burnin_updates": len(self.burnin_rewards),
            "burnin_oful_updates": self.burnin_oful.n_updates,
            "epoch_index": self.epoch_index,
            "epoch_start": self.epoch_start,
            "epoch_starts": list(self.epoch_starts),
            "next_epoch_start": self.next_epoch_start,
            "epsilon_e": self.epsilon_e,
            "b_e": self.b_e,
            "rank_selection": self.config.rank_selection,
            "active_m": self.active_m,
            "rank_history": list(self.rank_history),
            "rank_times": list(self.rank_times),
            "rank_threshold_e": self.rank_threshold_e,
            "covariance_radius_e": self.covariance_radius_e,
            "U_hat": None if self.U_hat is None else self.U_hat.copy(),
            "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.copy(),
            "all_eigenvalues": (
                None if self.all_eigenvalues is None else self.all_eigenvalues.copy()
            ),
            "V": None if self.oful is None else self.oful.V.copy(),
            "y": None if self.oful is None else self.oful.y.copy(),
            "theta_hat": None if self.oful is None else self.oful.theta_hat.copy(),
            "last_scores": None if self.last_scores is None else self.last_scores.copy(),
        }

    def _validate_round(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        X = np.asarray(masked_arms, dtype=float)
        M = np.asarray(masks, dtype=bool)
        expected = (self.config.K, self.config.d)
        if X.shape != expected:
            raise ValueError(f"masked_arms must have shape {expected}.")
        if M.shape != expected:
            raise ValueError(f"masks must have shape {expected}.")
        return X, M

    def _set_pending(
        self,
        round_t: int,
        action: int,
        feature: NDArray[np.float64] | None,
        masked_arm: NDArray[np.float64] | None,
        mask: NDArray[np.bool_] | None,
        learning_update: bool,
        burnin_update: bool,
    ) -> None:
        self._pending_round = round_t
        self._pending_action = action
        self._pending_feature = feature
        self._pending_masked_arm = masked_arm
        self._pending_mask = mask
        self._pending_learning_update = learning_update
        self._pending_burnin_update = burnin_update

    def _start_new_epoch(self, tau_e: int) -> None:
        if not self.masked_history:
            raise RuntimeError("Cannot estimate a subspace before collecting any arms.")

        historical_arms = np.vstack(self.masked_history)
        n_history = historical_arms.shape[0]
        covariance = corrected_covariance(historical_arms, self.config.p)
        all_eigenvalues, all_eigenvectors = sorted_eigendecomposition(covariance)
        active_m = self._select_epoch_rank(all_eigenvalues, tau_e, n_history)

        self.active_m = active_m
        self.rank_history.append(active_m)
        self.rank_times.append(tau_e)
        self.all_eigenvalues = all_eigenvalues
        self.eigenvalues = all_eigenvalues[:active_m]
        self.U_hat = all_eigenvectors[:, :active_m]
        self.oful = OFULModel(active_m, self.config.lambda_reg)
        if self._should_warm_start_epoch():
            self._warm_start_epoch()
        self.epoch_index += 1
        self.epoch_start = tau_e
        self.epoch_starts.append(tau_e)
        self.epsilon_e = self._epsilon(tau_e)
        self.b_e = self.config.c_b * self.config.S * self.config.L * (
            2.0 + 2.0 / self.config.p
        ) * self.epsilon_e

    def _select_epoch_rank(
        self,
        eigenvalues: NDArray[np.float64],
        tau_e: int,
        n_history: int,
    ) -> int:
        max_rank = self.config.m if self.config.max_rank is None else self.config.max_rank
        if self.config.rank_selection == "fixed":
            self.covariance_radius_e = None
            self.rank_threshold_e = None
            return self.config.m

        radius = self._covariance_radius(tau_e, n_history)
        threshold = self.config.rank_threshold_constant * radius
        self.covariance_radius_e = radius
        self.rank_threshold_e = threshold
        return threshold_rank(
            eigenvalues,
            threshold=threshold,
            min_rank=self.config.min_rank,
            max_rank=max_rank,
        )

    def _covariance_radius(self, tau_e: int, n_history: int) -> float:
        if self.config.covariance_radius_schedule is not None:
            radius = float(self.config.covariance_radius_schedule(tau_e, n_history))
            if radius < 0.0:
                raise ValueError("covariance_radius_schedule must return nonnegative values.")
            return radius

        assert self.config.lambda_1 is not None
        assert self.config.M is not None
        delta_rep = self._delta_e()
        log_term = math.log(2.0 * self.config.d * self.config.T / delta_rep)
        return float(
            self.config.c_sub
            * self.config.lambda_1
            * self.config.M
            / self.config.p
            * math.sqrt(max(log_term, 0.0) / max(n_history, 1))
        )

    def _should_warm_start_epoch(self) -> bool:
        if not self.config.warm_start_from_burnin:
            return False
        if self.config.warm_start_replay == "full_history_every_epoch":
            return True
        if self.config.warm_start_replay == "every_epoch":
            return True
        return self._warm_started_epoch_count == 0

    def _warm_start_epoch(self) -> None:
        if self.config.warm_start_replay == "full_history_every_epoch":
            self._warm_start_epoch_from_selected_history()
        else:
            self._warm_start_epoch_from_burnin()

    def _warm_start_epoch_from_burnin(self) -> None:
        if self.oful is None or self.U_hat is None:
            raise RuntimeError("Warm start requires an active OFUL model and subspace.")
        for arm, mask, reward in zip(
            self.burnin_selected_arms,
            self.burnin_selected_masks,
            self.burnin_rewards,
        ):
            imputed_arm, _ = impute_arm(
                arm,
                mask,
                self.U_hat,
                impute_ridge=self.config.impute_ridge,
            )
            feature = imputed_arm @ self.U_hat
            self.oful.update(feature, reward)
        self._warm_started_epoch_count += 1

    def _warm_start_epoch_from_selected_history(self) -> None:
        if self.oful is None or self.U_hat is None:
            raise RuntimeError("Full-history replay requires an active OFUL model and subspace.")
        for arm, mask, reward in zip(
            self.selected_arms,
            self.selected_masks,
            self.selected_rewards,
        ):
            imputed_arm, _ = impute_arm(
                arm,
                mask,
                self.U_hat,
                impute_ridge=self.config.impute_ridge,
            )
            feature = imputed_arm @ self.U_hat
            self.oful.update(feature, reward)
        self._warm_started_epoch_count += 1
        self._full_history_replayed_epoch_count += 1

    def _delta_e(self) -> float:
        epochs = math.ceil(math.log2(self.config.T)) + 2
        return self.config.delta / (2.0 * epochs)

    def _epsilon(self, tau_e: int) -> float:
        if self.config.epsilon_schedule is not None:
            epsilon = float(self.config.epsilon_schedule(tau_e))
            if epsilon < 0.0:
                raise ValueError("epsilon_schedule must return nonnegative values.")
            return epsilon

        assert self.config.lambda_1 is not None
        assert self.config.lambda_m is not None
        assert self.config.M is not None
        delta_rep = self._delta_e()
        log_term = math.log(2.0 * self.config.d * self.config.T / delta_rep)
        scale = (
            self.config.c_sub
            * self.config.lambda_1
            * self.config.M
            / (self.config.lambda_m * self.config.p)
        )
        return float(
            scale
            * math.sqrt(self._active_dimension() / (tau_e * self.config.K) * max(log_term, 0.0))
        )

    def _active_dimension(self) -> int:
        return self.active_m if self.active_m is not None else self.config.m

    def _beta(self, t: int) -> float:
        if self.oful is None or self.epoch_start is None:
            raise RuntimeError("No active OFUL epoch.")
        return oful_confidence_radius(
            self.oful,
            S=self.config.S,
            R=self.config.R,
            delta=self._delta_e(),
            bias=self.b_e,
            bias_rounds=max(0, t - self.epoch_start),
        )
