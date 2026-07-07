"""Baseline policies for experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from tofu_pov.oful import OFULModel, oful_confidence_radius
from tofu_pov.subspace import corrected_covariance, sorted_eigendecomposition, threshold_rank

PSLBArmSource = Literal["full", "masked"]
PSLBRankSelection = Literal["fixed", "threshold"]
PSLBIntersectionMethod = Literal["projected_sampled", "ambient_sampled", "min_ucb", "sampled"]
PSLBIntersectionFallback = Literal["ambient_ucb", "min_ucb"]
PSLBWarmupPolicy = Literal["random", "zero_oful"]
PSLBSubspaceUpdate = Literal["once", "every_round"]
PSLBCovarianceRadiusSchedule = Callable[[int, int], float]


@dataclass(frozen=True)
class PSLBConfig:
    """Configuration for the Lale et al. PSLB baseline.

    This implements a practical finite-action version of Projected Stochastic
    Linear Bandit (PSLB). The policy uses full offered actions, recovers a PCA
    subspace from all offered actions, and approximates optimism over the
    projected-space confidence-set intersection by sampling plausible projected
    models. The zero-imputed masked variant keeps the same mechanics but uses
    masked arms as its action observations.
    """

    d: int
    m: int
    K: int
    T: int
    p: float = 1.0
    lambda_reg: float = 1.0
    delta: float = 0.05
    L: float = 1.0
    S: float = 1.0
    R: float = 0.01
    lambda_plus: float | None = None
    lambda_minus: float | None = None
    sigma2: float = 0.0
    alpha: float | None = None
    warmup_rounds: int | None = None
    warmup_policy: PSLBWarmupPolicy = "random"
    use_intersection: bool = True
    intersection_method: PSLBIntersectionMethod = "projected_sampled"
    intersection_sample_count: int = 256
    intersection_fallback: PSLBIntersectionFallback = "ambient_ucb"
    include_current_in_pca: bool = True
    subspace_update: PSLBSubspaceUpdate = "once"
    random_seed: int | None = None
    arm_source: PSLBArmSource = "full"
    rank_selection: PSLBRankSelection = "fixed"
    min_rank: int = 1
    max_rank: int | None = None
    rank_threshold_constant: float = 4.0
    covariance_radius_schedule: PSLBCovarianceRadiusSchedule | None = None

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be positive.")
        if self.m <= 0 or self.m > self.d:
            raise ValueError("m must be in [1, d].")
        if self.K <= 0 or self.T <= 0:
            raise ValueError("K and T must be positive.")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("p must be in (0, 1].")
        if self.lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive.")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1).")
        if self.L <= 0.0 or self.S < 0.0 or self.R < 0.0:
            raise ValueError("Require L > 0 and S, R >= 0.")
        if self.sigma2 < 0.0:
            raise ValueError("sigma2 must be nonnegative.")
        if self.warmup_rounds is not None and self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be nonnegative.")
        if self.warmup_policy not in {"random", "zero_oful"}:
            raise ValueError("warmup_policy must be 'random' or 'zero_oful'.")
        if self.intersection_method == "sampled":
            object.__setattr__(self, "intersection_method", "projected_sampled")
        if self.intersection_method not in {"projected_sampled", "ambient_sampled", "min_ucb"}:
            raise ValueError(
                "intersection_method must be 'projected_sampled', 'ambient_sampled', or 'min_ucb'."
            )
        if self.intersection_sample_count < 0:
            raise ValueError("intersection_sample_count must be nonnegative.")
        if self.intersection_fallback not in {"ambient_ucb", "min_ucb"}:
            raise ValueError("intersection_fallback must be 'ambient_ucb' or 'min_ucb'.")
        if self.subspace_update not in {"once", "every_round"}:
            raise ValueError("subspace_update must be 'once' or 'every_round'.")
        if self.lambda_plus is not None and self.lambda_plus <= 0.0:
            raise ValueError("lambda_plus must be positive when provided.")
        if self.lambda_minus is not None and self.lambda_minus <= 0.0:
            raise ValueError("lambda_minus must be positive when provided.")
        if self.alpha is not None and self.alpha <= 0.0:
            raise ValueError("alpha must be positive when provided.")
        if self.arm_source not in {"full", "masked"}:
            raise ValueError("arm_source must be 'full' or 'masked'.")
        if self.rank_selection not in {"fixed", "threshold"}:
            raise ValueError("rank_selection must be 'fixed' or 'threshold'.")
        if self.min_rank <= 0:
            raise ValueError("min_rank must be positive.")
        effective_max_rank = self.m if self.max_rank is None else self.max_rank
        if effective_max_rank <= 0 or effective_max_rank > self.d:
            raise ValueError("max_rank must be in [1, d].")
        if self.min_rank > effective_max_rank:
            raise ValueError("min_rank must be no larger than max_rank.")
        if self.rank_threshold_constant <= 0.0:
            raise ValueError("rank_threshold_constant must be positive.")


class OracleSubspaceOFUL:
    """OFUL that receives the true subspace and full, unmasked arms."""

    def __init__(
        self,
        U: NDArray[np.float64],
        lambda_reg: float = 1.0,
        S: float = 1.0,
        R: float = 0.01,
        delta: float = 0.05,
        beta_scale: float = 1.0,
    ):
        basis = np.asarray(U, dtype=float)
        if basis.ndim != 2:
            raise ValueError("U must have shape (d, m).")
        if beta_scale <= 0.0:
            raise ValueError("beta_scale must be positive.")
        self.U = basis
        self.lambda_reg = lambda_reg
        self.S = S
        self.R = R
        self.delta = delta
        self.beta_scale = beta_scale
        self.model = OFULModel(basis.shape[1], lambda_reg)
        self._pending_feature: NDArray[np.float64] | None = None
        self.last_scores: NDArray[np.float64] | None = None

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.model = OFULModel(self.U.shape[1], self.lambda_reg)
        self._pending_feature = None
        self.last_scores = None

    def observe(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
        full_arms: NDArray[np.float64] | None = None,
    ) -> int:
        del masked_arms, masks
        if full_arms is None:
            raise ValueError("OracleSubspaceOFUL requires full_arms.")
        features = np.asarray(full_arms, dtype=float) @ self.U
        beta = self.beta_scale * oful_confidence_radius(self.model, self.S, self.R, self.delta)
        action, scores = self.model.select(features, beta)
        self._pending_feature = features[action].copy()
        self.last_scores = scores
        return action

    def update(self, reward: float) -> None:
        if self._pending_feature is None:
            raise RuntimeError("Call observe before update.")
        self.model.update(self._pending_feature, reward)
        self._pending_feature = None


class PSLB:
    """Projected Stochastic Linear Bandit baseline from Lale et al.

    The paper's exact optimistic step maximizes over an intersection of the
    projected and ambient confidence sets. For finite action sets, this class
    approximates that inner optimization by sampling projected models from the
    m-dimensional intersection. Set `intersection_method="min_ucb"` to recover
    the older deterministic surrogate, `intersection_method="ambient_sampled"`
    to recover the earlier residual-lift sampler, or `use_intersection=False`
    to use only the projected confidence set.
    """

    def __init__(self, config: PSLBConfig):
        self.config = config
        self._base_seed = config.random_seed
        self.rng = np.random.default_rng(self._base_seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self._base_seed
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.decision_history: list[NDArray[np.float64]] = []
        self.selected_actions: list[NDArray[np.float64]] = []
        self.rewards: list[float] = []
        self.ambient_model = OFULModel(self.config.d, self.config.lambda_reg)
        self.warmup_model = OFULModel(self.config.d, self.config.lambda_reg)
        self.U_hat: NDArray[np.float64] | None = None
        self.eigenvalues: NDArray[np.float64] | None = None
        self.all_eigenvalues: NDArray[np.float64] | None = None
        self.active_m: int | None = None
        self.rank_history: list[int] = []
        self.rank_times: list[int] = []
        self.rank_threshold_t: float | None = None
        self.covariance_radius_t: float | None = None
        self.last_scores: NDArray[np.float64] | None = None
        self.last_projected_scores: NDArray[np.float64] | None = None
        self.last_ambient_scores: NDArray[np.float64] | None = None
        self.last_intersection_sample_count = 0
        self.last_intersection_feasible_count = 0
        self.last_intersection_fallback: str | None = None
        self._pending_action: int | None = None
        self._pending_full_arm: NDArray[np.float64] | None = None
        self._pending_decision_set: NDArray[np.float64] | None = None

    def observe(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
        full_arms: NDArray[np.float64] | None = None,
    ) -> int:
        del masks
        if self._pending_action is not None:
            raise RuntimeError("Call update(reward) before observing the next round.")

        X = self._resolve_arms(masked_arms, full_arms)
        round_t = self.t + 1
        if self.config.include_current_in_pca:
            self.decision_history.append(X.copy())
        else:
            self._pending_decision_set = X.copy()

        if round_t <= self.warmup_rounds:
            if self.config.warmup_policy == "zero_oful":
                beta = oful_confidence_radius(
                    self.warmup_model,
                    S=self.config.S,
                    R=self.config.R,
                    delta=self.config.delta,
                )
                action, scores = self.warmup_model.select(X, beta)
                self.last_scores = scores
                self.last_ambient_scores = scores
            else:
                action = int(self.rng.integers(self.config.K))
                self.last_scores = None
                self.last_ambient_scores = None
            self._set_pending(action, X[action])
            self.last_projected_scores = None
            return action

        should_update_subspace = self.config.subspace_update == "every_round" or self.U_hat is None
        if should_update_subspace:
            self._estimate_subspace_if_possible(
                current_arms=None if self.config.include_current_in_pca else X
            )
        if self.U_hat is None:
            action = int(self.rng.integers(self.config.K))
            self._set_pending(action, X[action])
            return action

        projected_model = self._projected_model()
        projected_features = self._projected_features(X)
        projected_beta = self._projected_beta(projected_model, round_t)
        projected_scores = projected_model.scores(projected_features, projected_beta)
        ambient_beta = oful_confidence_radius(
            self.ambient_model,
            S=self.config.S,
            R=self.config.R,
            delta=self.config.delta,
        )
        ambient_scores = self.ambient_model.scores(X, ambient_beta)

        if self.config.use_intersection:
            if self.config.intersection_method == "min_ucb":
                scores = np.minimum(projected_scores, ambient_scores)
                self.last_intersection_sample_count = 0
                self.last_intersection_feasible_count = 0
                self.last_intersection_fallback = None
            elif self.config.intersection_method == "ambient_sampled":
                scores = self._ambient_residual_intersection_scores(
                    arms=X,
                    projected_model=projected_model,
                    projected_beta=projected_beta,
                    projected_scores=projected_scores,
                    ambient_beta=ambient_beta,
                    ambient_scores=ambient_scores,
                )
            else:
                scores = self._sampled_intersection_scores(
                    arms=X,
                    projected_model=projected_model,
                    projected_beta=projected_beta,
                    projected_scores=projected_scores,
                    ambient_beta=ambient_beta,
                    ambient_scores=ambient_scores,
                )
        else:
            scores = projected_scores
            self.last_intersection_sample_count = 0
            self.last_intersection_feasible_count = 0
            self.last_intersection_fallback = None

        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("PSLB produced non-finite scores.")

        action = int(np.argmax(scores))
        self.last_scores = scores
        self.last_projected_scores = projected_scores
        self.last_ambient_scores = ambient_scores
        self._set_pending(action, X[action])
        return action

    def update(self, reward: float) -> None:
        if self._pending_action is None or self._pending_full_arm is None:
            raise RuntimeError("Call observe before update.")
        value = float(reward)
        self.selected_actions.append(self._pending_full_arm.copy())
        self.rewards.append(value)
        if not self.config.include_current_in_pca and self._pending_decision_set is not None:
            self.decision_history.append(self._pending_decision_set.copy())
        self.ambient_model.update(self._pending_full_arm, value)
        if self.t < self.warmup_rounds and self.config.warmup_policy == "zero_oful":
            self.warmup_model.update(self._pending_full_arm, value)
        self.t += 1
        self._pending_action = None
        self._pending_full_arm = None
        self._pending_decision_set = None

    @property
    def warmup_rounds(self) -> int:
        if self.config.warmup_rounds is not None:
            return self.config.warmup_rounds
        if not self._has_structural_constants:
            return 1
        return max(1, int(math.ceil(self._n_delta / self.config.K)))

    def state_dict(self) -> dict[str, object]:
        return {
            "t": self.t,
            "warmup_rounds": self.warmup_rounds,
            "warmup_policy": self.config.warmup_policy,
            "history_size": sum(batch.shape[0] for batch in self.decision_history),
            "n_selected": len(self.selected_actions),
            "warmup_updates": self.warmup_model.n_updates,
            "arm_source": self.config.arm_source,
            "rank_selection": self.config.rank_selection,
            "subspace_update": self.config.subspace_update,
            "intersection_method": self.config.intersection_method,
            "last_intersection_sample_count": self.last_intersection_sample_count,
            "last_intersection_feasible_count": self.last_intersection_feasible_count,
            "last_intersection_fallback": self.last_intersection_fallback,
            "active_m": self.active_m,
            "rank_history": list(self.rank_history),
            "rank_times": list(self.rank_times),
            "rank_threshold_t": self.rank_threshold_t,
            "covariance_radius_t": self.covariance_radius_t,
            "U_hat": None if self.U_hat is None else self.U_hat.copy(),
            "eigenvalues": None if self.eigenvalues is None else self.eigenvalues.copy(),
            "all_eigenvalues": (
                None if self.all_eigenvalues is None else self.all_eigenvalues.copy()
            ),
            "last_scores": None if self.last_scores is None else self.last_scores.copy(),
            "last_projected_scores": (
                None if self.last_projected_scores is None else self.last_projected_scores.copy()
            ),
            "last_ambient_scores": (
                None if self.last_ambient_scores is None else self.last_ambient_scores.copy()
            ),
        }

    def _resolve_arms(
        self,
        masked_arms: NDArray[np.float64],
        full_arms: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        if self.config.arm_source == "full":
            X = np.asarray(full_arms if full_arms is not None else masked_arms, dtype=float)
        else:
            X = np.asarray(masked_arms, dtype=float)
        expected = (self.config.K, self.config.d)
        if X.shape != expected:
            raise ValueError(f"PSLB requires arms with shape {expected}.")
        return X

    def _set_pending(self, action: int, full_arm: NDArray[np.float64]) -> None:
        self._pending_action = action
        self._pending_full_arm = np.asarray(full_arm, dtype=float).copy()

    def _estimate_subspace_if_possible(
        self,
        current_arms: NDArray[np.float64] | None,
    ) -> None:
        batches = list(self.decision_history)
        if current_arms is not None:
            batches.append(current_arms.copy())
        if not batches:
            return
        all_arms = np.vstack(batches)
        # Full PSLB intentionally uses all offered arms for PCA, as in Lale et
        # al. MaskedPSLB applies the same zero-filled covariance naively rather
        # than correcting for missing features; that is the baseline definition.
        covariance = corrected_covariance(all_arms, p=1.0)
        all_eigenvalues, all_eigenvectors = sorted_eigendecomposition(covariance)
        active_m = self._select_rank(all_eigenvalues, n_history=all_arms.shape[0])

        self.active_m = active_m
        self.rank_history.append(active_m)
        self.rank_times.append(self.t + 1)
        self.all_eigenvalues = all_eigenvalues
        self.eigenvalues = all_eigenvalues[:active_m]
        self.U_hat = all_eigenvectors[:, :active_m]

    def _select_rank(self, eigenvalues: NDArray[np.float64], n_history: int) -> int:
        max_rank = self.config.m if self.config.max_rank is None else self.config.max_rank
        if self.config.rank_selection == "fixed":
            self.covariance_radius_t = None
            self.rank_threshold_t = None
            return self.config.m

        radius = self._covariance_radius(n_history)
        threshold = self.config.rank_threshold_constant * radius
        self.covariance_radius_t = radius
        self.rank_threshold_t = threshold
        return threshold_rank(
            eigenvalues,
            threshold=threshold,
            min_rank=self.config.min_rank,
            max_rank=max_rank,
        )

    def _covariance_radius(self, n_history: int) -> float:
        if self.config.covariance_radius_schedule is not None:
            radius = float(self.config.covariance_radius_schedule(self.t + 1, n_history))
            if radius < 0.0:
                raise ValueError("covariance_radius_schedule must return nonnegative values.")
            return radius

        scale = self.config.lambda_plus if self.config.lambda_plus is not None else self.config.L * self.config.L
        log_term = math.log(2.0 * self.config.d * self.config.T / self.config.delta)
        return float(scale / self.config.p * math.sqrt(max(log_term, 0.0) / max(n_history, 1)))

    def _projected_model(self) -> OFULModel:
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")

        active_m = self.U_hat.shape[1]
        model = OFULModel(active_m, self.config.lambda_reg)
        if self.selected_actions:
            selected = np.vstack(self.selected_actions)
            projected_selected = selected @ self.U_hat
            for feature, reward in zip(projected_selected, self.rewards):
                model.update(feature, reward)
        return model

    def _projected_features(self, full_arms: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")
        return np.asarray(full_arms, dtype=float) @ self.U_hat

    def _projected_beta(self, model: OFULModel, round_t: int) -> float:
        beta = oful_confidence_radius(
            model,
            S=self.config.S,
            R=self.config.R,
            delta=self.config.delta,
        )
        beta += self._projection_bonus(round_t)
        return beta

    def _projected_scores(self, full_arms: NDArray[np.float64], round_t: int) -> NDArray[np.float64]:
        model = self._projected_model()
        features = self._projected_features(full_arms)
        beta = self._projected_beta(model, round_t)
        return model.scores(features, beta)

    def _sampled_intersection_scores(
        self,
        arms: NDArray[np.float64],
        projected_model: OFULModel,
        projected_beta: float,
        projected_scores: NDArray[np.float64],
        ambient_beta: float,
        ambient_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Approximate the paper PSLB optimistic step in projected space.

        The ambient confidence set is projected through U_hat.T, giving an
        m-dimensional ellipsoid that can be intersected directly with the
        projected OFUL confidence ellipsoid. Candidate projected parameters are
        sampled from both ellipsoids, filtered by both constraints, and scored
        only against projected arms. This avoids adding any ad-hoc residual
        reward term outside the recovered subspace.
        """
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")

        X = np.asarray(arms, dtype=float)
        projected_features = X @ self.U_hat
        projected_center = projected_model.theta_hat
        ambient_center, ambient_shape = self._projected_ambient_confidence()

        candidates = [
            projected_center,
            ambient_center,
        ]
        candidates.extend(
            self._ellipsoid_action_directed_candidates(
                projected_features,
                projected_model.V,
                projected_center,
                projected_beta,
            )
        )
        candidates.extend(
            self._ellipsoid_action_directed_candidates(
                projected_features,
                ambient_shape,
                ambient_center,
                ambient_beta,
            )
        )

        random_budget = self.config.intersection_sample_count
        projected_random = random_budget // 2
        ambient_random = random_budget - projected_random
        candidates.extend(
            self._sample_confidence_ball(
                projected_center,
                projected_model.V,
                projected_beta,
                projected_random,
            )
        )
        candidates.extend(
            self._sample_confidence_ball(
                ambient_center,
                ambient_shape,
                ambient_beta,
                ambient_random,
            )
        )

        active_m = self.U_hat.shape[1]
        candidate_matrix = np.vstack(candidates) if candidates else np.empty((0, active_m))
        self.last_intersection_sample_count = int(candidate_matrix.shape[0])
        feasible = self._filter_projected_intersection_candidates(
            candidate_matrix,
            projected_model,
            projected_beta,
            ambient_center,
            ambient_shape,
            ambient_beta,
        )
        self.last_intersection_feasible_count = int(feasible.shape[0])
        self.last_intersection_fallback = None

        if feasible.shape[0] == 0:
            self.last_intersection_fallback = self.config.intersection_fallback
            if self.config.intersection_fallback == "ambient_ucb":
                return ambient_scores
            return np.minimum(projected_scores, ambient_scores)

        scores = np.max(feasible @ projected_features.T, axis=0)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("Sampled PSLB produced non-finite scores.")
        return scores

    def _projected_ambient_confidence(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")
        center = self.ambient_model.theta_hat @ self.U_hat
        solved = self.ambient_model.V_inv @ self.U_hat
        projected_covariance = self.U_hat.T @ solved
        projected_covariance = 0.5 * (projected_covariance + projected_covariance.T)
        shape = np.linalg.solve(projected_covariance, np.eye(projected_covariance.shape[0]))
        shape = 0.5 * (shape + shape.T)
        return center, shape

    def _filter_projected_intersection_candidates(
        self,
        candidates: NDArray[np.float64],
        projected_model: OFULModel,
        projected_beta: float,
        ambient_center: NDArray[np.float64],
        ambient_shape: NDArray[np.float64],
        ambient_beta: float,
    ) -> NDArray[np.float64]:
        active_m = projected_model.dimension
        if candidates.size == 0:
            return candidates.reshape(0, active_m)

        projected_diff = candidates - projected_model.theta_hat
        projected_norms = self._model_norms(projected_diff, projected_model.V)
        ambient_diff = candidates - ambient_center
        ambient_norms = self._model_norms(ambient_diff, ambient_shape)
        tolerance = 1e-8
        feasible = (projected_norms <= projected_beta + tolerance) & (
            ambient_norms <= ambient_beta + tolerance
        )
        return candidates[feasible]

    def _ambient_residual_intersection_scores(
        self,
        arms: NDArray[np.float64],
        projected_model: OFULModel,
        projected_beta: float,
        projected_scores: NDArray[np.float64],
        ambient_beta: float,
        ambient_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")

        X = np.asarray(arms, dtype=float)
        ambient_center = self.ambient_model.theta_hat
        projected_center = projected_model.theta_hat
        projected_residual = ambient_center - self.U_hat @ (self.U_hat.T @ ambient_center)

        candidates = [
            ambient_center,
            self.U_hat @ projected_center + projected_residual,
        ]
        candidates.extend(
            self._action_directed_candidates(
                X,
                self.ambient_model,
                ambient_center,
                ambient_beta,
            )
        )
        candidates.extend(
            self._lift_projected_candidates(
                self._action_directed_candidates(
                    X @ self.U_hat,
                    projected_model,
                    projected_center,
                    projected_beta,
                ),
                projected_residual,
            )
        )

        random_budget = self.config.intersection_sample_count
        ambient_random = random_budget // 2
        projected_random = random_budget - ambient_random
        candidates.extend(
            self._sample_confidence_ball(
                ambient_center,
                self.ambient_model.V,
                ambient_beta,
                ambient_random,
            )
        )
        candidates.extend(
            self._lift_projected_candidates(
                self._sample_confidence_ball(
                    projected_center,
                    projected_model.V,
                    projected_beta,
                    projected_random,
                ),
                projected_residual,
            )
        )

        candidate_matrix = np.vstack(candidates) if candidates else np.empty((0, self.config.d))
        self.last_intersection_sample_count = int(candidate_matrix.shape[0])
        feasible = self._filter_intersection_candidates(
            candidate_matrix,
            projected_model,
            projected_beta,
            ambient_beta,
        )
        self.last_intersection_feasible_count = int(feasible.shape[0])
        self.last_intersection_fallback = None

        if feasible.shape[0] == 0:
            self.last_intersection_fallback = self.config.intersection_fallback
            if self.config.intersection_fallback == "ambient_ucb":
                return ambient_scores
            return np.minimum(projected_scores, ambient_scores)

        scores = np.max(feasible @ X.T, axis=0)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("Sampled PSLB produced non-finite scores.")
        return scores

    def _filter_intersection_candidates(
        self,
        candidates: NDArray[np.float64],
        projected_model: OFULModel,
        projected_beta: float,
        ambient_beta: float,
    ) -> NDArray[np.float64]:
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")
        if candidates.size == 0:
            return candidates.reshape(0, self.config.d)

        ambient_diff = candidates - self.ambient_model.theta_hat
        ambient_norms = self._model_norms(ambient_diff, self.ambient_model.V)
        projected_diff = candidates @ self.U_hat - projected_model.theta_hat
        projected_norms = self._model_norms(projected_diff, projected_model.V)
        tolerance = 1e-8
        feasible = (ambient_norms <= ambient_beta + tolerance) & (
            projected_norms <= projected_beta + tolerance
        )
        return candidates[feasible]

    def _action_directed_candidates(
        self,
        arms: NDArray[np.float64],
        model: OFULModel,
        center: NDArray[np.float64],
        beta: float,
    ) -> list[NDArray[np.float64]]:
        return self._ellipsoid_action_directed_candidates(arms, model.V, center, beta)

    def _ellipsoid_action_directed_candidates(
        self,
        arms: NDArray[np.float64],
        V: NDArray[np.float64],
        center: NDArray[np.float64],
        beta: float,
    ) -> list[NDArray[np.float64]]:
        candidates = []
        for arm in np.asarray(arms, dtype=float):
            solved = np.linalg.solve(V, arm)
            uncertainty = float(np.sqrt(max(arm @ solved, 0.0)))
            if uncertainty <= 0.0:
                candidates.append(center.copy())
            else:
                candidates.append(center + beta * solved / uncertainty)
        return candidates

    def _sample_confidence_ball(
        self,
        center: NDArray[np.float64],
        V: NDArray[np.float64],
        beta: float,
        count: int,
    ) -> list[NDArray[np.float64]]:
        if count <= 0:
            return []
        dimension = center.shape[0]
        directions = self.rng.normal(size=(count, dimension))
        norms = np.linalg.norm(directions, axis=1)
        nonzero = norms > 0.0
        if not np.any(nonzero):
            return []
        directions = directions[nonzero] / norms[nonzero, None]
        radii = self.rng.random(directions.shape[0]) ** (1.0 / max(dimension, 1))
        unit_ball = directions * radii[:, None]
        chol = np.linalg.cholesky(V)
        offsets = np.linalg.solve(chol.T, (beta * unit_ball).T).T
        return [center + offset for offset in offsets]

    def _lift_projected_candidates(
        self,
        projected_candidates: list[NDArray[np.float64]],
        residual: NDArray[np.float64],
    ) -> list[NDArray[np.float64]]:
        if self.U_hat is None:
            raise RuntimeError("No estimated subspace is available.")
        return [self.U_hat @ candidate + residual for candidate in projected_candidates]

    @staticmethod
    def _model_norms(diff: NDArray[np.float64], V: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.einsum("ij,ij->i", diff @ V, diff)
        return np.sqrt(np.maximum(values, 0.0))

    def _projection_bonus(self, round_t: int) -> float:
        if not self._has_structural_constants:
            return 0.0
        log_term = math.log(1.0 + round_t * self.config.L * self.config.L / (self.config.m * self.config.lambda_reg))
        gamma = (
            self.config.L
            * self.config.L
            / self.config.lambda_reg
            * math.log(1.0 + self.config.L * self.config.L / self.config.lambda_reg)
        )
        return float(
            self.config.L
            * self.config.S
            * self._phi_delta
            * math.sqrt(max(gamma * self.config.m * log_term, 0.0))
        )

    @property
    def _has_structural_constants(self) -> bool:
        return (
            self.config.lambda_plus is not None
            and self.config.lambda_minus is not None
            and self.config.alpha is not None
        )

    @property
    def _gamma_structure(self) -> float:
        if not self._has_structural_constants:
            return 0.0
        assert self.config.lambda_plus is not None
        assert self.config.lambda_minus is not None
        gx = self.config.lambda_plus / self.config.lambda_minus
        gpsi = self.config.sigma2 / self.config.lambda_minus
        return float(2.0 * gpsi + 4.0 * math.sqrt(max(gx * gpsi, 0.0)))

    @property
    def _phi_delta(self) -> float:
        assert self.config.alpha is not None
        return float(
            2.0
            * self._gamma_structure
            * math.sqrt(
                self.config.alpha
                / self.config.K
                * max(math.log(2.0 * self.config.d / self.config.delta), 0.0)
            )
        )

    @property
    def _n_delta(self) -> float:
        assert self.config.lambda_plus is not None
        assert self.config.lambda_minus is not None
        assert self.config.alpha is not None
        gx = self.config.lambda_plus / self.config.lambda_minus
        first = self._gamma_structure * math.sqrt(
            max(math.log(2.0 * self.config.d / self.config.delta), 0.0)
        )
        second = math.sqrt(max(2.0 * gx * math.log(max(self.config.m, 2) / self.config.delta), 0.0))
        return float(4.0 * self.config.alpha * (first + second) ** 2)


class MaskedPSLB(PSLB):
    """Zero-imputed partial-observation adaptation of PSLB.

    This is not the original full-information PSLB setting. It is a fairer
    missing-feature baseline that applies PSLB machinery directly to the masked
    zero-filled arms.
    """

    def __init__(self, config: PSLBConfig):
        masked_config = PSLBConfig(
            d=config.d,
            m=config.m,
            K=config.K,
            T=config.T,
            p=config.p,
            lambda_reg=config.lambda_reg,
            delta=config.delta,
            L=config.L,
            S=config.S,
            R=config.R,
            lambda_plus=config.lambda_plus,
            lambda_minus=config.lambda_minus,
            sigma2=config.sigma2,
            alpha=config.alpha,
            warmup_rounds=config.warmup_rounds,
            warmup_policy=config.warmup_policy,
            use_intersection=config.use_intersection,
            intersection_method=config.intersection_method,
            intersection_sample_count=config.intersection_sample_count,
            intersection_fallback=config.intersection_fallback,
            include_current_in_pca=config.include_current_in_pca,
            subspace_update=config.subspace_update,
            random_seed=config.random_seed,
            arm_source="masked",
            rank_selection=config.rank_selection,
            min_rank=config.min_rank,
            max_rank=config.max_rank,
            rank_threshold_constant=config.rank_threshold_constant,
            covariance_radius_schedule=config.covariance_radius_schedule,
        )
        super().__init__(masked_config)


class ZeroImputedOFUL:
    """OFUL in the ambient dimension using masked vectors with zeros."""

    def __init__(
        self,
        d: int,
        lambda_reg: float = 1.0,
        S: float = 1.0,
        R: float = 0.01,
        delta: float = 0.05,
        beta_scale: float = 1.0,
    ):
        if beta_scale <= 0.0:
            raise ValueError("beta_scale must be positive.")
        self.d = d
        self.lambda_reg = lambda_reg
        self.S = S
        self.R = R
        self.delta = delta
        self.beta_scale = beta_scale
        self.model = OFULModel(d, lambda_reg)
        self._pending_feature: NDArray[np.float64] | None = None
        self.last_scores: NDArray[np.float64] | None = None

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.model = OFULModel(self.d, self.lambda_reg)
        self._pending_feature = None
        self.last_scores = None

    def observe(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
        full_arms: NDArray[np.float64] | None = None,
    ) -> int:
        del masks, full_arms
        features = np.asarray(masked_arms, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.d:
            raise ValueError("masked_arms must have shape (K, d).")
        beta = self.beta_scale * oful_confidence_radius(self.model, self.S, self.R, self.delta)
        action, scores = self.model.select(features, beta)
        self._pending_feature = features[action].copy()
        self.last_scores = scores
        return action

    def update(self, reward: float) -> None:
        if self._pending_feature is None:
            raise RuntimeError("Call observe before update.")
        self.model.update(self._pending_feature, reward)
        self._pending_feature = None


class FullInformationOFUL:
    """OFUL reference policy that observes full, unmasked arms."""

    def __init__(
        self,
        d: int,
        lambda_reg: float = 1.0,
        S: float = 1.0,
        R: float = 0.01,
        delta: float = 0.05,
        beta_scale: float = 1.0,
    ):
        if beta_scale <= 0.0:
            raise ValueError("beta_scale must be positive.")
        self.d = d
        self.lambda_reg = lambda_reg
        self.S = S
        self.R = R
        self.delta = delta
        self.beta_scale = beta_scale
        self.model = OFULModel(d, lambda_reg)
        self._pending_feature: NDArray[np.float64] | None = None
        self.last_scores: NDArray[np.float64] | None = None

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.model = OFULModel(self.d, self.lambda_reg)
        self._pending_feature = None
        self.last_scores = None

    def observe(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
        full_arms: NDArray[np.float64] | None = None,
    ) -> int:
        del masked_arms, masks
        if full_arms is None:
            raise ValueError("FullInformationOFUL requires full_arms.")
        features = np.asarray(full_arms, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.d:
            raise ValueError("full_arms must have shape (K, d).")
        beta = self.beta_scale * oful_confidence_radius(self.model, self.S, self.R, self.delta)
        action, scores = self.model.select(features, beta)
        self._pending_feature = features[action].copy()
        self.last_scores = scores
        return action

    def update(self, reward: float) -> None:
        if self._pending_feature is None:
            raise RuntimeError("Call observe before update.")
        self.model.update(self._pending_feature, reward)
        self._pending_feature = None


class RandomPolicy:
    """Uniform random arm selection baseline."""

    def __init__(self, K: int, seed: int | None = None):
        if K <= 0:
            raise ValueError("K must be positive.")
        self.K = K
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed)

    def observe(
        self,
        masked_arms: NDArray[np.float64],
        masks: NDArray[np.bool_],
        full_arms: NDArray[np.float64] | None = None,
    ) -> int:
        del masks, full_arms
        if np.asarray(masked_arms).shape[0] != self.K:
            raise ValueError("masked_arms first dimension must equal K.")
        return int(self.rng.integers(self.K))

    def update(self, reward: float) -> None:
        del reward
