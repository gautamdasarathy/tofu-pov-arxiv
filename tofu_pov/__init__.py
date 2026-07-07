"""TOFU-POV package public API."""

from tofu_pov.baselines import (
    FullInformationOFUL,
    MaskedPSLB,
    OracleSubspaceOFUL,
    PSLB,
    PSLBConfig,
    RandomPolicy,
    ZeroImputedOFUL,
)
from tofu_pov.config import TOFUPOVConfig
from tofu_pov.cnn_image_bandit import (
    load_cnn_image_classification_full_dataset,
    make_mock_cnn_lowrank_full_dataset,
)
from tofu_pov.envs import BanditEnv, SyntheticLowRankBanditEnv
from tofu_pov.experiments import ExperimentResult, compare_policies, run_bandit
from tofu_pov.imputation import ImputationError, impute_arm, impute_arms
from tofu_pov.image_bandit import (
    ImageClassificationBanditDataset,
    ImageClassificationFullDataset,
    load_image_classification_full_dataset,
    make_image_classification_full_dataset,
    mask_image_classification_dataset,
)
from tofu_pov.learner import TOFUPOV
from tofu_pov.movielens import (
    MovieLensBanditDataset,
    MovieLensRawData,
    load_movielens_bandit_dataset,
    load_movielens_100k,
    make_movielens_bandit_dataset,
    make_synthetic_movielens_raw,
)
from tofu_pov.product_context_text import (
    load_text_product_context_full_dataset,
    make_mock_text_product_context_full_dataset,
    make_text_product_context_full_dataset,
)
from tofu_pov.real_world import ArrayBanditEnv
from tofu_pov.real_world_datasets import (
    DatasetUnavailableError,
    RealFeatureMatrix,
    RealWorldBanditDataset,
    load_real_feature_matrix,
    load_real_world_bandit_dataset,
    make_classification_bandit_dataset,
)
from tofu_pov.real_feature_synthetic import (
    RealFeatureSyntheticDataset,
    load_real_feature_synthetic_dataset,
    make_real_feature_synthetic_dataset,
)
from tofu_pov.subspace import (
    corrected_covariance,
    estimate_subspace,
    estimate_subspace_from_covariance,
    sorted_eigendecomposition,
    subspace_distance,
    threshold_rank,
)

__all__ = [
    "ArrayBanditEnv",
    "BanditEnv",
    "DatasetUnavailableError",
    "ExperimentResult",
    "FullInformationOFUL",
    "ImputationError",
    "ImageClassificationBanditDataset",
    "ImageClassificationFullDataset",
    "MaskedPSLB",
    "MovieLensBanditDataset",
    "MovieLensRawData",
    "OracleSubspaceOFUL",
    "PSLB",
    "PSLBConfig",
    "RandomPolicy",
    "RealFeatureMatrix",
    "RealFeatureSyntheticDataset",
    "RealWorldBanditDataset",
    "SyntheticLowRankBanditEnv",
    "TOFUPOV",
    "TOFUPOVConfig",
    "ZeroImputedOFUL",
    "compare_policies",
    "corrected_covariance",
    "estimate_subspace",
    "estimate_subspace_from_covariance",
    "impute_arm",
    "impute_arms",
    "load_image_classification_full_dataset",
    "load_cnn_image_classification_full_dataset",
    "load_real_feature_matrix",
    "load_real_feature_synthetic_dataset",
    "load_movielens_100k",
    "load_movielens_bandit_dataset",
    "load_text_product_context_full_dataset",
    "load_real_world_bandit_dataset",
    "make_classification_bandit_dataset",
    "make_image_classification_full_dataset",
    "make_mock_cnn_lowrank_full_dataset",
    "make_mock_text_product_context_full_dataset",
    "make_movielens_bandit_dataset",
    "make_real_feature_synthetic_dataset",
    "make_synthetic_movielens_raw",
    "make_text_product_context_full_dataset",
    "mask_image_classification_dataset",
    "run_bandit",
    "sorted_eigendecomposition",
    "subspace_distance",
    "threshold_rank",
]
