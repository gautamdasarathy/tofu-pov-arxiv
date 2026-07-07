import numpy as np

from tofu_pov import (
    OracleSubspaceOFUL,
    RandomPolicy,
    SyntheticLowRankBanditEnv,
    TOFUPOV,
    TOFUPOVConfig,
    run_bandit,
)
from tofu_pov.real_world import ArrayBanditEnv


def test_end_to_end_synthetic_regret_is_finite_and_beats_random_on_easy_instance():
    T = 80

    def make_env(seed):
        return SyntheticLowRankBanditEnv(d=6, m=2, K=4, p=1.0, T=T, noise_std=0.0, seed=seed)

    config = TOFUPOVConfig(
        d=6,
        m=2,
        K=4,
        p=1.0,
        lambda_reg=1.0,
        t_b=8,
        T=T,
        delta=0.05,
        L=4.0,
        S=1.0,
        R=0.01,
        lambda_1=1.5,
        lambda_m=0.5,
        M=1.0,
        c_b=0.0,
        random_seed=7,
    )

    tofu = run_bandit(TOFUPOV(config), make_env(10), seed=10, policy_seed=7)
    random = run_bandit(RandomPolicy(K=4, seed=7), make_env(10), seed=10, policy_seed=7)

    assert np.all(np.isfinite(tofu.cumulative_regret))
    assert tofu.cumulative_regret[-1] < random.cumulative_regret[-1]


def test_oracle_subspace_oful_is_strong_on_synthetic_instance():
    T = 60
    env = SyntheticLowRankBanditEnv(d=6, m=2, K=4, p=0.8, T=T, noise_std=0.0, seed=11)
    U = env.U.copy()

    result = run_bandit(
        OracleSubspaceOFUL(U, lambda_reg=1.0, S=1.0, R=0.01, delta=0.05),
        env,
        seed=11,
    )

    assert np.isfinite(result.cumulative_regret[-1])
    assert result.cumulative_regret[-1] < 20.0


def test_array_bandit_env_supports_real_world_style_reward_matrices():
    masked = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, 0.0]],
        ]
    )
    masks = np.ones_like(masked, dtype=bool)
    rewards = np.array([[1.0, 0.0], [0.2, 0.4]])
    env = ArrayBanditEnv(masked_arms=masked, masks=masks, rewards=rewards)

    policy = RandomPolicy(K=2, seed=0)
    result = run_bandit(policy, env, seed=0, policy_seed=0)

    assert result.actions.shape == (2,)
    assert result.rewards.shape == (2,)
    assert np.all(result.regrets >= 0.0)
