# TOFU-POV

Standalone NumPy implementation of an epoch-wise TOFU-POV learner for partially
observed low-rank contextual bandits.

The package includes:

- `TOFUPOV`, the learner with burn-in, doubling epochs, corrected covariance
  subspace estimation, imputation, and within-epoch OFUL.
- Optional threshold-based adaptive rank selection at epoch starts, where `m`
  can be treated as a maximum rank instead of a known exact rank.
- Synthetic low-rank bandit environments for controlled regret experiments.
- Baselines: oracle-subspace OFUL, zero-imputed OFUL, and random policy.
- PSLB, the Lale et al. projected stochastic linear bandit baseline for
  full-information action sets, with projected-space sampled optimistic model
  selection over the projected/ambient confidence-set intersection, plus a
  masked zero-imputed PSLB adaptation.
- Array-backed experiment utilities for real-world-style datasets that can be
  represented as candidate arms plus reward labels.

## Quick Start

```python
from tofu_pov import SyntheticLowRankBanditEnv, TOFUPOV, TOFUPOVConfig, run_bandit

env = SyntheticLowRankBanditEnv(d=8, m=2, K=4, p=0.8, T=100, noise_std=0.01, seed=0)

config = TOFUPOVConfig(
    d=8,
    m=2,
    K=4,
    p=0.8,
    lambda_reg=1.0,
    t_b=10,
    T=100,
    delta=0.05,
    L=5.0,
    S=1.0,
    R=0.05,
    lambda_1=2.0,
    lambda_m=0.5,
    M=1.0,
    impute_ridge=1e-8,
    random_seed=1,
)

result = run_bandit(TOFUPOV(config), env, seed=0)
print(result.cumulative_regret[-1])
```

Run tests with:

```bash
pytest
```

Run the synthetic benchmark suite with:

```bash
PYTHONPATH=. python3 experiments/run_synthetic_benchmarks.py
```

Run the rank-misspecification/adaptive-rank benchmark with:

```bash
PYTHONPATH=. python3 experiments/run_rank_adaptation_experiments.py
```

## Reference

This repository accompanies the paper:

> Gautam Dasarathy, Vineet Gattani, and Lalit Jain.
> *Stochastic Linear Bandits with Partially Observed Actions.*
> arXiv preprint [arXiv:2607.08971](https://arxiv.org/abs/2607.08971), 2026.

```bibtex
@article{dasarathy2026stochastic,
  title={Stochastic Linear Bandits with Partially Observed Actions},
  author={Dasarathy, Gautam and Gattani, Vineet and Jain, Lalit},
  journal={arXiv preprint arXiv:2607.08971},
  year={2026}
}
```

## License

MIT (see `LICENSE`).
