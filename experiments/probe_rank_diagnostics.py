"""Rank diagnostics on cached CNN MNIST bandit data.

Three diagnostics:
1. Singular-value spectrum of the per-round arm feature matrix.
2. Optimal-arm recovery rate when only top-k feature SVD components are kept.
3. Active-rank trajectory of TOFU-adaptive across t (from existing CSV).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path("data/cnn_image_cache")
TRAJ = Path("results/cnn_image_narrow/cnn_image_trajectories.csv")
RANK_GRID = (1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 200, 300)
T_MILESTONES = (1, 25, 50, 100, 200, 400, 800)


def load_seed(seed: int) -> tuple[np.ndarray, np.ndarray]:
    matches = sorted(CACHE.glob(f"mnist_openml_cnn_m20_d300_T800_seed{seed}_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No cache for seed={seed}")
    payload = np.load(matches[0], allow_pickle=True)
    return payload["full_arms"], payload["rewards"]


def feature_singular_spectrum(seeds: list[int]) -> np.ndarray:
    """Return mean (over seeds) of top singular values of centered (T*K, d) matrix."""
    sv_stack = []
    for seed in seeds:
        arms, _ = load_seed(seed)
        flat = arms.reshape(-1, arms.shape[-1])
        flat = flat - flat.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(flat, compute_uv=False)
        sv_stack.append(sv)
    return np.stack(sv_stack).mean(axis=0)


def optimal_arm_recovery_curve(
    seeds: list[int],
    rank_grid: tuple[int, ...],
) -> dict[int, list[float]]:
    """For each k, fit linear reward predictor in top-k SVD subspace and report
    fraction of rounds where argmax_k matches the true optimum."""
    by_rank: dict[int, list[float]] = defaultdict(list)
    for seed in seeds:
        arms, rewards = load_seed(seed)
        T, K, d = arms.shape
        X = arms.reshape(-1, d)
        y = rewards.reshape(-1)
        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - y.mean()
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        feat_full = U * S  # (T*K, d), columns are top-k principal feature scores
        true_opt = np.argmax(rewards, axis=1)
        for k in rank_grid:
            if k > d:
                continue
            Xk = feat_full[:, :k]
            coef, *_ = np.linalg.lstsq(Xk, yc, rcond=None)
            pred = (Xk @ coef).reshape(T, K)
            chosen = np.argmax(pred, axis=1)
            recovery = float(np.mean(chosen == true_opt))
            by_rank[k].append(recovery)
    return by_rank


def expected_regret_curve(
    seeds: list[int],
    rank_grid: tuple[int, ...],
) -> dict[int, list[float]]:
    """For each k, expected per-round regret of the rank-k linear predictor."""
    by_rank: dict[int, list[float]] = defaultdict(list)
    for seed in seeds:
        arms, rewards = load_seed(seed)
        T, K, d = arms.shape
        X = arms.reshape(-1, d)
        y = rewards.reshape(-1)
        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - y.mean()
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        feat_full = U * S
        for k in rank_grid:
            if k > d:
                continue
            Xk = feat_full[:, :k]
            coef, *_ = np.linalg.lstsq(Xk, yc, rcond=None)
            pred = (Xk @ coef).reshape(T, K)
            chosen = np.argmax(pred, axis=1)
            regret = float(np.mean(rewards.max(axis=1) - rewards[np.arange(T), chosen]))
            by_rank[k].append(regret)
    return by_rank


def active_rank_trajectory_from_csv(
    path: Path,
    method: str = "TOFU full-history replay adaptive-rank",
    t_milestones: tuple[int, ...] = T_MILESTONES,
) -> dict[float, dict[int, float]]:
    """Return mean active_rank per (p, t) for the given method."""
    by_pt: dict[tuple[float, int], list[int]] = defaultdict(list)
    with path.open() as h:
        for row in csv.DictReader(h):
            if row["method"] != method:
                continue
            if row["active_rank"] == "" or row["split"] != "report":
                continue
            t = int(row["t"])
            if t not in t_milestones:
                continue
            p = float(row["p"])
            by_pt[(p, t)].append(int(row["active_rank"]))
    out: dict[float, dict[int, float]] = defaultdict(dict)
    for (p, t), values in by_pt.items():
        out[p][t] = float(np.mean(values))
    return out


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    print("=" * 64)
    print("Diagnostic 1: feature singular spectrum (avg over 5 seeds)")
    print("=" * 64)
    sv = feature_singular_spectrum(seeds)
    sv_norm = sv / sv[0]
    energy = sv ** 2
    cum = np.cumsum(energy) / energy.sum()
    for k in (1, 2, 3, 5, 10, 15, 20, 25, 30, 50, 100):
        print(f"  rank-{k:>3d}: sigma_k/sigma_1 = {sv_norm[k-1]:.4f}  cum.var = {cum[k-1]:.4f}")

    print()
    print("=" * 64)
    print("Diagnostic 2: optimal-arm recovery and per-round regret vs feature rank")
    print("(rank-k linear reward predictor on top-k SVD features, evaluated by")
    print("how often it picks the true best arm — full-info, no exploration)")
    print("=" * 64)
    recovery = optimal_arm_recovery_curve(seeds, RANK_GRID)
    regret = expected_regret_curve(seeds, RANK_GRID)
    print(f"  {'rank':>5s} | {'recovery':>10s} | {'regret/round':>12s}")
    for k in RANK_GRID:
        if k not in recovery:
            continue
        rec = float(np.mean(recovery[k]))
        reg = float(np.mean(regret[k]))
        print(f"  {k:>5d} | {rec:>10.4f} | {reg:>12.4f}")
    print(f"  random arm baseline: recovery=1/K=0.10, regret=(K-1)/K=0.90")
    print(f"  oracle (full-info best linear predictor at d=300) is the last row")

    print()
    print("=" * 64)
    print("Diagnostic 3: TOFU-adaptive active rank vs round t (mean over seeds)")
    print("=" * 64)
    traj = active_rank_trajectory_from_csv(TRAJ)
    if not traj:
        print("  no trajectory data found")
        return
    ps = sorted(traj)
    print(f"  t       : " + "  ".join(f"{t:>5d}" for t in T_MILESTONES))
    for p in ps:
        row = traj[p]
        cells = []
        for t in T_MILESTONES:
            if t in row:
                cells.append(f"{row[t]:>5.2f}")
            else:
                cells.append(f"{'-':>5s}")
        print(f"  p={p:5g}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
