"""Create figures for product-context text bandit experiments."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_plot_cache_dir = tempfile.mkdtemp(prefix="tofu-text-product-plot-cache-")
os.environ.setdefault("MPLCONFIGDIR", _plot_cache_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/text_product_context")
FIGURES_DIR = Path("figures/text_product_context")

FAIR_METHOD_ORDER = [
    "TOFU full-history replay fixed-rank best-val",
    "TOFU full-history replay adaptive-rank",
    "TOFU first-epoch replay fixed-rank best-val",
    "Zero-imputed OFUL",
    "Masked PSLB fixed-rank best-val",
    "Masked PSLB adaptive-rank",
]

APPENDIX_METHOD_ORDER = FAIR_METHOD_ORDER + ["Random", "Full-info OFUL", "Full-info PSLB"]

METHOD_STYLE = {
    "TOFU full-history replay fixed-rank best-val": {"color": "#1B7837", "marker": "o"},
    "TOFU full-history replay adaptive-rank": {"color": "#00441B", "marker": "s"},
    "TOFU first-epoch replay fixed-rank best-val": {"color": "#7FBF7B", "marker": "^"},
    "Zero-imputed OFUL": {"color": "#2166AC", "marker": "v"},
    "Masked PSLB fixed-rank best-val": {"color": "#B2182B", "marker": "D"},
    "Masked PSLB adaptive-rank": {"color": "#762A83", "marker": "P"},
    "Random": {"color": "#737373", "marker": "x"},
    "Full-info OFUL": {"color": "#67A9CF", "marker": "*"},
    "Full-info PSLB": {"color": "#EF8A62", "marker": "h"},
}


def load_csv(results_dir: Path, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save_figure(fig, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def representative_p(values) -> float:
    values = sorted(float(v) for v in values)
    for target in (0.4, 0.3, 0.2):
        if target in values:
            return target
    return values[len(values) // 2]


def plot_missingness_sweep(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "text_product_summary.csv")
    df = df[df["method"].isin(FAIR_METHOD_ORDER)]
    scenarios = list(dict.fromkeys(df["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 4.2), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        sub = df[df["scenario"] == scenario]
        for method in FAIR_METHOD_ORDER:
            group = sub[sub["method"] == method].sort_values("p")
            if group.empty:
                continue
            style = METHOD_STYLE[method]
            ax.errorbar(
                group["p"],
                group["mean_final_regret"],
                yerr=group["stderr_final_regret"],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                capsize=3,
            )
        ax.set_title(scenario)
        ax.set_xlabel("Observation probability p")
        ax.set_ylabel("Final cumulative regret")
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save_figure(fig, figures_dir, "text_product_missingness_sweep")


def plot_regret_over_time(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "text_product_trajectories.csv")
    df = df[df["method"].isin(FAIR_METHOD_ORDER)]
    for scenario in dict.fromkeys(df["scenario"]):
        sub_s = df[df["scenario"] == scenario]
        p_value = representative_p(sub_s["p"].unique())
        sub = sub_s[sub_s["p"] == p_value]
        grouped = (
            sub.groupby(["method", "t"], as_index=False)["cumulative_regret"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method in FAIR_METHOD_ORDER:
            group = grouped[grouped["method"] == method]
            if group.empty:
                continue
            style = METHOD_STYLE[method]
            ax.plot(group["t"], group["mean"], label=method, color=style["color"], linewidth=2)
            ax.fill_between(
                group["t"].to_numpy(),
                (group["mean"] - group["sem"].fillna(0.0)).to_numpy(),
                (group["mean"] + group["sem"].fillna(0.0)).to_numpy(),
                color=style["color"],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(f"{scenario}: regret over time at p={p_value:g}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative regret")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        stem_p = str(p_value).replace(".", "p")
        save_figure(fig, figures_dir, f"text_product_regret_over_time_p{stem_p}")


def plot_rank_diagnostics(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "text_product_trajectories.csv")
    df["active_rank_numeric"] = pd.to_numeric(df["active_rank"], errors="coerce")
    df = df[
        df["method"].isin(
            ["TOFU full-history replay adaptive-rank", "Masked PSLB adaptive-rank"]
        )
        & df["active_rank_numeric"].notna()
    ]
    if df.empty:
        return
    p_value = representative_p(df["p"].unique())
    df = df[df["p"] == p_value]
    grouped = (
        df.groupby(["scenario", "method", "t"], as_index=False)["active_rank_numeric"]
        .mean()
        .rename(columns={"active_rank_numeric": "mean_rank"})
    )
    scenarios = list(dict.fromkeys(grouped["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.0 * len(scenarios), 3.8), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        sub = grouped[grouped["scenario"] == scenario]
        for method in ["TOFU full-history replay adaptive-rank", "Masked PSLB adaptive-rank"]:
            group = sub[sub["method"] == method]
            if group.empty:
                continue
            style = METHOD_STYLE[method]
            ax.plot(group["t"], group["mean_rank"], label=method, color=style["color"], linewidth=2)
        ax.set_title(f"{scenario}: adaptive ranks at p={p_value:g}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean active rank")
        ax.grid(alpha=0.25)
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save_figure(fig, figures_dir, "text_product_rank_diagnostics")


def plot_fixed_rank_validation(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "text_product_rank_selection.csv")
    if df.empty:
        return
    p_value = representative_p(df["p"].unique())
    df = df[df["p"] == p_value]
    grouped = (
        df.groupby(["scenario", "method_family", "rank"], as_index=False)["final_regret"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    scenarios = list(dict.fromkeys(grouped["scenario"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.0 * len(scenarios), 3.8), squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        sub = grouped[grouped["scenario"] == scenario]
        for family in sorted(sub["method_family"].unique()):
            group = sub[sub["method_family"] == family].sort_values("rank")
            ax.errorbar(group["rank"], group["mean"], yerr=group["sem"], marker="o", label=family, capsize=3)
        ax.set_title(f"{scenario}: fixed-rank validation at p={p_value:g}")
        ax.set_xlabel("Candidate rank")
        ax.set_ylabel("Validation final regret")
        ax.grid(alpha=0.25)
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save_figure(fig, figures_dir, "text_product_fixed_rank_validation")


def plot_calibration_diagnostics(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "text_product_calibration.csv")
    if df.empty:
        return
    p_value = representative_p(df["p"].unique())
    df = df[df["p"] == p_value]
    adaptive = df[df["method_family"].str.contains("adaptive-rank", na=False)].copy()
    oful = df[df["method_family"] == "Zero-imputed OFUL"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    if not adaptive.empty:
        adaptive["rank_threshold_constant"] = pd.to_numeric(adaptive["rank_threshold_constant"])
        grouped = (
            adaptive.groupby(["method_family", "rank_threshold_constant"], as_index=False)["final_regret"]
            .agg(["mean", "sem"])
            .reset_index()
        )
        for family in sorted(grouped["method_family"].unique()):
            group = grouped[grouped["method_family"] == family].sort_values("rank_threshold_constant")
            axes[0].errorbar(
                group["rank_threshold_constant"],
                group["mean"],
                yerr=group["sem"],
                marker="o",
                label=family,
                capsize=3,
            )
        axes[0].set_xscale("log")
    axes[0].set_title(f"Adaptive threshold calibration at p={p_value:g}")
    axes[0].set_xlabel("Threshold constant")
    axes[0].set_ylabel("Validation final regret")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    if not oful.empty:
        labels = [
            f"λ={float(row.lambda_reg):g}, β={float(row.beta_scale):g}"
            for row in oful.itertuples()
        ]
        grouped = oful.assign(label=labels).groupby("label", as_index=False)["final_regret"].mean()
        axes[1].bar(grouped["label"], grouped["final_regret"], color="#2166AC")
        axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title(f"OFUL calibration at p={p_value:g}")
    axes[1].set_ylabel("Validation final regret")
    axes[1].grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir, "text_product_calibration_diagnostics")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_missingness_sweep(args.results_dir, args.figures_dir)
    plot_regret_over_time(args.results_dir, args.figures_dir)
    plot_rank_diagnostics(args.results_dir, args.figures_dir)
    plot_fixed_rank_validation(args.results_dir, args.figures_dir)
    plot_calibration_diagnostics(args.results_dir, args.figures_dir)
    print(f"Wrote text product-context figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
