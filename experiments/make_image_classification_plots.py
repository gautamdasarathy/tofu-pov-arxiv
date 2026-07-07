"""Create figures for image-classification-to-bandit experiments."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_plot_cache_dir = tempfile.mkdtemp(prefix="tofu-image-plot-cache-")
os.environ.setdefault("MPLCONFIGDIR", _plot_cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", _plot_cache_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

FAIR_METHOD_ORDER = [
    "Adaptive TOFU",
    "TOFU fixed-rank best-val",
    "Zero-imputed OFUL",
    "Masked PSLB fixed-rank best-val",
    "Masked PSLB adaptive-rank",
]

APPENDIX_METHOD_ORDER = [
    *FAIR_METHOD_ORDER,
    "Random",
    "Full-info OFUL",
    "Full-info PSLB",
]

ADAPTIVE_METHODS = [
    "Adaptive TOFU",
    "Masked PSLB adaptive-rank",
]

METHOD_STYLES = {
    "Adaptive TOFU": {"color": "#0B6E69", "marker": "o", "linestyle": "-"},
    "TOFU fixed-rank best-val": {"color": "#1F77B4", "marker": "s", "linestyle": "-"},
    "Zero-imputed OFUL": {"color": "#D95F02", "marker": "^", "linestyle": "-"},
    "Masked PSLB fixed-rank best-val": {"color": "#E34A33", "marker": "v", "linestyle": "-"},
    "Masked PSLB adaptive-rank": {"color": "#B2182B", "marker": "D", "linestyle": "-"},
    "Random": {"color": "#7A7A7A", "marker": "x", "linestyle": ":"},
    "Full-info OFUL": {"color": "#4D4D4D", "marker": "o", "linestyle": "--"},
    "Full-info PSLB": {"color": "#9467BD", "marker": "s", "linestyle": "--"},
}


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_csv(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return pd.read_csv(path)


def subplot_grid(n: int, *, width: float = 5.2, height: float = 3.8) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, n, figsize=(max(width * n, width), height), squeeze=False)
    return fig, axes[0]


def representative_p(values: pd.Series) -> float:
    unique = sorted(float(value) for value in values.dropna().unique())
    if not unique:
        raise ValueError("No p values available.")
    return 0.4 if any(np.isclose(value, 0.4) for value in unique) else unique[len(unique) // 2]


def plot_missingness_sweep(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "image_classification_summary.csv")
    scenarios = sorted(df["scenario"].unique())
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = subplot_grid(len(scenarios), width=5.0, height=3.8)

    for ax, scenario in zip(axes, scenarios):
        panel = df[df["scenario"] == scenario]
        for method in FAIR_METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("p")
            if series.empty:
                continue
            style = METHOD_STYLES[method]
            ax.errorbar(
                series["p"],
                series["mean_final_regret"],
                yerr=series["stderr_final_regret"],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                capsize=3,
            )
        ax.invert_xaxis()
        ax.set_title(scenario)
        ax.set_xlabel("Observation probability p")
        ax.set_ylabel("Final cumulative regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Image classification robustness to missing features", y=1.22, fontsize=12)
    save_figure(fig, figures_dir, "image_classification_missingness_sweep")


def aggregate_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["scenario", "p", "method", "t"], as_index=False)["cumulative_regret"]
    stats = grouped.agg(mean="mean", std="std", n="count")
    stats["stderr"] = stats["std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1))
    return stats


def plot_regret_over_time(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "image_classification_trajectories.csv")
    p_value = representative_p(df["p"])
    stats = aggregate_trajectory(df[np.isclose(df["p"], p_value)])
    scenarios = sorted(stats["scenario"].unique())

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = subplot_grid(len(scenarios), width=5.2, height=3.8)
    for ax, scenario in zip(axes, scenarios):
        panel = stats[stats["scenario"] == scenario]
        for method in APPENDIX_METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("t")
            if series.empty:
                continue
            style = METHOD_STYLES[method]
            x = series["t"].to_numpy()
            y = series["mean"].to_numpy()
            err = series["stderr"].to_numpy()
            ax.plot(x, y, label=method, color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
            ax.fill_between(x, y - err, y + err, color=style["color"], alpha=0.15, linewidth=0)
        ax.set_title(scenario)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle(f"Image classification regret over time at p = {p_value:g}", y=1.22, fontsize=12)
    save_figure(fig, figures_dir, "image_classification_regret_over_time")


def plot_rank_diagnostics(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "image_classification_trajectories.csv")
    p_value = representative_p(df["p"])
    frame = df[np.isclose(df["p"], p_value)].copy()
    frame["active_rank_numeric"] = pd.to_numeric(frame["active_rank"], errors="coerce")
    frame = frame[frame["method"].isin(ADAPTIVE_METHODS) & frame["active_rank_numeric"].notna()]
    scenarios = sorted(frame["scenario"].unique())
    if not scenarios:
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.text(0.5, 0.5, "No adaptive-rank diagnostics available", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, figures_dir, "image_classification_rank_diagnostics")
        return

    stats = (
        frame.groupby(["scenario", "method", "t"], as_index=False)["active_rank_numeric"]
        .mean()
        .rename(columns={"active_rank_numeric": "mean_rank"})
    )
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = subplot_grid(len(scenarios), width=4.8, height=3.6)
    for ax, scenario in zip(axes, scenarios):
        panel = stats[stats["scenario"] == scenario]
        for method in ADAPTIVE_METHODS:
            series = panel[panel["method"] == method].sort_values("t")
            if series.empty:
                continue
            style = METHOD_STYLES[method]
            ax.plot(
                series["t"],
                series["mean_rank"],
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
        ax.set_title(scenario)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean active rank")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(f"Image adaptive-rank diagnostics at p = {p_value:g}", y=1.20, fontsize=12)
    save_figure(fig, figures_dir, "image_classification_rank_diagnostics")


def plot_full_info_refs(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "image_classification_summary.csv")
    scenarios = sorted(df["scenario"].unique())
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = subplot_grid(len(scenarios), width=5.0, height=3.8)

    for ax, scenario in zip(axes, scenarios):
        panel = df[df["scenario"] == scenario]
        for method in APPENDIX_METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("p")
            if series.empty:
                continue
            style = METHOD_STYLES[method]
            ax.errorbar(
                series["p"],
                series["mean_final_regret"],
                yerr=series["stderr_final_regret"],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                capsize=3,
            )
        ax.invert_xaxis()
        ax.set_title(scenario)
        ax.set_xlabel("Observation probability p")
        ax.set_ylabel("Final cumulative regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Image classification full-information references", y=1.22, fontsize=12)
    save_figure(fig, figures_dir, "image_classification_full_info_refs")


def plot_fixed_rank_validation(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "image_classification_rank_selection.csv")
    p_value = representative_p(df["p"])
    frame = df[np.isclose(df["p"], p_value)]
    stats = (
        frame.groupby(["scenario", "method_family", "rank"], as_index=False)["final_regret"]
        .agg(mean="mean", std="std", n="count")
    )
    stats["stderr"] = stats["std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1))
    scenarios = sorted(stats["scenario"].unique())

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = subplot_grid(len(scenarios), width=4.8, height=3.6)
    colors = {"TOFU fixed-rank": "#1F77B4", "Masked PSLB fixed-rank": "#E34A33"}
    for ax, scenario in zip(axes, scenarios):
        panel = stats[stats["scenario"] == scenario]
        for family, color in colors.items():
            series = panel[panel["method_family"] == family].sort_values("rank")
            if series.empty:
                continue
            ax.errorbar(
                series["rank"],
                series["mean"],
                yerr=series["stderr"],
                label=family,
                color=color,
                marker="o",
                linewidth=2.0,
                capsize=3,
            )
        ax.set_title(scenario)
        ax.set_xlabel("Fixed rank")
        ax.set_ylabel("Validation final regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(f"Image fixed-rank validation at p = {p_value:g}", y=1.20, fontsize=12)
    save_figure(fig, figures_dir, "image_classification_fixed_rank_validation")


def plot_calibration_diagnostics(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "image_classification_calibration.csv")
    p_value = representative_p(df["p"])
    scenarios = sorted(df["scenario"].unique())
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = subplot_grid(len(scenarios), width=6.0, height=4.2)

    for ax, scenario in zip(axes, scenarios):
        frame = df[(df["scenario"] == scenario) & np.isclose(df["p"], p_value)].copy()
        frame = frame[
            frame["method_family"].isin(
                ["TOFU adaptive-rank", "Masked PSLB adaptive-rank", "Zero-imputed OFUL"]
            )
        ]
        stats = (
            frame.groupby(["method_family", "candidate_label"], as_index=False)
            .agg(mean=("final_regret", "mean"), selected=("selected", "max"))
            .sort_values(["method_family", "mean"])
        )
        labels = [f"{row.method_family}\n{row.candidate_label}" for row in stats.itertuples()]
        colors = ["#0B6E69" if int(row.selected) else "#BDBDBD" for row in stats.itertuples()]
        ax.bar(np.arange(len(stats)), stats["mean"], color=colors)
        ax.set_xticks(np.arange(len(stats)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("Validation final regret")
        ax.set_title(scenario)
    fig.suptitle(f"Image calibration diagnostics at p = {p_value:g}", y=1.08, fontsize=12)
    save_figure(fig, figures_dir, "image_classification_calibration_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Accepted for symmetry with runners.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    del args.quick
    plot_missingness_sweep(args.results_dir, args.figures_dir)
    plot_regret_over_time(args.results_dir, args.figures_dir)
    plot_rank_diagnostics(args.results_dir, args.figures_dir)
    plot_full_info_refs(args.results_dir, args.figures_dir)
    plot_fixed_rank_validation(args.results_dir, args.figures_dir)
    plot_calibration_diagnostics(args.results_dir, args.figures_dir)
    print(f"Wrote image-classification figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
