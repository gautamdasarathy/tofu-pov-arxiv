"""Create paper figures from generated experiment CSVs."""

from __future__ import annotations

import argparse
import tempfile
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_plot_cache_dir = tempfile.mkdtemp(prefix="tofu-plot-cache-")
os.environ.setdefault("MPLCONFIGDIR", _plot_cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", _plot_cache_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

MAIN_METHOD_ORDER = [
    "Adaptive TOFU",
    "Known-rank TOFU",
    "Zero-imputed OFUL",
    "Masked PSLB known-rank",
    "Masked PSLB adaptive-rank",
]

APPENDIX_TRAJECTORY_METHOD_ORDER = [
    *MAIN_METHOD_ORDER,
    "Random",
]

METHOD_STYLES = {
    "Adaptive TOFU": {"color": "#0B6E69", "marker": "o", "linestyle": "-"},
    "Known-rank TOFU": {"color": "#1F77B4", "marker": "s", "linestyle": "-"},
    "Zero-imputed OFUL": {"color": "#D95F02", "marker": "^", "linestyle": "-"},
    "Masked PSLB known-rank": {"color": "#E34A33", "marker": "v", "linestyle": "-"},
    "Masked PSLB adaptive-rank": {"color": "#B2182B", "marker": "D", "linestyle": "-"},
    "Random": {"color": "#7A7A7A", "marker": "x", "linestyle": ":"},
    "Oracle-subspace OFUL": {"color": "#4D4D4D", "marker": "o", "linestyle": "--"},
    "PSLB full-action": {"color": "#9467BD", "marker": "s", "linestyle": "--"},
    "Masked PSLB": {"color": "#B2182B", "marker": "D", "linestyle": "-"},
    "Masked PSLB min-UCB": {"color": "#F4A582", "marker": "v", "linestyle": "--"},
}

MAIN_RANK_METHOD_LABELS = {
    ("TOFU fixed-rank", "true"): "Known-rank TOFU",
    ("TOFU adaptive-rank", "adaptive"): "Adaptive TOFU",
    ("Zero-imputed OFUL", "ambient"): "Zero-imputed OFUL",
    ("Masked PSLB fixed-rank", "true"): "Masked PSLB known-rank",
    ("Masked PSLB adaptive-rank", "adaptive"): "Masked PSLB adaptive-rank",
}

APPENDIX_RANK_METHOD_LABELS = {
    ("TOFU fixed-rank", "under"): "TOFU fixed under",
    ("TOFU fixed-rank", "true"): "TOFU fixed true",
    ("TOFU fixed-rank", "over"): "TOFU fixed over",
    ("TOFU adaptive-rank", "adaptive"): "TOFU adaptive",
    ("Masked PSLB fixed-rank", "under"): "Masked PSLB fixed under",
    ("Masked PSLB fixed-rank", "true"): "Masked PSLB fixed true",
    ("Masked PSLB fixed-rank", "over"): "Masked PSLB fixed over",
    ("Masked PSLB adaptive-rank", "adaptive"): "Masked PSLB adaptive",
    ("Zero-imputed OFUL", "ambient"): "Zero-imputed OFUL",
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


def aggregate_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["p", "method", "t"], as_index=False)["cumulative_regret"]
    stats = grouped.agg(mean="mean", std="std", n="count")
    stats["stderr"] = stats["std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1))
    return stats


def plot_main_regret_over_time(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "main_plot_trajectories.csv")
    stats = aggregate_trajectory(df)
    p_values = [0.8, 0.4, 0.2]
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, len(p_values), figsize=(13.5, 3.6), sharey=True)

    for ax, p in zip(axes, p_values):
        panel = stats[np.isclose(stats["p"], p)]
        for method in MAIN_METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("t")
            if series.empty:
                continue
            style = METHOD_STYLES[method]
            x = series["t"].to_numpy()
            y = series["mean"].to_numpy()
            err = series["stderr"].to_numpy()
            ax.plot(x, y, label=method, color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
            ax.fill_between(x, y - err, y + err, color=style["color"], alpha=0.15, linewidth=0)
        ax.set_title(f"p = {p:g}")
        ax.set_xlabel("Round")
        ax.set_xlim(left=1)
    axes[0].set_ylabel("Cumulative regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Regret over time under increasing missingness", y=1.20, fontsize=12)
    save_figure(fig, figures_dir, "main_regret_over_time")


def plot_main_missingness_sweep(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "main_plot_summary.csv")
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method in MAIN_METHOD_ORDER:
        series = df[df["method"] == method].sort_values("p")
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
    ax.set_xlabel("Observation probability p")
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Robustness to missing features")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, figures_dir, "main_missingness_sweep")


def rank_plot_frame(results_dir: Path, label_map: dict[tuple[str, str], str]) -> pd.DataFrame:
    df = load_csv(results_dir, "rank_adaptation_summary.csv")
    rows = []
    for _, row in df.iterrows():
        label = label_map.get((row["method"], row["rank_label"]))
        if label is None:
            continue
        item = row.to_dict()
        item["plot_method"] = label
        rows.append(item)
    return pd.DataFrame(rows)


def plot_main_adaptive_rank(results_dir: Path, figures_dir: Path) -> None:
    frame = rank_plot_frame(results_dir, MAIN_RANK_METHOD_LABELS)
    scenarios = ["dense_d12_m3_p08", "sparse_d30_m3_p03", "sparse_d50_m5_p02"]
    methods = [
        "Known-rank TOFU",
        "Adaptive TOFU",
        "Zero-imputed OFUL",
        "Masked PSLB known-rank",
        "Masked PSLB adaptive-rank",
    ]
    colors = {
        method: METHOD_STYLES[method]["color"] for method in methods
    }

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(scenarios))
    width = 0.14
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for offset, method in zip(offsets, methods):
        values = []
        errors = []
        for scenario in scenarios:
            row = frame[(frame["scenario"] == scenario) & (frame["plot_method"] == method)]
            values.append(float(row["mean_final_regret"].iloc[0]))
            errors.append(float(row["stderr_final_regret"].iloc[0]))
        ax.bar(x + offset, values, width, yerr=errors, label=method, color=colors[method], capsize=2)

    adaptive = frame[frame["plot_method"] == "Adaptive TOFU"]
    rank_notes = []
    for scenario in scenarios:
        row = adaptive[adaptive["scenario"] == scenario].iloc[0]
        rank_notes.append(f"{scenario}: adaptive final rank {row['mean_final_rank']:.1f} (true {int(row['true_m'])})")
    ax.text(
        0.01,
        0.98,
        "\n".join(rank_notes),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.92},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["dense\nm*=3,p=.8", "sparse\nm*=3,p=.3", "sparse\nm*=5,p=.2"])
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Adaptive rank without knowing m*")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save_figure(fig, figures_dir, "main_adaptive_rank")


def plot_appendix_random_sanity(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "main_plot_trajectories.csv")
    stats = aggregate_trajectory(df)
    p_values = [0.8, 0.4, 0.2]
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, len(p_values), figsize=(13.5, 3.6), sharey=True)

    for ax, p in zip(axes, p_values):
        panel = stats[np.isclose(stats["p"], p)]
        for method in APPENDIX_TRAJECTORY_METHOD_ORDER:
            series = panel[panel["method"] == method].sort_values("t")
            if series.empty:
                continue
            style = METHOD_STYLES[method]
            x = series["t"].to_numpy()
            y = series["mean"].to_numpy()
            err = series["stderr"].to_numpy()
            ax.plot(x, y, label=method, color=style["color"], linestyle=style["linestyle"], linewidth=2.0)
            ax.fill_between(x, y - err, y + err, color=style["color"], alpha=0.15, linewidth=0)
        ax.set_title(f"p = {p:g}")
        ax.set_xlabel("Round")
        ax.set_xlim(left=1)
    axes[0].set_ylabel("Cumulative regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Regret over time with random sanity check", y=1.20, fontsize=12)
    save_figure(fig, figures_dir, "appendix_random_sanity")


def plot_appendix_rank_misspecification(results_dir: Path, figures_dir: Path) -> None:
    frame = rank_plot_frame(results_dir, APPENDIX_RANK_METHOD_LABELS)
    scenarios = ["dense_d12_m3_p08", "sparse_d30_m3_p03", "sparse_d50_m5_p02"]
    methods = [
        "TOFU fixed under",
        "TOFU fixed true",
        "TOFU fixed over",
        "TOFU adaptive",
        "Masked PSLB fixed under",
        "Masked PSLB fixed true",
        "Masked PSLB fixed over",
        "Masked PSLB adaptive",
        "Zero-imputed OFUL",
    ]
    colors = {
        "TOFU fixed under": "#9ECAE1",
        "TOFU fixed true": "#1F77B4",
        "TOFU fixed over": "#6BAED6",
        "TOFU adaptive": "#0B6E69",
        "Masked PSLB fixed under": "#FCA082",
        "Masked PSLB fixed true": "#E34A33",
        "Masked PSLB fixed over": "#FB6A4A",
        "Masked PSLB adaptive": "#B2182B",
        "Zero-imputed OFUL": "#D95F02",
    }

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    x = np.arange(len(scenarios))
    width = 0.085
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for offset, method in zip(offsets, methods):
        values = []
        errors = []
        for scenario in scenarios:
            row = frame[(frame["scenario"] == scenario) & (frame["plot_method"] == method)]
            values.append(float(row["mean_final_regret"].iloc[0]))
            errors.append(float(row["stderr_final_regret"].iloc[0]))
        ax.bar(x + offset, values, width, yerr=errors, label=method, color=colors[method], capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(["dense\nm*=3,p=.8", "sparse\nm*=3,p=.3", "sparse\nm*=5,p=.2"])
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Rank misspecification diagnostics")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, figures_dir, "appendix_rank_misspecification")


def plot_appendix_warm_start(results_dir: Path, figures_dir: Path) -> None:
    mask = load_csv(results_dir, "synthetic_mask_sweep.csv")
    methods = [
        "TOFU-POV practical",
        "TOFU-POV zero-burnin",
        "TOFU-POV warm-start first-epoch",
        "TOFU-POV warm-start every-epoch",
    ]
    frame = (
        mask[mask["policy"].isin(methods)]
        .groupby(["p", "policy"], as_index=False)["final_regret"]
        .agg(mean="mean", std="std", n="count")
    )
    frame["stderr"] = frame["std"].fillna(0.0) / np.sqrt(frame["n"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    palette = ["#9ECAE1", "#41B6C4", "#0B6E69", "#00441B"]
    for method, color in zip(methods, palette):
        series = frame[frame["policy"] == method].sort_values("p")
        ax.errorbar(series["p"], series["mean"], yerr=series["stderr"], label=method, color=color, marker="o", capsize=3)
    ax.invert_xaxis()
    ax.set_xlabel("Observation probability p")
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Warm-start variants")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, figures_dir, "appendix_warm_start")


def parse_rank_history(text: str) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [int(value) for value in text.split()]


def parse_rank_times(text: str, length: int) -> list[int]:
    if isinstance(text, str) and text.strip():
        return [int(value) for value in text.split()]
    return list(range(1, length + 1))


def expand_rank_history(row: pd.Series) -> pd.DataFrame:
    history = parse_rank_history(row["rank_history"])
    if not history:
        return pd.DataFrame(columns=["t", "rank", "method", "scenario", "true_m"])
    times = parse_rank_times(str(row.get("rank_times", "")), len(history))
    if len(times) != len(history):
        times = list(range(1, len(history) + 1))
    horizon = int(row["T"])
    start = max(1, min(times))
    rank_by_t = []
    idx = 0
    current = history[0]
    for t in range(start, horizon + 1):
        while idx + 1 < len(times) and times[idx + 1] <= t:
            idx += 1
            current = history[idx]
        rank_by_t.append(
            {
                "t": t,
                "rank": current,
                "method": row["method"],
                "scenario": row["scenario"],
                "true_m": int(row["true_m"]),
            }
        )
    return pd.DataFrame(rank_by_t)


def plot_appendix_rank_recovery(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "rank_adaptation.csv")
    df = df[df["method"].isin(["TOFU adaptive-rank", "Masked PSLB adaptive-rank"])]
    expanded = pd.concat([expand_rank_history(row) for _, row in df.iterrows()], ignore_index=True)
    scenarios = list(dict.fromkeys(df["scenario"].tolist()))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(13.0, 3.7), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        panel = expanded[expanded["scenario"] == scenario]
        true_m = int(panel["true_m"].iloc[0])
        for method, color in [("TOFU adaptive-rank", "#0B6E69"), ("Masked PSLB adaptive-rank", "#B2182B")]:
            method_panel = panel[panel["method"] == method]
            stats = (
                method_panel.groupby("t", as_index=False)["rank"]
                .agg(mean="mean", std="std", n="count")
                .sort_values("t")
            )
            stats["stderr"] = stats["std"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1))
            ax.plot(stats["t"], stats["mean"], color=color, label=method, linewidth=2.0)
            ax.fill_between(
                stats["t"].to_numpy(),
                (stats["mean"] - stats["stderr"]).to_numpy(),
                (stats["mean"] + stats["stderr"]).to_numpy(),
                color=color,
                alpha=0.15,
                linewidth=0,
            )
        ax.axhline(true_m, color="#4D4D4D", linestyle="--", linewidth=1.5, label="true rank")
        ax.set_title(scenario)
        ax.set_xlabel("Round")
        ax.set_ylabel("Selected rank")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    save_figure(fig, figures_dir, "appendix_rank_recovery")


def plot_appendix_calibration(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "synthetic_bias_sweep.csv")
    frame = df.groupby("c_b", as_index=False)["final_regret"].agg(mean="mean", std="std", n="count")
    frame["stderr"] = frame["std"].fillna(0.0) / np.sqrt(frame["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.errorbar(frame["c_b"], frame["mean"], yerr=frame["stderr"], marker="o", color="#0B6E69", capsize=3)
    ax.set_xlabel("Subspace-bias multiplier c_b")
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Finite-horizon calibration sensitivity")
    save_figure(fig, figures_dir, "appendix_calibration")


def plot_appendix_feature_noise(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "synthetic_feature_noise_sweep.csv")
    methods = [
        "TOFU-POV warm-start first-epoch",
        "TOFU-POV warm-start every-epoch",
        "Zero-imputed OFUL",
        "Masked PSLB",
        "Oracle-subspace OFUL",
    ]
    styles = {
        "TOFU-POV warm-start first-epoch": {"color": "#0B6E69", "marker": "o", "linestyle": "-"},
        "TOFU-POV warm-start every-epoch": {"color": "#00441B", "marker": "s", "linestyle": "-"},
        "Zero-imputed OFUL": METHOD_STYLES["Zero-imputed OFUL"],
        "Masked PSLB": METHOD_STYLES["Masked PSLB"],
        "Oracle-subspace OFUL": METHOD_STYLES["Oracle-subspace OFUL"],
    }
    frame = (
        df[df["policy"].isin(methods)]
        .groupby(["scenario", "perturbation_std", "policy"], as_index=False)["final_regret"]
        .agg(mean="mean", std="std", n="count")
    )
    frame["stderr"] = frame["std"].fillna(0.0) / np.sqrt(frame["n"].clip(lower=1))
    scenarios = list(dict.fromkeys(frame["scenario"].tolist()))

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, len(scenarios), figsize=(11.2, 4.0), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        panel = frame[frame["scenario"] == scenario]
        for method in methods:
            series = panel[panel["policy"] == method].sort_values("perturbation_std")
            if series.empty:
                continue
            style = styles[method]
            ax.errorbar(
                series["perturbation_std"],
                series["mean"],
                yerr=series["stderr"],
                label=method,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                capsize=3,
            )
        ax.set_xlabel("Feature perturbation std")
        ax.set_title(scenario)
    axes[0].set_ylabel("Final cumulative regret")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Sensitivity to off-subspace feature perturbations", y=1.18, fontsize=12)
    save_figure(fig, figures_dir, "appendix_feature_noise")


def plot_appendix_full_information_references(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "synthetic_mask_sweep.csv")
    methods = [
        "TOFU-POV warm-start first-epoch",
        "TOFU-POV warm-start every-epoch",
        "Zero-imputed OFUL",
        "Masked PSLB",
        "Oracle-subspace OFUL",
        "PSLB full-action",
    ]
    frame = (
        df[df["policy"].isin(methods)]
        .groupby(["p", "policy"], as_index=False)["final_regret"]
        .agg(mean="mean", std="std", n="count")
    )
    frame["stderr"] = frame["std"].fillna(0.0) / np.sqrt(frame["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for method in methods:
        series = frame[frame["policy"] == method].sort_values("p")
        style = METHOD_STYLES.get(method, {"color": "#333333", "marker": "o", "linestyle": "-"})
        if method == "TOFU-POV warm-start first-epoch":
            style = {"color": "#0B6E69", "marker": "o", "linestyle": "-"}
        if method == "TOFU-POV warm-start every-epoch":
            style = {"color": "#00441B", "marker": "s", "linestyle": "-"}
        ax.errorbar(
            series["p"],
            series["mean"],
            yerr=series["stderr"],
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
        )
    ax.invert_xaxis()
    ax.set_xlabel("Observation probability p")
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("Full-information references")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, figures_dir, "appendix_full_information_references")


def plot_appendix_pslb_intersection_diagnostic(results_dir: Path, figures_dir: Path) -> None:
    df = load_csv(results_dir, "synthetic_mask_sweep.csv")
    methods = [
        "Masked PSLB",
        "Masked PSLB min-UCB",
        "Zero-imputed OFUL",
    ]
    frame = (
        df[df["policy"].isin(methods)]
        .groupby(["p", "policy"], as_index=False)["final_regret"]
        .agg(mean="mean", std="std", n="count")
    )
    frame["stderr"] = frame["std"].fillna(0.0) / np.sqrt(frame["n"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in methods:
        series = frame[frame["policy"] == method].sort_values("p")
        if series.empty:
            continue
        style = METHOD_STYLES[method]
        ax.errorbar(
            series["p"],
            series["mean"],
            yerr=series["stderr"],
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            capsize=3,
        )
    ax.invert_xaxis()
    ax.set_xlabel("Observation probability p")
    ax.set_ylabel("Final cumulative regret")
    ax.set_title("PSLB projected-sampled vs min-UCB diagnostic")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, figures_dir, "appendix_pslb_intersection_diagnostic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Accepted for smoke-test compatibility.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="paper")
    plot_main_regret_over_time(args.results_dir, args.figures_dir)
    plot_main_missingness_sweep(args.results_dir, args.figures_dir)
    plot_main_adaptive_rank(args.results_dir, args.figures_dir)
    plot_appendix_random_sanity(args.results_dir, args.figures_dir)
    plot_appendix_rank_misspecification(args.results_dir, args.figures_dir)
    plot_appendix_warm_start(args.results_dir, args.figures_dir)
    plot_appendix_rank_recovery(args.results_dir, args.figures_dir)
    plot_appendix_calibration(args.results_dir, args.figures_dir)
    plot_appendix_feature_noise(args.results_dir, args.figures_dir)
    plot_appendix_full_information_references(args.results_dir, args.figures_dir)
    plot_appendix_pslb_intersection_diagnostic(args.results_dir, args.figures_dir)
    print(f"Wrote paper figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
