from __future__ import annotations

import ast
import argparse
import csv
import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path
import json
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import sys

sys.path.append(str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from src.models.base import BaseModel


def create_model(model_path: str, categories: Optional[List[str]] = None) -> "BaseModel":
    from src.models.yolo_model import YOLOModel

    return YOLOModel(
        model_path=model_path if Path(model_path).exists() else None,
        model_name=model_path
    )


def _resolve_results_file(results_path: str | Path) -> Path:
    """Resolve an experiment directory or an explicit CSV/JSON results file."""
    path = Path(results_path)
    if path.is_dir():
        for filename in ("results.csv", "results.json"):
            candidate = path / filename
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No results.csv or results.json found in {path}")
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    if path.suffix.lower() not in {".csv", ".json"}:
        raise ValueError(f"Unsupported results file format: {path.suffix}")
    return path


def load_experiment_results(results_path: str | Path) -> Dict[str, List]:
    """Load current runner CSV output, while retaining JSON compatibility."""
    results_file = _resolve_results_file(results_path)
    if results_file.suffix.lower() == ".json":
        with results_file.open("r") as handle:
            results = json.load(handle)
    else:
        rounds: List[int] = []
        metrics_history: List[Dict] = []
        with results_file.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = {"rounds", "metrics_history"}
            if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"{results_file} must contain columns: {sorted(expected_columns)}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    round_number = int(row["rounds"])
                    metrics = ast.literal_eval(row["metrics_history"])
                except (SyntaxError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid result row {line_number} in {results_file}"
                    ) from exc
                if not isinstance(metrics, dict):
                    raise ValueError(
                        f"metrics_history in row {line_number} of {results_file} is not a dictionary"
                    )
                rounds.append(round_number)
                metrics_history.append(metrics)
        results = {"rounds": rounds, "metrics_history": metrics_history}

    if not isinstance(results, dict) or "rounds" not in results or "metrics_history" not in results:
        raise ValueError(f"Invalid results schema in {results_file}")
    if len(results["rounds"]) != len(results["metrics_history"]):
        raise ValueError(f"Mismatched rounds and metrics_history lengths in {results_file}")
    return results


def _metric_value(metrics: Dict, metric_name: str = "map50-95") -> float:
    value = metrics.get(metric_name, 0.0)
    if value in (None, "N/A"):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def aggregate_seed_results(
    results_files: List[str], metric_name: str = "map50"
) -> Tuple[List[int], List[float], List[float]]:
    """Return per-round mean and sample standard deviation across seed runs.

    Every seed must contain the same round numbers in the same order.  This is
    intentional: silently aligning incomplete active-learning runs would make
    a mean curve look more reliable than the underlying experiments are.
    """
    if len(results_files) < 2:
        raise ValueError("Mean/std plotting requires results from at least two seeds")

    reference_rounds: Optional[List[int]] = None
    values_by_seed: List[List[float]] = []
    for results_file in results_files:
        results = load_experiment_results(results_file)
        rounds = results["rounds"]
        if reference_rounds is None:
            reference_rounds = rounds
        elif rounds != reference_rounds:
            raise ValueError(
                "All seed result files must contain identical round numbers: "
                f"{results_file} differs from the first file"
            )

        values: List[float] = []
        for round_number, metrics in zip(rounds, results["metrics_history"]):
            if metric_name not in metrics:
                raise ValueError(
                    f"Metric '{metric_name}' is missing at round {round_number} in {results_file}"
                )
            values.append(_metric_value(metrics, metric_name))
        values_by_seed.append(values)

    assert reference_rounds is not None
    means = [sum(values) / len(values) for values in zip(*values_by_seed)]
    sample_stds = [
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        for values, mean in zip(zip(*values_by_seed), means)
    ]
    return reference_rounds, means, sample_stds


def plot_seed_mean_std(
    results_files: List[str],
    strategy_name: str,
    save_path: str = "",
    metric_name: str = "map50",
    paper_style: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    round_offset: int = 0,
    caption: str = "",
):
    """Plot a strategy's seed mean with a translucent +/- one-std band."""
    source_rounds, mean_values, std_values = aggregate_seed_results(results_files, metric_name)
    rounds = [round_number + round_offset for round_number in source_rounds]
    lower = [mean - std for mean, std in zip(mean_values, std_values)]
    upper = [mean + std for mean, std in zip(mean_values, std_values)]

    if paper_style:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
        })
        fig, ax = plt.subplots(figsize=(12, 7.3))
        style_by_name = {
            "Random": ("#7f7f7f", "D", "white", 1.6),
            "Entropy": ("#5b8c55", "^", "#5b8c55", 1.6),
            "BADGE": ("#ff7f0e", "d", "white", 1.6),
            "CoreSet": ("#4aa3c0", "o", "#4aa3c0", 1.6),
            "CDAL": ("#9b59b6", "o", "white", 1.9),
            "ReCAL": ("#ef4444", "s", "white", 1.9),
        }
        color, marker, facecolor, edgewidth = style_by_name.get(
            strategy_name, ("#4c72b0", "o", "#4c72b0", 1.6)
        )
        ax.fill_between(rounds, lower, upper, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(
            rounds, mean_values, label=strategy_name, color=color, linewidth=2.8,
            marker=marker, markersize=7.5, markerfacecolor=facecolor,
            markeredgecolor=color, markeredgewidth=edgewidth, zorder=2,
        )
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=1.1)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.6)
        ax.tick_params(width=1.6, length=8)
        ax.legend(loc="lower right", frameon=True, framealpha=0.94,
                  edgecolor="#c8c8c8", fancybox=True, handlelength=2.25)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.fill_between(rounds, lower, upper, alpha=0.18)
        ax.plot(rounds, mean_values, marker="o", linewidth=2, label=strategy_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    metric_label = "mAP@50" if metric_name == "map50" else "mAP@0.5-0.95"
    ax.set_xlabel("Rounds" if paper_style else "Active Learning Round", labelpad=12 if paper_style else None)
    ax.set_ylabel(metric_label, labelpad=12 if paper_style else None)
    ax.set_xlim(rounds[0] - 0.35, rounds[-1] + 0.35)
    ax.set_xticks(rounds)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=18)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Mean/std plot saved to: {save_path}")
    else:
        plt.show()
    return source_rounds, mean_values, std_values


def compare_with_seed_mean_std(
    seed_results_files: List[str],
    seed_strategy_name: str,
    baseline_results_files: List[str],
    baseline_strategy_names: List[str],
    save_path: str = "",
    metric_name: str = "map50",
    paper_style: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    round_offset: int = 0,
    caption: str = "",
):
    """Compare single-run baselines with one strategy reported as mean +/- std."""
    if len(baseline_results_files) != len(baseline_strategy_names):
        raise ValueError("The number of baseline files must match the number of baseline names")

    source_rounds, seed_means, seed_stds = aggregate_seed_results(seed_results_files, metric_name)
    rounds = [round_number + round_offset for round_number in source_rounds]
    seed_lower = [mean - std for mean, std in zip(seed_means, seed_stds)]
    seed_upper = [mean + std for mean, std in zip(seed_means, seed_stds)]

    if paper_style:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
        })
        fig, ax = plt.subplots(figsize=(12, 7.3))
        style_by_name = {
            "Random": ("#7f7f7f", "D", "white", 1.6),
            "Entropy": ("#5b8c55", "^", "#5b8c55", 1.6),
            "CoreSet": ("#4aa3c0", "o", "#4aa3c0", 1.6),
            "Core-set": ("#4aa3c0", "o", "#4aa3c0", 1.6),
            "CDAL": ("#9b59b6", "o", "white", 1.9),
            "ReCAL": ("#ef4444", "s", "white", 1.9),
            "ReCAL1": ("#c77728", "h", "white", 1.8),
        }
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

    all_values = seed_lower + seed_upper
    for results_file, strategy_name in zip(baseline_results_files, baseline_strategy_names):
        results = load_experiment_results(results_file)
        if results["rounds"] != source_rounds:
            raise ValueError(
                "All baselines must contain the same round numbers as the seed runs: "
                f"{results_file} differs from the Random files"
            )
        values: List[float] = []
        for round_number, metrics in zip(results["rounds"], results["metrics_history"]):
            if metric_name not in metrics:
                raise ValueError(
                    f"Metric '{metric_name}' is missing at round {round_number} in {results_file}"
                )
            values.append(_metric_value(metrics, metric_name))
        all_values.extend(values)

        if paper_style:
            color, marker, facecolor, edgewidth = style_by_name.get(
                strategy_name, ("#4c72b0", "o", "#4c72b0", 1.6)
            )
            ax.plot(
                rounds, values, label=strategy_name, color=color, linewidth=2.8,
                marker=marker, markersize=7.5, markerfacecolor=facecolor,
                markeredgecolor=color, markeredgewidth=edgewidth, zorder=2,
            )
        else:
            ax.plot(rounds, values, marker="o", linewidth=2, label=strategy_name)

    if paper_style:
        color, marker, facecolor, edgewidth = style_by_name.get(
            seed_strategy_name, ("#4c72b0", "o", "#4c72b0", 1.6)
        )
        ax.fill_between(rounds, seed_lower, seed_upper, color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(
            rounds, seed_means, label=seed_strategy_name, color=color, linewidth=2.8,
            marker=marker, markersize=7.5, markerfacecolor=facecolor,
            markeredgecolor=color, markeredgewidth=edgewidth, zorder=3,
        )
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=1.1)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.6)
        ax.tick_params(width=1.6, length=8)
        ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.94,
                  edgecolor="#c8c8c8", fancybox=True, handlelength=2.25,
                  columnspacing=1.1, handletextpad=0.4)
    else:
        ax.fill_between(rounds, seed_lower, seed_upper, alpha=0.18)
        ax.plot(rounds, seed_means, marker="o", linewidth=2, label=seed_strategy_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    metric_label = "mAP@50" if metric_name == "map50" else "mAP@0.5-0.95"
    ax.set_xlabel("Rounds" if paper_style else "Active Learning Round", labelpad=12 if paper_style else None)
    ax.set_ylabel(metric_label, labelpad=12 if paper_style else None)
    ax.set_xlim(rounds[0] - 0.35, rounds[-1] + 0.35)
    ax.set_xticks(rounds)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif paper_style:
        ax.set_ylim(min(0.45, min(all_values) - 0.02), max(0.81, max(all_values) + 0.02))
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=18)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison mean/std plot saved to: {save_path}")
    else:
        plt.show()


def compare_multiple_seed_mean_std(
    seed_groups: Dict[str, List[str]],
    baseline_results_files: List[str],
    baseline_strategy_names: List[str],
    save_path: str = "",
    metric_name: str = "map50",
    paper_style: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    round_offset: int = 0,
    max_round: Optional[int] = None,
    y_tick_step: Optional[float] = None,
    y_ticks: Optional[List[float]] = None,
    caption: str = "",
):
    """Compare any number of mean +/- std strategies with single-run baselines."""
    if not seed_groups:
        raise ValueError("At least one seed group is required")
    if len(baseline_results_files) != len(baseline_strategy_names):
        raise ValueError("The number of baseline files must match the number of baseline names")

    aggregated_groups: Dict[str, Tuple[List[float], List[float]]] = {}
    full_source_rounds: Optional[List[int]] = None
    for strategy_name, results_files in seed_groups.items():
        group_rounds, means, stds = aggregate_seed_results(results_files, metric_name)
        if full_source_rounds is None:
            full_source_rounds = group_rounds
        elif group_rounds != full_source_rounds:
            raise ValueError(
                "All seed groups must contain identical round numbers: "
                f"{strategy_name} differs from the first seed group"
            )
        aggregated_groups[strategy_name] = (means, stds)

    assert full_source_rounds is not None
    selected_indices = [
        index for index, round_number in enumerate(full_source_rounds)
        if max_round is None or round_number <= max_round
    ]
    if not selected_indices:
        raise ValueError("--max-round excludes every result row")
    source_rounds = [full_source_rounds[index] for index in selected_indices]
    aggregated_groups = {
        strategy_name: (
            [means[index] for index in selected_indices],
            [stds[index] for index in selected_indices],
        )
        for strategy_name, (means, stds) in aggregated_groups.items()
    }
    rounds = [round_number + round_offset for round_number in source_rounds]
    if paper_style:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
        })
        fig, ax = plt.subplots(figsize=(12, 5.6))
        style_by_name = {
            "Random": ("#7f7f7f", "D", "white", 1.6),
            "Entropy": ("#5b8c55", "^", "#5b8c55", 1.6),
            "BADGE": ("#ff7f0e", "d", "white", 1.6),
            "CoreSet": ("#4aa3c0", "o", "#4aa3c0", 1.6),
            "Core-set": ("#4aa3c0", "o", "#4aa3c0", 1.6),
            "CDAL": ("#9b59b6", "o", "white", 1.9),
            "ReCAL": ("#ef4444", "s", "white", 1.9),
            "ReCAL1": ("#c77728", "h", "white", 1.8),
        }
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

    all_values: List[float] = []
    for results_file, strategy_name in zip(baseline_results_files, baseline_strategy_names):
        results = load_experiment_results(results_file)
        if results["rounds"] != full_source_rounds:
            raise ValueError(
                "All baselines must contain the same round numbers as the seed groups: "
                f"{results_file} differs from the seed results"
            )
        values: List[float] = []
        for index in selected_indices:
            round_number = results["rounds"][index]
            metrics = results["metrics_history"][index]
            if metric_name not in metrics:
                raise ValueError(
                    f"Metric '{metric_name}' is missing at round {round_number} in {results_file}"
                )
            values.append(_metric_value(metrics, metric_name))
        all_values.extend(values)

        if paper_style:
            color, marker, facecolor, edgewidth = style_by_name.get(
                strategy_name, ("#4c72b0", "o", "#4c72b0", 1.6)
            )
            ax.plot(
                rounds, values, label=strategy_name, color=color, linewidth=2.8,
                marker=marker, markersize=7.5, markerfacecolor=facecolor,
                markeredgecolor=color, markeredgewidth=edgewidth, zorder=2,
            )
        else:
            ax.plot(rounds, values, marker="o", linewidth=2, label=strategy_name)

    for strategy_name, (means, stds) in aggregated_groups.items():
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        all_values.extend(lower + upper)
        if paper_style:
            color, marker, facecolor, edgewidth = style_by_name.get(
                strategy_name, ("#4c72b0", "o", "#4c72b0", 1.6)
            )
            ax.fill_between(rounds, lower, upper, color=color, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(
                rounds, means, label=strategy_name, color=color, linewidth=2.8,
                marker=marker, markersize=7.5, markerfacecolor=facecolor,
                markeredgecolor=color, markeredgewidth=edgewidth, zorder=3,
            )
        else:
            ax.fill_between(rounds, lower, upper, alpha=0.18)
            ax.plot(rounds, means, marker="o", linewidth=2, label=strategy_name)

    if paper_style:
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=1.1)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.6)
        ax.tick_params(width=1.6, length=8)
        ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.94,
                  edgecolor="#c8c8c8", fancybox=True, handlelength=2.25,
                  columnspacing=1.1, handletextpad=0.4)
    else:
        ax.legend()
        ax.grid(True, alpha=0.3)

    metric_label = "mAP@50" if metric_name == "map50" else "mAP@0.5-0.95"
    ax.set_xlabel(
        "Rounds" if paper_style else "Active Learning Round",
        labelpad=12 if paper_style else None,
        fontweight="bold" if paper_style else "normal",
    )
    ax.set_ylabel(
        metric_label,
        labelpad=12 if paper_style else None,
        fontweight="bold" if paper_style else "normal",
    )
    ax.set_xlim(rounds[0] - 0.35, rounds[-1] + 0.35)
    ax.set_xticks(rounds)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif paper_style:
        ax.set_ylim(min(0.45, min(all_values) - 0.02), max(0.81, max(all_values) + 0.02))
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif y_tick_step is not None:
        if y_tick_step <= 0:
            raise ValueError("y_tick_step must be positive")
        y_min, y_max = ax.get_ylim()
        tick_count = int(math.floor((y_max - y_min) / y_tick_step + 1e-9))
        ax.set_yticks([round(y_min + index * y_tick_step, 10) for index in range(tick_count + 1)])
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=18)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Multi-strategy mean/std plot saved to: {save_path}")
    else:
        plt.show()

def plot_summary_mean_std(
    summary_file: str,
    strategy_names: List[str],
    save_path: str = "",
    paper_style: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[List[float]] = None,
):
    """Plot pre-aggregated ``<strategy>_mean``/``<strategy>_std`` CSV columns."""
    path = Path(summary_file)
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"Summary file has no data rows: {path}")
    if "display_round" not in fieldnames:
        raise ValueError(f"{path} must contain a display_round column")
    for strategy_name in strategy_names:
        required_columns = {f"{strategy_name}_mean", f"{strategy_name}_std"}
        if not required_columns.issubset(fieldnames):
            raise ValueError(f"{path} is missing columns: {sorted(required_columns)}")

    display_names = {
        "recal": "ReCAL", "badge": "BADGE", "random": "Random", "cdal": "CDAL",
        "entropy": "Entropy", "coreset": "Core-set",
    }
    style_by_name = {
        "ReCAL": ("#ef4444", "s", "white", 1.9),
        "BADGE": ("#ff7f0e", "d", "white", 1.6),
        "Random": ("#7f7f7f", "D", "white", 1.6),
        "CDAL": ("#9b59b6", "o", "white", 1.9),
        "Entropy": ("#5b8c55", "^", "#5b8c55", 1.6),
        "Core-set": ("#4aa3c0", "o", "#4aa3c0", 1.6),
    }
    rounds = [int(row["display_round"]) for row in rows]
    if paper_style:
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["DejaVu Serif"],
            "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.fontsize": 9,
        })
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

    lines = {}
    all_values: List[float] = []
    for strategy_name in strategy_names:
        label = display_names.get(strategy_name, strategy_name)
        means = [float(row[f"{strategy_name}_mean"]) for row in rows]
        stds = [float(row[f"{strategy_name}_std"]) for row in rows]
        lower = [mean - std for mean, std in zip(means, stds)]
        upper = [mean + std for mean, std in zip(means, stds)]
        all_values.extend(lower + upper)
        color, marker, facecolor, edgewidth = style_by_name.get(
            label, ("#4c72b0", "o", "#4c72b0", 1.6)
        )
        ax.fill_between(rounds, lower, upper, color=color, alpha=0.11, linewidth=0, zorder=1)
        lines[strategy_name], = ax.plot(
            rounds, means, label=label, color=color, linewidth=1.25, marker=marker,
            markersize=3.6, markerfacecolor=facecolor, markeredgecolor=color,
            markeredgewidth=min(edgewidth, 1.0), zorder=3,
        )

    ax.set_xlabel("Rounds" if paper_style else "Active Learning Round", labelpad=4 if paper_style else None,
                  fontweight="normal")
    ax.set_ylabel("mAP@50", labelpad=4 if paper_style else None, fontweight="normal")
    ax.set_xlim(rounds[0] - 0.55, rounds[-1] + 0.55)
    ax.set_xticks(rounds)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif paper_style:
        ax.set_ylim(min(0.45, min(all_values) - 0.02), max(0.81, max(all_values) + 0.02))
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if paper_style:
        ax.grid(True, color="#b0b0b0", alpha=0.32, linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3)
    else:
        ax.grid(True, alpha=0.3)

    preferred_order = ["recal", "random", "entropy", "badge", "cdal", "coreset"]
    legend_order = [name for name in preferred_order if name in lines]
    legend_order.extend(name for name in strategy_names if name not in legend_order)
    ax.legend(
        [lines[name] for name in legend_order], [lines[name].get_label() for name in legend_order],
        loc="lower right", ncol=2 if paper_style else 1, frameon=True, framealpha=0.94,
        edgecolor="#c8c8c8", fancybox=True, handlelength=1.55, columnspacing=0.65,
        handletextpad=0.35, borderpad=0.35, labelspacing=0.28,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Summary mean/std plot saved to: {save_path}")
    else:
        plt.show()


def plot_learning_curves(results_file: str, save_path: str = ""):
    results = load_experiment_results(results_file)
    
    rounds = results['rounds']
    metrics_history = results['metrics_history']
    
    map_values = []
    for metrics in metrics_history:
        map_values.append(_metric_value(metrics))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, map_values, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Active Learning Round')
    plt.ylabel('mAP@0.5-0.95')
    plt.title('Active Learning Progress')
    plt.grid(True, alpha=0.3)
    if len(map_values) > 1:
        improvement = map_values[-1] - map_values[0]
        plt.text(0.02, 0.98, f'Improvement: {improvement:.4f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")
    else:
        plt.show()


def compare_strategies(
    experiment_dirs: List[str],
    strategy_names: List[str],
    save_path: str = "",
    metric_name: str = "map50-95",
    paper_style: bool = False,
    y_limits: Optional[Tuple[float, float]] = None,
    round_offset: int = 0,
    max_round: Optional[int] = None,
    shadow_width: Optional[float] = None,
):

    if len(experiment_dirs) != len(strategy_names):
        raise ValueError("The number of result files must match the number of strategy names")

    if paper_style:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
        })
        fig, ax = plt.subplots(figsize=(12, 7.3))
        style_by_name = {
            "Random": ("#7f7f7f", "D", "white", 1.6),
            "Entropy": ("#5b8c55", "^", "#5b8c55", 1.6),
            "CoreSet": ("#4aa3c0", "o", "#4aa3c0", 1.6),
            "CDAL": ("#9b59b6", "o", "white", 1.9),
            "ReCAL": ("#ef4444", "s", "white", 1.9),
            "FDAL": ("#8e44ad", "X", "#8e44ad", 1.6),
            "PPAL": ("#c77728", "h", "#c77728", 1.6),
            "BADGE": ("#ff7f0e", "d", "white", 1.6),
        }
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

    all_values: List[float] = []
    all_rounds: List[int] = []
    for exp_dir, strategy_name in zip(experiment_dirs, strategy_names):
        try:
            results = load_experiment_results(exp_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            continue
        
        round_metrics = list(zip(results['rounds'], results['metrics_history']))
        if max_round is not None:
            round_metrics = [
                (round_num, metrics)
                for round_num, metrics in round_metrics
                if round_num <= max_round
            ]
        filtered_rounds = [round_num for round_num, _ in round_metrics]
        metrics_history = [metrics for _, metrics in round_metrics]
        rounds = [round_num + round_offset for round_num in filtered_rounds]
        all_rounds.extend(rounds)
        
        map_values = []
        for metrics in metrics_history:
            map_values.append(_metric_value(metrics, metric_name))
        all_values.extend(map_values)

        if paper_style:
            color, marker, facecolor, edgewidth = style_by_name.get(
                strategy_name, ("#4c72b0", "o", "#4c72b0", 1.6)
            )
            if shadow_width is not None:
                ax.fill_between(
                    rounds,
                    [value - shadow_width for value in map_values],
                    [value + shadow_width for value in map_values],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
            ax.plot(
                rounds,
                map_values,
                label=strategy_name,
                color=color,
                linewidth=2.8,
                marker=marker,
                markersize=7.5,
                markerfacecolor=facecolor,
                markeredgecolor=color,
                markeredgewidth=edgewidth,
                zorder=2,
            )
        else:
            ax.plot(rounds, map_values, marker='o', linewidth=2,
                    label=strategy_name, markersize=6)

    metric_label = "mAP@50" if metric_name == "map50" else "mAP@0.5-0.95"
    if paper_style:
        ax.set_xlabel("Rounds", labelpad=12)
        ax.set_ylabel(metric_label, labelpad=12)
        if all_rounds:
            displayed_rounds = sorted(set(all_rounds))
            ax.set_xlim(displayed_rounds[0] - 0.35, displayed_rounds[-1] + 0.35)
            ax.set_xticks(displayed_rounds)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        elif all_values:
            ax.set_ylim(min(0.45, min(all_values) - 0.02), max(0.78, max(all_values) + 0.02))
        ax.grid(True, color="#b0b0b0", alpha=0.28, linewidth=1.1)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.6)
        ax.tick_params(width=1.6, length=8)
        ax.legend(loc="lower right", frameon=True, framealpha=0.94,
                  edgecolor="#c8c8c8", fancybox=True, handlelength=2.25)
    else:
        ax.set_xlabel('Active Learning Round')
        ax.set_ylabel(metric_label)
        ax.set_title('Active Learning Strategy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()


def analyze_dataset_statistics(dataset_path: str) -> Dict:
    from src.data.dataset import ALDataset
    dataset = ALDataset(dataset_path, "analysis", ["dummy"])
    
    stats = {
        'total_images': dataset.total_images,
        'labeled_images': len(dataset.get_labeled_indices()),
        'unlabeled_images': len(dataset.get_unlabeled_indices()),
    }
    
    return stats


def create_summary_report(experiment_dir: str, output_file: str = ""):
    exp_path = Path(experiment_dir)
    
    metadata_file = exp_path / "experiment_metadata.yaml"
    if not metadata_file.exists():
        print("Required files not found for summary report")
        return
    
    import yaml
    
    with open(metadata_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    try:
        results = load_experiment_results(exp_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return
    
    report = []
    report.append("# Active Learning Experiment Report")
    report.append("")
    report.append("## Experiment Configuration")
    report.append(f"- Dataset: {metadata['dataset_name']}")
    report.append(f"- Strategy: {metadata['strategy_name']}")
    report.append(f"- Model: {metadata['model_name']}")
    report.append(f"- Initial Labeled: {metadata['initial_labeled_count']}")
    report.append(f"- Created: {metadata['created_at']}")
    report.append("")
    
    report.append("## Results Summary")
    if results['metrics_history']:
        initial_map = _metric_value(results['metrics_history'][0])
        final_map = _metric_value(results['metrics_history'][-1])
        improvement = final_map - initial_map
        
        report.append(f"- Rounds Completed: {len(results['rounds'])}")
        report.append(f"- Initial mAP: {initial_map:.4f}")
        report.append(f"- Final mAP: {final_map:.4f}")
        report.append(f"- Total Improvement: {improvement:.4f}")
    
    report.append("")
    report.append("## Round-by-Round Results")
    report.append("| Round | mAP@0.5-0.95 |")
    report.append("|-------|--------------|")
    
    for i, (round_num, metrics) in enumerate(zip(results['rounds'], results['metrics_history'])):
        map_val = f"{_metric_value(metrics):.4f}"
        report.append(f"| {round_num} | {map_val} |")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    else:
        print(report_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils.py <command> [args...]")
        print("Commands:")
        print("  plot <results.csv|results.json|experiment_dir> [save_path]")
        print("  compare <results1> <results2> ... --names <name1> <name2> ... [--metric map50] [--paper-style] [--shadow-width 0.008] [--max-round 6 --round-offset 1] [--y-min 0.53 --y-max 0.60] [--output output.png]")
        print("  mean-std <seed1.csv> <seed2.csv> ... --name Random [--metric map50] [--paper-style] [--round-offset 1] [--caption '(b) On KITTI.'] [--output output.png]")
        print("  compare-mean-std --seed-results <seed1.csv> <seed2.csv> ... --seed-name Random --baselines <baseline1.csv> ... --baseline-names <name1> ... [--metric map50] [--paper-style] [--output output.png]")
        print("  compare-multi-mean-std --seed-groups Random=file1,file2,... Entropy=file1,file2,... --baselines <baseline1.csv> ... --baseline-names <name1> ... [--metric map50] [--paper-style] [--max-round 6 --round-offset 1 --y-min 0.45 --y-max 0.81 --y-ticks 0.5,0.6,0.7,0.8] [--output output.png]")
        print("  plot-summary <summary.csv> --strategies recal,badge,... [--paper-style] [--y-min 0.45 --y-max 0.83 --y-ticks 0.5,0.6,0.7,0.8] [--output output.png]")
        print("  report <experiment_dir> [output_file]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "plot":
        results_file = sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        plot_learning_curves(results_file, save_path) # type: ignore
        
    elif command == "compare":
        parser = argparse.ArgumentParser(description="Compare active-learning results")
        parser.add_argument("results", nargs="+", help="CSV, JSON, or experiment directories")
        parser.add_argument("--names", nargs="+", required=True, help="Strategy labels")
        parser.add_argument("--metric", default="map50-95", choices=("map50", "map50-95"))
        parser.add_argument("--paper-style", action="store_true", help="Use the reference comparison-plot style")
        parser.add_argument("--shadow-width", type=float, help="Half-width of a decorative translucent band around each curve")
        parser.add_argument("--max-round", type=int, help="Plot only rows up to this source round number")
        parser.add_argument("--round-offset", type=int, default=0, help="Value added to each round label when plotting")
        parser.add_argument("--y-min", type=float, help="Lower limit for the metric axis")
        parser.add_argument("--y-max", type=float, help="Upper limit for the metric axis")
        parser.add_argument("--output", default="", help="Output PNG path")
        args = parser.parse_args(sys.argv[2:])
        if (args.y_min is None) != (args.y_max is None):
            parser.error("--y-min and --y-max must be provided together")
        if args.y_min is not None and args.y_min >= args.y_max:
            parser.error("--y-min must be smaller than --y-max")
        if args.shadow_width is not None and args.shadow_width < 0:
            parser.error("--shadow-width must be non-negative")
        compare_strategies(
            args.results,
            args.names,
            args.output,
            metric_name=args.metric,
            paper_style=args.paper_style,
            y_limits=(args.y_min, args.y_max) if args.y_min is not None else None,
            round_offset=args.round_offset,
            max_round=args.max_round,
            shadow_width=args.shadow_width,
        )

    elif command == "mean-std":
        parser = argparse.ArgumentParser(description="Plot a strategy's mean and standard deviation across seeds")
        parser.add_argument("results", nargs="+", help="CSV, JSON, or experiment directories for one strategy")
        parser.add_argument("--name", required=True, help="Strategy label")
        parser.add_argument("--metric", default="map50", choices=("map50", "map50-95"))
        parser.add_argument("--paper-style", action="store_true", help="Use the reference comparison-plot style")
        parser.add_argument("--round-offset", type=int, default=0, help="Value added to each round label when plotting")
        parser.add_argument("--y-min", type=float, help="Lower limit for the metric axis")
        parser.add_argument("--y-max", type=float, help="Upper limit for the metric axis")
        parser.add_argument("--caption", default="", help="Optional caption below the plot")
        parser.add_argument("--output", default="", help="Output PNG path")
        args = parser.parse_args(sys.argv[2:])
        if (args.y_min is None) != (args.y_max is None):
            parser.error("--y-min and --y-max must be provided together")
        if args.y_min is not None and args.y_min >= args.y_max:
            parser.error("--y-min must be smaller than --y-max")
        plot_seed_mean_std(
            args.results,
            args.name,
            args.output,
            metric_name=args.metric,
            paper_style=args.paper_style,
            y_limits=(args.y_min, args.y_max) if args.y_min is not None else None,
            round_offset=args.round_offset,
            caption=args.caption,
        )

    elif command == "compare-mean-std":
        parser = argparse.ArgumentParser(description="Compare baselines with one mean/std seed curve")
        parser.add_argument("--seed-results", nargs="+", required=True, help="Seed CSVs for one strategy")
        parser.add_argument("--seed-name", required=True, help="Label for the seeded strategy")
        parser.add_argument("--baselines", nargs="+", required=True, help="Single-run baseline CSVs")
        parser.add_argument("--baseline-names", nargs="+", required=True, help="Labels for baseline CSVs")
        parser.add_argument("--metric", default="map50", choices=("map50", "map50-95"))
        parser.add_argument("--paper-style", action="store_true", help="Use the reference comparison-plot style")
        parser.add_argument("--round-offset", type=int, default=0, help="Value added to each round label when plotting")
        parser.add_argument("--y-min", type=float, help="Lower limit for the metric axis")
        parser.add_argument("--y-max", type=float, help="Upper limit for the metric axis")
        parser.add_argument("--caption", default="", help="Optional caption below the plot")
        parser.add_argument("--output", default="", help="Output PNG path")
        args = parser.parse_args(sys.argv[2:])
        if (args.y_min is None) != (args.y_max is None):
            parser.error("--y-min and --y-max must be provided together")
        if args.y_min is not None and args.y_min >= args.y_max:
            parser.error("--y-min must be smaller than --y-max")
        compare_with_seed_mean_std(
            args.seed_results,
            args.seed_name,
            args.baselines,
            args.baseline_names,
            args.output,
            metric_name=args.metric,
            paper_style=args.paper_style,
            y_limits=(args.y_min, args.y_max) if args.y_min is not None else None,
            round_offset=args.round_offset,
            caption=args.caption,
        )

    elif command == "compare-multi-mean-std":
        parser = argparse.ArgumentParser(description="Compare several mean/std strategies with single-run baselines")
        parser.add_argument(
            "--seed-groups", nargs="+", required=True,
            help="One or more NAME=seed1.csv,seed2.csv,... specifications",
        )
        parser.add_argument("--baselines", nargs="*", default=[], help="Single-run baseline CSVs")
        parser.add_argument("--baseline-names", nargs="*", default=[], help="Labels for baseline CSVs")
        parser.add_argument("--metric", default="map50", choices=("map50", "map50-95"))
        parser.add_argument("--paper-style", action="store_true", help="Use the reference comparison-plot style")
        parser.add_argument("--round-offset", type=int, default=0, help="Value added to each round label when plotting")
        parser.add_argument("--max-round", type=int, help="Plot only rows up to this source round number")
        parser.add_argument("--y-min", type=float, help="Lower limit for the metric axis")
        parser.add_argument("--y-max", type=float, help="Upper limit for the metric axis")
        parser.add_argument("--y-tick-step", type=float, help="Spacing between metric-axis ticks")
        parser.add_argument("--y-ticks", help="Comma-separated metric-axis tick positions")
        parser.add_argument("--caption", default="", help="Optional caption below the plot")
        parser.add_argument("--output", default="", help="Output PNG path")
        args = parser.parse_args(sys.argv[2:])
        if (args.y_min is None) != (args.y_max is None):
            parser.error("--y-min and --y-max must be provided together")
        if args.y_min is not None and args.y_min >= args.y_max:
            parser.error("--y-min must be smaller than --y-max")
        if args.y_tick_step is not None and args.y_tick_step <= 0:
            parser.error("--y-tick-step must be positive")
        if args.y_tick_step is not None and args.y_min is None:
            parser.error("--y-tick-step requires --y-min and --y-max")
        if args.y_ticks is not None and args.y_tick_step is not None:
            parser.error("Use either --y-ticks or --y-tick-step, not both")
        try:
            y_ticks = [float(value) for value in args.y_ticks.split(",")] if args.y_ticks else None
        except ValueError:
            parser.error("--y-ticks must be comma-separated numbers")
        if y_ticks is not None and not y_ticks:
            parser.error("--y-ticks must contain at least one value")
        seed_groups: Dict[str, List[str]] = {}
        for specification in args.seed_groups:
            strategy_name, separator, files = specification.partition("=")
            results_files = [path for path in files.split(",") if path]
            if not separator or not strategy_name or len(results_files) < 2:
                parser.error(
                    "Each --seed-groups value must be NAME=seed1.csv,seed2.csv,..."
                )
            if strategy_name in seed_groups:
                parser.error(f"Duplicate seed-group name: {strategy_name}")
            seed_groups[strategy_name] = results_files
        compare_multiple_seed_mean_std(
            seed_groups,
            args.baselines,
            args.baseline_names,
            args.output,
            metric_name=args.metric,
            paper_style=args.paper_style,
            y_limits=(args.y_min, args.y_max) if args.y_min is not None else None,
            round_offset=args.round_offset,
            max_round=args.max_round,
            y_tick_step=args.y_tick_step,
            y_ticks=y_ticks,
            caption=args.caption,
        )

    elif command == "plot-summary":
        parser = argparse.ArgumentParser(description="Plot mean/std data already aggregated in a summary CSV")
        parser.add_argument("summary_file", help="CSV with display_round and <strategy>_mean/std columns")
        parser.add_argument("--strategies", required=True, help="Comma-separated strategy column prefixes")
        parser.add_argument("--paper-style", action="store_true", help="Use the reference comparison-plot style")
        parser.add_argument("--y-min", type=float, help="Lower limit for the metric axis")
        parser.add_argument("--y-max", type=float, help="Upper limit for the metric axis")
        parser.add_argument("--y-ticks", help="Comma-separated metric-axis tick positions")
        parser.add_argument("--output", default="", help="Output PNG path")
        args = parser.parse_args(sys.argv[2:])
        if (args.y_min is None) != (args.y_max is None):
            parser.error("--y-min and --y-max must be provided together")
        if args.y_min is not None and args.y_min >= args.y_max:
            parser.error("--y-min must be smaller than --y-max")
        try:
            y_ticks = [float(value) for value in args.y_ticks.split(",")] if args.y_ticks else None
        except ValueError:
            parser.error("--y-ticks must be comma-separated numbers")
        strategies = [name.strip() for name in args.strategies.split(",") if name.strip()]
        if not strategies:
            parser.error("--strategies must contain at least one strategy")
        plot_summary_mean_std(
            args.summary_file, strategies, args.output, paper_style=args.paper_style,
            y_limits=(args.y_min, args.y_max) if args.y_min is not None else None,
            y_ticks=y_ticks,
        )

    elif command == "report":
        exp_dir = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        create_summary_report(exp_dir, output_file) # type: ignore
        
    else:
        print(f"Unknown command: {command}")
