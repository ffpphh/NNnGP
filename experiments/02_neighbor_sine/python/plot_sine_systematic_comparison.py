#!/usr/bin/env python3
"""Plot systematic m-sensitivity metrics for NNnGP, NNGP, and NNMP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
DEFAULT_NNNGP_CSV = OUTPUT_ROOT / "no_split" / "nnngp" / "systematic_summary.csv"
DEFAULT_NNGP_CSV = OUTPUT_ROOT / "no_split" / "nngp" / "summary.csv"
DEFAULT_NNMP_CSV = OUTPUT_ROOT / "no_split" / "nnmp" / "summary.csv"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "no_split" / "comparison_plots"
COMPARISON_CONFIGS = {
    "no_split": {
        "nnngp_csv": OUTPUT_ROOT / "no_split" / "nnngp" / "systematic_summary.csv",
        "nngp_csv": OUTPUT_ROOT / "no_split" / "nngp" / "summary.csv",
        "nnmp_csv": OUTPUT_ROOT / "no_split" / "nnmp" / "summary.csv",
        "output_dir": OUTPUT_ROOT / "no_split" / "comparison_plots",
        "filename_suffix": "_no_split",
    },
    "split": {
        "nnngp_csv": OUTPUT_ROOT / "split" / "nnngp" / "systematic_summary.csv",
        "nngp_csv": OUTPUT_ROOT / "split" / "nngp" / "summary.csv",
        "nnmp_csv": OUTPUT_ROOT / "split" / "nnmp" / "summary.csv",
        "output_dir": OUTPUT_ROOT / "split" / "comparison_plots",
        "filename_suffix": "_split",
    },
}
METHOD_COLORS = {
    "NNnGP": "#FFC20A",
    "NNGP": "#3B9C97",
    "NNMP": "#7F3C78",
}
METHOD_ORDER = ["NNMP", "NNGP", "NNnGP"]
ORDER_STYLES = {
    "maximin": ("-", "o"),
    "coordinate": ("--", "s"),
    "random1": ("-.", "^"),
    "random2": (":", "D"),
    "random3": ((0, (3, 1, 1, 1)), "v"),
}
ORDER_LABELS = {
    "coordinate": "coordinate",
}
METRIC_LABELS = {
    "RSR": "RSR (%)",
    "CI_width": "CI width",
    "CI_coverage_percent": "CI coverage (%)",
}


def _order_column(df: pd.DataFrame) -> str | None:
    if "order_type" in df.columns:
        return "order_type"
    if "order_name" in df.columns:
        return "order_name"
    return None


def normalize_order_type(value: object) -> str:
    text = str(value)
    if text in {"x1_x2", "x2_x1"}:
        return "coordinate"
    return text


def load_all_orders_table(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"m", "RSR", "CRPS", "CI_width", "CI_coverage_percent"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")]

    order_col = _order_column(df)
    if order_col is None:
        raise ValueError(f"{path} must contain order_type or order_name")

    keep = ["m", "RSR", "CRPS", "CI_width", "CI_coverage_percent", order_col]
    if "order_id" in df.columns:
        keep.append("order_id")
    df = df[keep].copy()
    df = df.rename(columns={order_col: "order_type"})
    df["order_type"] = df["order_type"].map(normalize_order_type)
    if "order_id" not in df.columns:
        order_lookup = {name: i + 1 for i, name in enumerate(sorted(df["order_type"].unique()))}
        df["order_id"] = df["order_type"].map(order_lookup)

    df["m"] = pd.to_numeric(df["m"], errors="coerce")
    df["RSR"] = pd.to_numeric(df["RSR"], errors="coerce")
    df["CRPS"] = pd.to_numeric(df["CRPS"], errors="coerce")
    df["CI_width"] = pd.to_numeric(df["CI_width"], errors="coerce")
    df["CI_coverage_percent"] = pd.to_numeric(df["CI_coverage_percent"], errors="coerce")
    df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")
    df = df.dropna(
        subset=["m", "RSR", "CRPS", "CI_width", "CI_coverage_percent", "order_id", "order_type"]
    )
    df["m"] = df["m"].astype(int)
    df["order_id"] = df["order_id"].astype(int)
    df["method"] = method
    return df.sort_values(["method", "order_id", "m"])


def plot_metric_all_orders(df: pd.DataFrame, metric: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    order_sort = df[["order_id", "order_type"]].drop_duplicates().sort_values("order_id")
    for method in METHOD_ORDER:
        method_df = df[df["method"].eq(method)]
        if method_df.empty:
            continue
        for _, order_row in order_sort.iterrows():
            order_type = order_row["order_type"]
            group = method_df[method_df["order_type"].eq(order_type)].sort_values("m")
            if group.empty:
                continue
            linestyle, marker = ORDER_STYLES.get(order_type, ("-", "o"))
            ax.plot(
                group["m"],
                group[metric],
                color=METHOD_COLORS.get(method),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.2,
                markersize=10.0,
                alpha=0.95,
            )

    ax.set_xlabel("m", fontsize=17, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xticks(sorted(df["m"].unique()))
    if metric == "RSR":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.margins(y=0.03)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    fig.savefig(output)
    plt.close(fig)


def plot_legend(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 2.2), constrained_layout=True)
    ax.axis("off")
    handles = []
    labels = []
    order_sort = df[["order_id", "order_type"]].drop_duplicates().sort_values("order_id")
    display_methods = ["NNnGP", "NNGP", "NNMP"]
    for method in display_methods:
        if method not in set(df["method"]):
            continue
        for _, order_row in order_sort.iterrows():
            order_type = order_row["order_type"]
            linestyle, marker = ORDER_STYLES.get(order_type, ("-", "o"))
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=METHOD_COLORS.get(method),
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.2,
                    markersize=8.5,
                )
            )
            labels.append(f"{method} | {ORDER_LABELS.get(order_type, order_type)}")
    ax.legend(
        handles,
        labels,
        ncol=3,
        fontsize=9.5,
        frameon=False,
        loc="center",
        labelspacing=0.55,
        columnspacing=1.7,
        handlelength=3.2,
        handletextpad=0.65,
        markerscale=1.15,
    )
    fig.savefig(output)
    plt.close(fig)


def generate_comparison_plots(
    nnngp_csv: Path,
    nngp_csv: Path,
    nnmp_csv: Path,
    output_dir: Path,
    filename_suffix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_orders = pd.concat(
        [
            load_all_orders_table(nnngp_csv, "NNnGP"),
            load_all_orders_table(nngp_csv, "NNGP"),
            load_all_orders_table(nnmp_csv, "NNMP"),
        ],
        ignore_index=True,
    )
    all_orders_csv = output_dir / "method_order_metrics.csv"
    all_orders.to_csv(all_orders_csv, index=False)

    for metric in ("RSR", "CRPS", "CI_width", "CI_coverage_percent"):
        plot_path = output_dir / f"{metric}_all_orders_methods{filename_suffix}.pdf"
        plot_metric_all_orders(all_orders, metric, plot_path)
        print(f"Saved plot: {plot_path}")
    legend_path = output_dir / f"methods_orders_legend{filename_suffix}.pdf"
    plot_legend(all_orders, legend_path)
    print(f"Saved legend: {legend_path}")
    print(f"Saved all-order metric table: {all_orders_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create line plots comparing NNnGP, NNGP, and NNMP across m."
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "no_split", "split"],
        default="all",
        help="Named comparison to generate. Default generates both paired comparisons.",
    )
    parser.add_argument("--nnngp-csv", type=Path, default=DEFAULT_NNNGP_CSV)
    parser.add_argument("--nngp-csv", type=Path, default=DEFAULT_NNGP_CSV)
    parser.add_argument("--nnmp-csv", type=Path, default=DEFAULT_NNMP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    explicit_paths = any(
        option in sys.argv
        for option in ("--nnngp-csv", "--nngp-csv", "--nnmp-csv", "--output-dir")
    )
    if explicit_paths:
        generate_comparison_plots(args.nnngp_csv, args.nngp_csv, args.nnmp_csv, args.output_dir)
        return

    experiment_names = (
        COMPARISON_CONFIGS.keys()
        if args.experiment == "all"
        else [args.experiment]
    )
    for experiment_name in experiment_names:
        config = COMPARISON_CONFIGS[experiment_name]
        print(f"\nGenerating paired comparison: {experiment_name}")
        generate_comparison_plots(**config)


if __name__ == "__main__":
    main()
