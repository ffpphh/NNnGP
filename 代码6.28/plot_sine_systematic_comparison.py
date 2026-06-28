#!/usr/bin/env python3
"""Plot systematic m-sensitivity metrics for NNnGP, NNGP, and NNMP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


DEFAULT_NNNGP_CSV = Path(
    "/home/dongyuhan/NNnGP/results/vi_results/exp_neighbor_sine_systematic_vi_nf/systematic_summary.csv"
)
DEFAULT_NNGP_CSV = Path("/home/dongyuhan/NNnGP/results/nngp/summary.csv")
DEFAULT_NNMP_CSV = Path("/home/dongyuhan/NNnGP/results/nnmp/summary.csv")
DEFAULT_OUTPUT_DIR = Path(
    "/home/dongyuhan/NNnGP/results/vi_results/exp_neighbor_sine_systematic_vi_nf/method_comparison_plots"
)
METHOD_COLORS = {
    "NNnGP": "#2F6B9A",
    "NNGP": "#C44E52",
    "NNMP": "#4C8C3B",
}
ORDER_STYLES = {
    "maximin": ("-", "o"),
    "x1_x2": ("--", "s"),
    "random1": ("-.", "^"),
    "random2": (":", "D"),
    "random3": ((0, (3, 1, 1, 1)), "v"),
}
ORDER_LABELS = {
    "x1_x2": "coordinates",
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
    for method, method_df in df.groupby("method", sort=False):
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
                linewidth=2.0,
                markersize=8.0,
                alpha=0.95,
            )

    ax.set_xlabel("m", fontsize=15, fontweight="bold")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=15, fontweight="bold")
    ax.set_xticks(sorted(df["m"].unique()))
    if metric == "RSR":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.margins(y=0.03)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    fig.savefig(output)
    plt.close(fig)


def plot_legend(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.4), constrained_layout=True)
    ax.axis("off")
    handles = []
    labels = []
    order_sort = df[["order_id", "order_type"]].drop_duplicates().sort_values("order_id")
    for method in df["method"].drop_duplicates():
        for _, order_row in order_sort.iterrows():
            order_type = order_row["order_type"]
            linestyle, marker = ORDER_STYLES.get(order_type, ("-", "o"))
            handle = plt.Line2D(
                [0],
                [0],
                color=METHOD_COLORS.get(method),
                linestyle=linestyle,
                marker=marker,
                linewidth=2.0,
                markersize=8.0,
            )
            handles.append(handle)
            labels.append(f"{method} | {ORDER_LABELS.get(order_type, order_type)}")
    ax.legend(
        handles,
        labels,
        ncol=3,
        fontsize=11,
        frameon=True,
        loc="center",
        borderpad=0.9,
        labelspacing=0.85,
        columnspacing=1.4,
        handlelength=2.4,
        handletextpad=0.65,
        markerscale=1.2,
    )
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create line plots comparing NNnGP, NNGP, and NNMP across m."
    )
    parser.add_argument("--nnngp-csv", type=Path, default=DEFAULT_NNNGP_CSV)
    parser.add_argument("--nngp-csv", type=Path, default=DEFAULT_NNGP_CSV)
    parser.add_argument("--nnmp-csv", type=Path, default=DEFAULT_NNMP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_orders = pd.concat(
        [
            load_all_orders_table(args.nnngp_csv, "NNnGP"),
            load_all_orders_table(args.nngp_csv, "NNGP"),
            load_all_orders_table(args.nnmp_csv, "NNMP"),
        ],
        ignore_index=True,
    )
    all_orders_csv = args.output_dir / "method_order_metrics.csv"
    all_orders.to_csv(all_orders_csv, index=False)

    for metric in ("RSR", "CRPS", "CI_width", "CI_coverage_percent"):
        plot_metric_all_orders(all_orders, metric, args.output_dir / f"{metric}_all_orders_methods.pdf")
        print(f"Saved plot: {args.output_dir / f'{metric}_all_orders_methods.pdf'}")
    plot_legend(all_orders, args.output_dir / "methods_orders_legend.pdf")
    print(f"Saved legend: {args.output_dir / 'methods_orders_legend.pdf'}")
    print(f"Saved all-order metric table: {all_orders_csv}")


if __name__ == "__main__":
    main()
