"""Plot lower/upper tail seed-band panels for each metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = EXPERIMENT_DIR / "outputs" / "nnngp" / "tail_event_metric_summaries"
DEFAULT_NNNGP_CSV = DEFAULT_SUMMARY_DIR / "tail_event_metric_summary.csv"
DEFAULT_NNGP_NNMP_CSV = DEFAULT_SUMMARY_DIR / "tail_event_metric_summary_nngp_nnmp.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_DIR / "seed_band_plots"
AXIS_LABEL_FONTSIZE = 14

METHODS = ("NNnGP", "NNGP", "NNMP")
DRAW_ORDER = ("NNMP", "NNGP", "NNnGP")
TAILS = ("lower", "upper")
METRICS = {
    "brier_skill_score": {
        "title": "Brier Skill Score",
        "ylabel": "BSS",
        "scale": 100.0,
        "target": None,
    },
    "twcrps_tail_weighted": {
        "title": "Tail-weighted CRPS",
        "ylabel": "twCRPS",
        "scale": 1.0,
        "target": None,
    },
    "extreme_value_coverage_95": {
        "title": "95% Extreme-value Coverage",
        "ylabel": "Extreme CI Coverage",
        "scale": 100.0,
        "target": 95.0,
    },
    "extreme_value_ci_width_95": {
        "title": "95% Extreme-value CI Width",
        "ylabel": "Extreme CI Width",
        "scale": 1.0,
        "target": None,
    },
}
COLORS = {"NNnGP": "#FFC20A", "NNGP": "#3B9C97", "NNMP": "#7F3C78"}
BAND_COLORS = {**COLORS, "NNGP": "#74C69D"}

def load_summaries(nnngp_csv: Path, nngp_nnmp_csv: Path) -> pd.DataFrame:
    nnngp = pd.read_csv(nnngp_csv)
    other = pd.read_csv(nngp_nnmp_csv)
    data = pd.concat([nnngp, other], ignore_index=True)
    data = data.loc[data["method"].isin(METHODS) & data["tail"].isin(TAILS)].copy()

    required = {"random_seed", "method", "tail", "quantile_probability", *METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Metric input is missing columns: {sorted(missing)}")
    if data.duplicated(["random_seed", "method", "tail", "quantile_probability"]).any():
        raise ValueError("Duplicate seed/method/tail/quantile rows found in metric inputs")

    counts = data.groupby(["method", "tail", "quantile_probability"])["random_seed"].nunique()
    incomplete = counts[counts.ne(10)]
    if not incomplete.empty:
        raise ValueError(
            "Expected exactly 10 seeds for every method/tail/quantile group; "
            f"incomplete groups:\n{incomplete.to_string()}"
        )
    return data


def seed_band(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby(["method", "tail", "quantile_probability"], as_index=False)[metric]
        .agg(
            median="median",
            lower_10=lambda values: values.quantile(0.10),
            upper_90=lambda values: values.quantile(0.90),
        )
        .sort_values("quantile_probability")
    )


def shared_ylim(summary: pd.DataFrame, scale: float, target: float | None) -> tuple[float, float]:
    values = summary[["median", "lower_10", "upper_90"]].to_numpy(dtype=float).ravel() * scale
    if target is not None:
        values = np.concatenate([values, np.array([float(target)])])
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    span = ymax - ymin
    pad = span * 0.06 if span > 0 else max(abs(ymax) * 0.06, 1.0)
    return ymin - pad, ymax + pad


def plot_metric(data: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    config = METRICS[metric]
    summary = seed_band(data, metric)
    scale = float(config["scale"])
    ylim = shared_ylim(summary, scale, config["target"])
    if metric == "extreme_value_coverage_95":
        ylim = (0.0, 100.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True, sharey=True)

    xticks = np.round(np.arange(0.80, 0.981, 0.03), 2)
    for axis, tail in zip(axes, TAILS):
        for draw_index, method in enumerate(DRAW_ORDER):
            rows = summary.loc[summary["method"].eq(method) & summary["tail"].eq(tail)]
            x = rows["quantile_probability"].to_numpy(dtype=float)
            median = rows["median"].to_numpy(dtype=float) * scale
            lower = rows["lower_10"].to_numpy(dtype=float) * scale
            upper = rows["upper_90"].to_numpy(dtype=float) * scale
            color = COLORS[method]
            axis.fill_between(
                x,
                lower,
                upper,
                color=BAND_COLORS[method],
                alpha=0.18,
                linewidth=0,
                zorder=1 + draw_index,
            )
            axis.plot(
                x,
                median,
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4.2,
                zorder=10 + draw_index,
            )

        if config["target"] is not None:
            target = float(config["target"])
            axis.axhline(
                target,
                color="#555555",
                linestyle="--",
                linewidth=1.2,
            )
        axis.set_xlim(0.795, 0.985)
        axis.set_xticks(xticks)
        axis.set_xticklabels([f"{tick:.2f}" for tick in xticks])
        axis.set_xlabel("probability", fontsize=AXIS_LABEL_FONTSIZE)
        axis.set_ylabel("")
        axis.set_title("")
        axis.set_ylim(*ylim)
        if config["target"] is not None:
            target = float(config["target"])
            ticks = axis.get_yticks()
            ticks = np.array([tick for tick in ticks if not np.isclose(tick, target)])
            ticks = np.sort(np.append(ticks, target))
            axis.set_yticks(ticks)
        axis.grid(True, linewidth=0.45, alpha=0.4)
    axes[0].tick_params(axis="y", labelleft=True)
    axes[1].tick_params(axis="y", labelleft=False)

    axes[0].set_ylabel(str(config["ylabel"]), fontsize=AXIS_LABEL_FONTSIZE)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.14, wspace=0.04)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metric}_lower_upper_seed_median_q10_q90.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_method_legend(output_dir: Path) -> Path:
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linewidth=2.2,
            marker="o",
            markersize=6.5,
        )
        for method in METHODS
    ]
    fig = plt.figure(figsize=(4.8, 0.55))
    fig.legend(
        handles,
        list(METHODS),
        loc="center",
        ncol=len(METHODS),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "method_legend.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnngp-csv", type=Path, default=DEFAULT_NNNGP_CSV)
    parser.add_argument("--nngp-nnmp-csv", type=Path, default=DEFAULT_NNGP_NNMP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    data = load_summaries(args.nnngp_csv.resolve(), args.nngp_nnmp_csv.resolve())
    for metric in METRICS:
        path = plot_metric(data, metric, args.output_dir.resolve())
        print(f"Saved {metric}: {path}")
    legend_path = plot_method_legend(args.output_dir.resolve())
    print(f"Saved legend: {legend_path}")


if __name__ == "__main__":
    main()
