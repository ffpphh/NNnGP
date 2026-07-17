"""Compute and combine tail-event metrics for Montana 800m split-seed VI runs."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from compute_tx_tail_event_metrics import (
    DEFAULT_QUANTILE_PROBS,
    VALUE_COL,
    load_nnngp_samples,
    plot_metric_lines,
    summarize_method,
)


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
RUN_NAME = "prism_montana_october_ppt_2025_800m_split_seeds"
DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "outputs" / "nnngp"
DEFAULT_DATA_DIR = EXPERIMENT_DIR / "data" / "split_seeds"
ORIGINAL_DATA_FILENAME = "montana_800m_original_data.csv"
ORDER_TABLE_FILENAME = "montana_800m_split_point_orders.csv"
SEED_RESULTS_DIRNAME = "seed_results"


def parse_seeds(text: str | None, available: list[int]) -> list[int]:
    if text is None:
        return available
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    missing = sorted(set(seeds).difference(available))
    if missing:
        raise ValueError(f"Seeds absent from point-order table: {missing}")
    return seeds


def write_split_csv(
    original: pd.DataFrame,
    point_order: np.ndarray,
    train_size: int,
    output_path: Path,
) -> None:
    if len(point_order) != len(original) or not np.array_equal(
        np.sort(point_order), np.arange(len(original))
    ):
        raise ValueError("Point order must be a permutation of every original-data row index")
    ordered = original.iloc[point_order].reset_index(drop=True).copy()
    ordered.insert(2, "split", np.where(np.arange(len(ordered)) < train_size, "S", "U"))
    ordered[["lon", "lat", "split", VALUE_COL]].to_csv(output_path, index=False)


def summarize_seed(seed: int, result_dir: Path, data_csv: Path) -> list[dict]:
    coords, truth, samples, point_set = load_nnngp_samples(result_dir, data_csv)
    rows = []
    for probability in DEFAULT_QUANTILE_PROBS:
        magnitude = float(norm.ppf(probability))
        for tail, threshold in (("lower", -magnitude), ("upper", magnitude)):
            metric = summarize_method(
                "NNnGP",
                coords,
                truth,
                samples,
                point_set,
                threshold,
                tail,
                quantile_probability=float(probability),
            )
            rows.append({"random_seed": seed, **metric})
    return rows


def save_global_metrics_by_seed(result_dir: Path, seeds: list[int], output_dir: Path) -> Path:
    rows = []
    for seed in seeds:
        summary_path = result_dir / SEED_RESULTS_DIRNAME / f"seed_{seed}" / "summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing global summary for seed {seed}: {summary_path}")
        summary = pd.read_csv(summary_path)
        if len(summary) != 1:
            raise ValueError(f"Expected one row in {summary_path}, found {len(summary)}")
        row = summary.iloc[0]
        rows.append(
            {
                "random_seed": seed,
                "method": row["method"],
                "RSR": float(row["RSR"]),
                "CRPS": float(row["CRPS"]),
                "CI_rate": float(row["CI_coverage_percent"]),
                "CI_width": float(row["CI_width"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "global_metric_summary_by_seed.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved global metrics by seed: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated subset; default uses every row in the order table.")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Only combine existing per-seed global summaries; skip tail-event recomputation.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    data_dir = args.data_dir.resolve()
    original = pd.read_csv(data_dir / ORIGINAL_DATA_FILENAME)
    orders = pd.read_csv(data_dir / ORDER_TABLE_FILENAME)
    available = orders["random_seed"].astype(int).tolist()
    seeds = parse_seeds(args.seeds, available)
    order_columns = [column for column in orders.columns if column != "random_seed"]
    output_dir = result_dir / "tail_event_metric_summaries"
    save_global_metrics_by_seed(result_dir, seeds, output_dir)
    if args.global_only:
        return

    rows = []
    with tempfile.TemporaryDirectory(prefix="montana_800m_metric_splits_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for seed in seeds:
            seed_result_dir = result_dir / SEED_RESULTS_DIRNAME / f"seed_{seed}"
            required = [seed_result_dir / "vi_results.npz", seed_result_dir / "vi_prediction_results.npz"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Cannot summarize seed {seed}; missing {missing}")

            point_order = orders.loc[
                orders["random_seed"].astype(int).eq(seed), order_columns
            ].iloc[0].to_numpy(dtype=np.int64)
            split_csv = tmp_dir / f"seed_{seed}.csv"
            write_split_csv(original, point_order, args.train_size, split_csv)
            rows.extend(summarize_seed(seed, seed_result_dir, split_csv))
            print(f"Computed tail-event metrics for seed {seed}")

    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "tail_event_metric_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_data = summary.copy()
    plot_data["method"] = plot_data.apply(
        lambda row: f"{row['method']} seed {int(row['random_seed'])}", axis=1
    )
    plot_metric_lines(plot_data, output_dir)
    print(f"Saved combined tail-event summary: {summary_path}")
    print(f"Saved metric plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
