"""Create separate seed-43 truth/method heatmaps and a shared standalone colorbar."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import rdata


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CSV = EXPERIMENT_DIR / "data" / "split_seeds" / "montana_800m_original_data.csv"
DEFAULT_ORDER_CSV = EXPERIMENT_DIR / "data" / "split_seeds" / "montana_800m_split_point_orders.csv"
DEFAULT_NNNGP_DIR = EXPERIMENT_DIR / "outputs" / "nnngp" / "seed_results" / "seed_43"
DEFAULT_NNGP_DIR = EXPERIMENT_DIR / "outputs" / "r_nngp" / "seed_43"
DEFAULT_NNMP_DIR = EXPERIMENT_DIR / "outputs" / "r_nnmp" / "seed_43"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "plots" / "seed_43_mean_heatmaps"
VALUE_COL = "log_ppt_2025_standardized"
SEED = 43
TRAIN_SIZE = 800
FIGSIZE = (10.0, 4.8)
DPI = 300

def load_seed_order(original: pd.DataFrame, order_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    orders = pd.read_csv(order_csv)
    row = orders.loc[orders["random_seed"].astype(int).eq(SEED)]
    if len(row) != 1:
        raise ValueError(f"Expected exactly one point-order row for seed {SEED}, found {len(row)}")
    order = row.drop(columns="random_seed").iloc[0].to_numpy(dtype=np.int64)
    if len(order) != len(original) or not np.array_equal(np.sort(order), np.arange(len(original))):
        raise ValueError("Seed-43 point order is not a complete permutation of original-data indices")
    ordered_coords = original.iloc[order][["lon", "lat"]].to_numpy(dtype=float)
    return ordered_coords[:TRAIN_SIZE], ordered_coords[TRAIN_SIZE:]


def load_nnngp_mean(result_dir: Path, s_coords: np.ndarray, u_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vi = np.load(result_dir / "vi_results.npz", allow_pickle=True)
    pred = np.load(result_dir / "vi_prediction_results.npz", allow_pickle=True)
    all_coords = np.vstack([s_coords, u_coords])
    mins = all_coords.min(axis=0)
    spans = np.maximum(np.ptp(all_coords, axis=0), 1e-12)
    scaled_s = (s_coords - mins) / spans
    x_s = np.column_stack([np.ones(len(s_coords)), scaled_s])

    w_s = np.asarray(vi["w_S_samples"], dtype=float)
    beta = np.asarray(vi["beta_samples"], dtype=float)
    s_mean = np.mean(beta @ x_s.T + w_s, axis=0)
    u_mean = np.mean(np.asarray(pred["pred_samples"], dtype=float), axis=0)
    if len(s_mean) != len(s_coords) or len(u_mean) != len(u_coords):
        raise ValueError("NNnGP posterior dimensions do not match seed-43 S/U coordinates")
    return all_coords, np.concatenate([s_mean, u_mean])


def rds_location_means(path: Path, expected_rows: int) -> np.ndarray:
    samples = rdata.read_rds(path)
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} location rows in {path}, found {values.shape}")
    means = np.mean(values, axis=1)
    del samples, values
    gc.collect()
    return means


def load_rds_method_mean(method_dir: Path, s_coords: np.ndarray, u_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_mean = rds_location_means(method_dir / "posterior_samples_y_S.rds", len(s_coords))
    u_mean = rds_location_means(method_dir / "posterior_samples_y_U.rds", len(u_coords))
    return np.vstack([s_coords, u_coords]), np.concatenate([s_mean, u_mean])


def plot_heatmap(
    coords: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    axis = fig.add_axes([0.10, 0.14, 0.87, 0.82])
    axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=5,
        marker="s",
        linewidths=0,
        cmap="RdBu",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_colorbar(output_path: Path, vmin: float, vmax: float) -> None:
    fig = plt.figure(figsize=(FIGSIZE[0], 0.9))
    color_axis = fig.add_axes([0.10, 0.35, 0.87, 0.28])
    fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="RdBu"),
        cax=color_axis,
        orientation="horizontal",
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--order-csv", type=Path, default=DEFAULT_ORDER_CSV)
    parser.add_argument("--nnngp-dir", type=Path, default=DEFAULT_NNNGP_DIR)
    parser.add_argument("--nngp-dir", type=Path, default=DEFAULT_NNGP_DIR)
    parser.add_argument("--nnmp-dir", type=Path, default=DEFAULT_NNMP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    original = pd.read_csv(args.data_csv.resolve())
    required = {"lon", "lat", VALUE_COL}
    missing = required.difference(original.columns)
    if missing:
        raise ValueError(f"Original data is missing columns: {sorted(missing)}")
    s_coords, u_coords = load_seed_order(original, args.order_csv.resolve())
    truth = (
        original[["lon", "lat", VALUE_COL]]
        .rename(columns={VALUE_COL: "value"})
    )

    results = {
        "truth": (truth[["lon", "lat"]].to_numpy(dtype=float), truth["value"].to_numpy(dtype=float)),
        "nnngp": load_nnngp_mean(args.nnngp_dir.resolve(), s_coords, u_coords),
        "nngp": load_rds_method_mean(args.nngp_dir.resolve(), s_coords, u_coords),
        "nnmp": load_rds_method_mean(args.nnmp_dir.resolve(), s_coords, u_coords),
    }
    all_values = np.concatenate([values[np.isfinite(values)] for _, values in results.values()])
    limit = float(np.max(np.abs(all_values)))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (coords, values) in results.items():
        path = output_dir / f"seed43_{name}_posterior_mean_heatmap.pdf"
        plot_heatmap(coords, values, path, -limit, limit)
        print(f"Saved {name}: {path}")
    colorbar_path = output_dir / "seed43_shared_colorbar.pdf"
    plot_colorbar(colorbar_path, -limit, limit)
    print(f"Saved shared colorbar: {colorbar_path}")
    print(f"Shared color range: [{-limit:.6g}, {limit:.6g}]")


if __name__ == "__main__":
    main()
