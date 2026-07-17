"""Create standalone visualizations for the VI tau-sensitivity experiment."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = str(EXPERIMENT_DIR / "outputs")
DEFAULT_SIMULATION_DIR = str(EXPERIMENT_DIR / "data")
EXTERNAL_RESULTS_DIR = str(EXPERIMENT_DIR / "outputs" / "r_baselines")
RESULTS_DIR = DEFAULT_RESULTS_DIR
SIMULATION_DIR = DEFAULT_SIMULATION_DIR
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, "matern_gp_nnngp")
HMC_MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, "matern_gp_nnngp")

DATA_TYPES = ("weak", "median", "strong")
STRONG_Y_COLOR_LIMIT = 36.0
STRONG_Y_COLOR_TICKS = np.linspace(
    -STRONG_Y_COLOR_LIMIT,
    STRONG_Y_COLOR_LIMIT,
    5,
)
def configure_result_paths(results_dir=DEFAULT_RESULTS_DIR, optimize_z=False):
    """Configure input/output directories for fixed-Z or optimized-Z results."""
    global RESULTS_DIR, SIMULATION_DIR, VISUALIZATION_DIR, MODEL_RESULTS_DIR, HMC_MODEL_RESULTS_DIR

    RESULTS_DIR = os.path.abspath(results_dir)
    inference_root = os.path.join(RESULTS_DIR, "optimized_z") if optimize_z else RESULTS_DIR
    SIMULATION_DIR = DEFAULT_SIMULATION_DIR
    VISUALIZATION_DIR = os.path.join(inference_root, "visualizations")
    MODEL_RESULTS_DIR = os.path.join(inference_root, "matern_gp_nnngp")
    HMC_MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, "matern_gp_nnngp")


def simulation_npz_path(data_type):
    return os.path.join(
        SIMULATION_DIR,
        data_type,
        "matern_gp_nnngp_data.npz",
    )


def simulation_csv_path(data_type):
    return os.path.splitext(simulation_npz_path(data_type))[0] + ".csv"


def load_simulation_csv_rows(data_type):
    csv_path = simulation_csv_path(data_type)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Simulation data not found: {simulation_npz_path(data_type)} or {csv_path}"
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"row_order", "point_index", "split", "x", "y", "y_obs"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV simulation data missing columns: {sorted(missing)}")
        return list(reader)


def load_simulation_points_and_indices(data_type):
    """Load all grid locations and point indices from NPZ, falling back to CSV."""
    data_path = simulation_npz_path(data_type)
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        return (
            np.asarray(data["all_points"], dtype=np.float64),
            np.arange(len(data["all_points"]), dtype=int),
        )

    rows = load_simulation_csv_rows(data_type)
    rows = sorted(rows, key=lambda row: int(row["point_index"]))
    point_indices = np.asarray([int(row["point_index"]) for row in rows], dtype=int)
    points = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in rows],
        dtype=np.float64,
    )
    return points, point_indices


def load_training_point_indices(data_type):
    """Return S point indices in the same order used by inference samples."""
    data_path = simulation_npz_path(data_type)
    if os.path.exists(data_path):
        return np.asarray(np.load(data_path, allow_pickle=True)["s_indices"], dtype=int)

    rows = load_simulation_csv_rows(data_type)
    s_rows = [row for row in rows if row["split"] == "S"]
    s_rows = sorted(s_rows, key=lambda row: int(row["row_order"]))
    return np.asarray([int(row["point_index"]) for row in s_rows], dtype=int)


def load_full_y_field(data_type):
    """Load all S and U response values in their original grid order."""
    data_path = simulation_npz_path(data_type)
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        points = np.asarray(data["all_points"], dtype=np.float64)
        y_all = np.full(len(points), np.nan, dtype=np.float64)
        y_all[np.asarray(data["s_indices"], dtype=int)] = np.asarray(
            data["y_S"], dtype=np.float64
        ).flatten()
        y_all[np.asarray(data["u_indices"], dtype=int)] = np.asarray(
            data["y_U"], dtype=np.float64
        ).flatten()
        if np.isnan(y_all).any():
            raise ValueError(f"Incomplete y field in {data_path}")
        return points, y_all

    rows = sorted(load_simulation_csv_rows(data_type), key=lambda row: int(row["point_index"]))
    points = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in rows],
        dtype=np.float64,
    )
    y_all = np.asarray([float(row["y_obs"]) for row in rows], dtype=np.float64)
    if np.isnan(y_all).any():
        raise ValueError(f"Incomplete y field in {simulation_csv_path(data_type)}")
    return points, y_all


def reshape_regular_grid(points, values):
    """Reshape values from point rows onto a regular x-y grid."""
    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    if len(x_values) * len(y_values) != len(points):
        raise ValueError("The spatial locations do not form a complete regular grid")

    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for point, value in zip(points, values):
        grid[y_lookup[point[1]], x_lookup[point[0]]] = value

    if np.isnan(grid).any():
        raise ValueError("The spatial grid contains missing values")
    return x_values, y_values, grid


def plot_y_heatmaps():
    """Save weak, median, and strong observed y fields as separate figures."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    for data_type in DATA_TYPES:
        fig, ax = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
        points, y_all = load_full_y_field(data_type)
        x_values, y_values, y_grid = reshape_regular_grid(points, y_all)
        if data_type == "strong":
            limit = STRONG_Y_COLOR_LIMIT
        else:
            limit = float(np.max(np.abs(y_grid)))
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = ax.pcolormesh(
            x_values,
            y_values,
            y_grid,
            cmap="Spectral_r",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.set_aspect("equal", adjustable="box")
        colorbar = fig.colorbar(
            image,
            ax=ax,
            shrink=0.88,
            pad=0.03,
        )
        if data_type == "strong":
            colorbar.set_ticks(STRONG_Y_COLOR_TICKS)
        pdf_path = os.path.join(
            VISUALIZATION_DIR,
            f"y_heatmap_{data_type}.pdf",
        )
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {pdf_path}")


def load_nnngp_reference_w(method_dir, data_type="strong", model_results_dir=None):
    """Load true and posterior-mean w values at the requested S locations."""
    if model_results_dir is None:
        model_results_dir = MODEL_RESULTS_DIR
    path = os.path.join(
        model_results_dir,
        method_dir,
        data_type,
        "point_predictions.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["point_set"] == "S"]
    if not rows:
        raise ValueError(f"No S rows found in {path}")
    coordinates = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in rows],
        dtype=np.float64,
    )
    point_indices = np.asarray([int(row["point_index"]) for row in rows], dtype=int)
    true_values = np.asarray([float(row["w_true"]) for row in rows])
    predicted_values = np.asarray([float(row["w_pred"]) for row in rows])
    return coordinates, point_indices, true_values, predicted_values


def load_external_reference_w(method_dir, reference_point_indices):
    """Match NNGP/NNMP full-grid predictions to the NNnGP S locations."""
    path = os.path.join(
        EXTERNAL_RESULTS_DIR,
        method_dir,
        "strong",
        "predictions_w.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_point_index = {int(row["point_index"]): row for row in rows}
    matched_rows = []
    for point_index in reference_point_indices:
        if int(point_index) not in by_point_index:
            raise ValueError(f"Point index {point_index} not found in {path}")
        matched_rows.append(by_point_index[int(point_index)])

    true_values = np.asarray([float(row["true_value"]) for row in matched_rows])
    predicted_values = np.asarray([float(row["pred_mean"]) for row in matched_rows])
    return true_values, predicted_values


def load_nnngp_reference_w_interval(method_dir, data_type="strong", model_results_dir=None):
    """Load true w_S, posterior mean, and posterior 95% interval at S."""
    _, point_indices, true_values, predicted_values = load_nnngp_reference_w(
        method_dir,
        data_type=data_type,
        model_results_dir=model_results_dir,
    )
    if model_results_dir is None:
        model_results_dir = MODEL_RESULTS_DIR

    results_filename = "hmc_results.npz" if method_dir == "hmc" else "vi_results.npz"
    samples_path = os.path.join(
        model_results_dir,
        method_dir,
        data_type,
        results_filename,
    )
    samples = np.load(samples_path, allow_pickle=True)["w_S_samples"]
    lower_by_sample_order, upper_by_sample_order = np.quantile(
        samples,
        [0.025, 0.975],
        axis=0,
    )
    if len(lower_by_sample_order) != len(true_values):
        raise ValueError(
            f"w_S sample count in {samples_path} does not match point_predictions.csv"
        )
    sample_point_indices = load_training_point_indices(data_type)
    sample_column_by_point = {
        int(point_index): column
        for column, point_index in enumerate(sample_point_indices)
    }
    sample_columns = np.asarray(
        [sample_column_by_point[int(point_index)] for point_index in point_indices],
        dtype=int,
    )
    lower = lower_by_sample_order[sample_columns]
    upper = upper_by_sample_order[sample_columns]
    return point_indices, true_values, predicted_values, lower, upper


def load_external_reference_w_interval(method_dir, reference_point_indices):
    """Load external-method true w, posterior mean, and posterior 95% interval."""
    path = os.path.join(
        EXTERNAL_RESULTS_DIR,
        method_dir,
        "strong",
        "predictions_w.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_point_index = {int(row["point_index"]): row for row in rows}
    matched_rows = []
    for point_index in reference_point_indices:
        if int(point_index) not in by_point_index:
            raise ValueError(f"Point index {point_index} not found in {path}")
        matched_rows.append(by_point_index[int(point_index)])

    true_values = np.asarray([float(row["true_value"]) for row in matched_rows])
    predicted_values = np.asarray([float(row["pred_mean"]) for row in matched_rows])
    lower = np.asarray([float(row["pred_q025"]) for row in matched_rows])
    upper = np.asarray([float(row["pred_q975"]) for row in matched_rows])
    return true_values, predicted_values, lower, upper


def draw_reference_w_panel(
    ax,
    title,
    true_values,
    predicted_values,
    plot_min,
    plot_max,
    title_size=20,
    label_size=16,
    tick_size=14,
):
    """Draw one true-vs-predicted w_S panel."""
    ax.scatter(
        predicted_values,
        true_values,
        s=15,
        color="#2878B5",
        alpha=0.65,
        edgecolors="none",
        rasterized=True,
    )
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=1.2,
        color="#7A7A7A",
    )
    ax.set_title(
        title,
        fontsize=title_size,
        fontweight="bold",
        color="black",
    )
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)


def plot_reference_w_comparison():
    """Compare posterior mean and true w_S for available inference methods."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    _, point_indices, mf_true, mf_pred = load_nnngp_reference_w("mean_field")
    _, _, lr_true, lr_pred = load_nnngp_reference_w("lowrank")
    _, _, nf_true, nf_pred = load_nnngp_reference_w("vi_nf")
    nngp_true, nngp_pred = load_external_reference_w("nngp", point_indices)
    nnmp_true, nnmp_pred = load_external_reference_w("nnmp", point_indices)

    panels = [
        ("NNnGP-VI-MF", mf_true, mf_pred),
        ("NNnGP-VI-LR", lr_true, lr_pred),
        ("NNGP", nngp_true, nngp_pred),
        ("NNMP", nnmp_true, nnmp_pred),
        ("NNnGP-VI-NF", nf_true, nf_pred),
    ]
    hmc_path = os.path.join(HMC_MODEL_RESULTS_DIR, "hmc", "strong", "point_predictions.csv")
    if os.path.exists(hmc_path):
        _, _, hmc_true, hmc_pred = load_nnngp_reference_w(
            "hmc",
            model_results_dir=HMC_MODEL_RESULTS_DIR,
        )
        panels.insert(2, ("NNnGP-HMC", hmc_true, hmc_pred))
    else:
        print(f"HMC point predictions not found, skipping HMC panel: {hmc_path}")

    all_values = np.concatenate(
        [np.concatenate([true_values, predicted_values]) for _, true_values, predicted_values in panels]
    )
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    padding = 0.04 * max(value_max - value_min, 1.0)
    plot_min = value_min - padding
    plot_max = value_max + padding

    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (title, true_values, predicted_values) in zip(axes_flat, panels):
        draw_reference_w_panel(
            ax,
            title,
            true_values,
            predicted_values,
            plot_min,
            plot_max,
        )

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    axes_grid = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes_grid[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"$w_S$ (prediction mean)", fontsize=17)
    for ax in axes_grid[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(r"$w_S$ (true)", fontsize=17)

    pdf_path = os.path.join(VISUALIZATION_DIR, "wS_prediction_mean_vs_true.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def plot_reference_w_interval_coverage():
    """Compare w_S 95% interval coverage on the true-vs-predicted scale."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    point_indices, mf_true, mf_pred, mf_lower, mf_upper = load_nnngp_reference_w_interval(
        "mean_field"
    )
    _, lr_true, lr_pred, lr_lower, lr_upper = load_nnngp_reference_w_interval("lowrank")
    _, nf_true, nf_pred, nf_lower, nf_upper = load_nnngp_reference_w_interval("vi_nf")
    nngp_true, nngp_pred, nngp_lower, nngp_upper = load_external_reference_w_interval(
        "nngp",
        point_indices,
    )
    nnmp_true, nnmp_pred, nnmp_lower, nnmp_upper = load_external_reference_w_interval(
        "nnmp",
        point_indices,
    )

    panels = [
        ("NNnGP-VI-MF", mf_true, mf_pred, mf_lower, mf_upper),
        ("NNnGP-VI-LR", lr_true, lr_pred, lr_lower, lr_upper),
        ("NNGP", nngp_true, nngp_pred, nngp_lower, nngp_upper),
        ("NNMP", nnmp_true, nnmp_pred, nnmp_lower, nnmp_upper),
        ("NNnGP-VI-NF", nf_true, nf_pred, nf_lower, nf_upper),
    ]
    hmc_path = os.path.join(HMC_MODEL_RESULTS_DIR, "hmc", "strong", "point_predictions.csv")
    if os.path.exists(hmc_path):
        _, hmc_true, hmc_pred, hmc_lower, hmc_upper = load_nnngp_reference_w_interval(
            "hmc",
            model_results_dir=HMC_MODEL_RESULTS_DIR,
        )
        panels.insert(2, ("NNnGP-HMC", hmc_true, hmc_pred, hmc_lower, hmc_upper))
    else:
        print(f"HMC point predictions not found, skipping HMC panel: {hmc_path}")

    all_values = np.concatenate(
        [
            np.concatenate([true_values, predicted_values, lower, upper])
            for _, true_values, predicted_values, lower, upper in panels
        ]
    )
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))
    padding = 0.04 * max(value_max - value_min, 1.0)
    plot_min = value_min - padding
    plot_max = value_max + padding

    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    covered_color = "#2A9D8F"
    missed_color = "#D55E00"
    for ax, (title, true_values, predicted_values, lower, upper) in zip(axes_flat, panels):
        covered = (lower <= true_values) & (true_values <= upper)
        interval_segments = [
            [(lo, true), (hi, true)]
            for lo, hi, true in zip(lower, upper, true_values)
        ]
        interval_colors = np.where(covered, covered_color, missed_color)
        ax.scatter(
            predicted_values,
            true_values,
            s=7,
            color="#1F2933",
            alpha=0.32,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        cap_height = 0.012 * max(plot_max - plot_min, 1.0)
        cap_segments = [
            [(x, true - cap_height), (x, true + cap_height)]
            for lo, hi, true in zip(lower, upper, true_values)
            for x in (lo, hi)
        ]
        cap_colors = np.repeat(interval_colors, 2)
        ax.add_collection(
            LineCollection(
                interval_segments,
                colors=interval_colors,
                linewidths=1.05,
                alpha=0.92,
                rasterized=True,
                zorder=4,
            )
        )
        ax.add_collection(
            LineCollection(
                cap_segments,
                colors=cap_colors,
                linewidths=0.9,
                alpha=0.92,
                rasterized=True,
                zorder=4,
            )
        )
        ax.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            linestyle="--",
            linewidth=1.2,
            color="#7A7A7A",
            zorder=3,
        )
        coverage = 100.0 * float(np.mean(covered))
        ax.set_title(
            f"{title} ({coverage:.1f}% covered)",
            fontsize=18,
            fontweight="bold",
            color="black",
        )
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.5, alpha=0.25)

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    axes_grid = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes_grid[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"$w_S$ (prediction mean and 95% interval)")
    for ax in axes_grid[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(r"$w_S$ (true)")

    legend_handles = [
        plt.Line2D([0], [0], color=covered_color, linewidth=2.0, label="95% interval covers true"),
        plt.Line2D([0], [0], color=missed_color, linewidth=2.0, label="95% interval misses true"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    pdf_path = os.path.join(VISUALIZATION_DIR, "wS_prediction_interval_coverage.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def load_external_y_prediction(method_dir, data_type="strong"):
    """Load external-method y prediction means and standard deviations."""
    path = os.path.join(
        EXTERNAL_RESULTS_DIR,
        method_dir,
        data_type,
        "predictions_y.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    coordinates = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in rows],
        dtype=np.float64,
    )
    point_indices = np.asarray([int(row["point_index"]) for row in rows], dtype=int)
    prediction_mean = np.asarray([float(row["pred_mean"]) for row in rows])
    prediction_std = np.asarray([float(row["pred_sd"]) for row in rows])
    return coordinates, point_indices, prediction_mean, prediction_std


def load_nnngp_y_prediction(data_type="strong"):
    """Load NNnGP-VI-NF y predictions."""
    path = os.path.join(
        MODEL_RESULTS_DIR,
        "vi_nf",
        data_type,
        "point_predictions.csv",
    )
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    coordinates = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in rows],
        dtype=np.float64,
    )
    point_indices = np.asarray([int(row["point_index"]) for row in rows], dtype=int)
    prediction_mean = np.asarray([float(row["y_pred"]) for row in rows])
    prediction_std = np.asarray([float(row["y_pred_std"]) for row in rows])
    return coordinates, point_indices, prediction_mean, prediction_std


def values_to_grid(point_indices, values, data_type="strong"):
    """Place S and U prediction values on the full regular grid."""
    all_points, all_point_indices = load_simulation_points_and_indices(data_type)
    row_by_point_index = {
        int(point_index): row
        for row, point_index in enumerate(np.asarray(all_point_indices, dtype=int))
    }
    flat_values = np.full(len(all_points), np.nan, dtype=np.float64)
    for point_index, value in zip(np.asarray(point_indices, dtype=int), np.asarray(values)):
        try:
            row = row_by_point_index[int(point_index)]
        except KeyError as exc:
            raise ValueError(
                f"Point index {int(point_index)} not found in simulation grid for {data_type}"
            ) from exc
        flat_values[row] = value
    return reshape_regular_grid(all_points, flat_values)


def plot_y_prediction_heatmaps():
    """Plot ground truth and prediction means for weak, median, and strong data."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    for data_type in DATA_TYPES:
        method_data = (
            ("NNGP", *load_external_y_prediction("nngp", data_type=data_type)),
            ("NNMP", *load_external_y_prediction("nnmp", data_type=data_type)),
            ("NNnGP", *load_nnngp_y_prediction(data_type=data_type)),
        )

        points, y_all = load_full_y_field(data_type)
        x_values, y_values, truth_grid = reshape_regular_grid(points, y_all)
        grids = [("Ground Truth", x_values, y_values, truth_grid)]
        for method_name, coordinates, point_indices, prediction_mean, prediction_std in method_data:
            x_values, y_values, prediction_grid = values_to_grid(
                point_indices,
                prediction_mean,
                data_type=data_type,
            )
            grids.append((method_name, x_values, y_values, prediction_grid))

        if data_type == "strong":
            mean_limit = STRONG_Y_COLOR_LIMIT
        else:
            mean_limit = float(
                np.max([np.nanmax(np.abs(grid[-1])) for grid in grids])
            )
        mean_norm = colors.TwoSlopeNorm(
            vmin=-mean_limit,
            vcenter=0.0,
            vmax=mean_limit,
        )

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(15.2, 4.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.04, wspace=0.04)
        mean_image = None
        for ax, (method_name, x_values, y_values, mean_grid) in zip(axes, grids):
            mean_image = ax.pcolormesh(
                x_values,
                y_values,
                mean_grid,
                cmap="Spectral_r",
                norm=mean_norm,
                shading="auto",
                rasterized=True,
            )
            ax.set_title(
                method_name,
                fontsize=20,
                fontweight="bold",
                color="black",
            )
            ax.set_aspect("equal", adjustable="box")

        for ax in axes:
            ax.set_xlabel("Easting")
        axes[0].set_ylabel("Northing")

        mean_colorbar = fig.colorbar(
            mean_image,
            ax=axes,
            location="right",
            shrink=0.88,
            pad=0.025,
        )
        if data_type == "strong":
            mean_colorbar.set_ticks(STRONG_Y_COLOR_TICKS)

        pdf_path = os.path.join(
            VISUALIZATION_DIR,
            f"y_prediction_mean_heatmaps_{data_type}.pdf",
        )
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.set_defaults(optimize_z=False)
    parser.add_argument(
        "--fixed-z",
        dest="optimize_z",
        action="store_false",
        help="Read fixed-Z tau-sensitivity results. This is the default.",
    )
    parser.add_argument(
        "--optimize-z",
        dest="optimize_z",
        action="store_true",
        help="Read ELBO-optimized-Z tau-sensitivity results from the optimized_z directory.",
    )
    args = parser.parse_args()

    configure_result_paths(results_dir=args.results_dir, optimize_z=args.optimize_z)
    print(f"Simulation input root: {SIMULATION_DIR}")
    print(f"Model result root: {MODEL_RESULTS_DIR}")
    print(f"Visualization output root: {VISUALIZATION_DIR}")

    plot_y_heatmaps()
    plot_reference_w_comparison()
    plot_reference_w_interval_coverage()
    plot_y_prediction_heatmaps()


if __name__ == "__main__":
    main()
