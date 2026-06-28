"""Create standalone visualizations for the VI tau-sensitivity experiment."""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results", "vi_tau_sensitivity")
RESULTS_DIR = DEFAULT_RESULTS_DIR
SIMULATION_DIR = os.path.join(
    RESULTS_DIR,
    "simulation_outputs",
    "matern_gp_nnngp",
)
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
STD_COLORMAP = colors.LinearSegmentedColormap.from_list(
    "Spectral_r_green_to_orange",
    plt.get_cmap("Spectral_r")(np.linspace(0.28, 0.82, 256)),
)


def configure_result_paths(results_dir=DEFAULT_RESULTS_DIR, optimize_z=False):
    """Configure input/output directories for fixed-Z or optimized-Z results."""
    global RESULTS_DIR, SIMULATION_DIR, VISUALIZATION_DIR, MODEL_RESULTS_DIR, HMC_MODEL_RESULTS_DIR

    RESULTS_DIR = os.path.abspath(results_dir)
    inference_root = os.path.join(RESULTS_DIR, "optimized_z") if optimize_z else RESULTS_DIR
    SIMULATION_DIR = os.path.join(
        RESULTS_DIR,
        "simulation_outputs",
        "matern_gp_nnngp",
    )
    VISUALIZATION_DIR = os.path.join(inference_root, "visualizations")
    MODEL_RESULTS_DIR = os.path.join(inference_root, "matern_gp_nnngp")
    HMC_MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, "matern_gp_nnngp")


def load_full_y_field(data_type):
    """Load all S and U response values in their original grid order."""
    data_path = os.path.join(
        SIMULATION_DIR,
        data_type,
        "matern_gp_nnngp_data.npz",
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Simulation data not found: {data_path}")

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
        BASE_DIR,
        "results",
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
            color="#C82423",
        )
        ax.set_title(
            title,
            fontsize=20,
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
            ax.set_xlabel(r"$w_S$ (prediction mean)")
    for ax in axes_grid[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(r"$w_S$ (true)")

    pdf_path = os.path.join(VISUALIZATION_DIR, "wS_prediction_mean_vs_true.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def load_external_y_prediction(method_dir):
    """Load strong-data U prediction means and standard deviations."""
    path = os.path.join(
        BASE_DIR,
        "results",
        method_dir,
        "strong",
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


def load_nnngp_y_prediction():
    """Load strong-data NNnGP-VI-NF U predictions."""
    path = os.path.join(
        MODEL_RESULTS_DIR,
        "vi_nf",
        "strong",
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


def values_to_grid(point_indices, values):
    """Place S and U prediction values on the full regular grid."""
    data_path = os.path.join(
        SIMULATION_DIR,
        "strong",
        "matern_gp_nnngp_data.npz",
    )
    data = np.load(data_path, allow_pickle=True)
    all_points = np.asarray(data["all_points"], dtype=np.float64)
    x_values = np.unique(all_points[:, 0])
    y_values = np.unique(all_points[:, 1])
    flat_values = np.full(len(all_points), np.nan, dtype=np.float64)
    flat_values[np.asarray(point_indices, dtype=int)] = np.asarray(values)
    grid = flat_values.reshape(len(y_values), len(x_values))
    return x_values, y_values, grid


def plot_y_prediction_heatmaps():
    """Plot strong-data S/U prediction means and standard deviations."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    method_data = (
        ("NNGP", *load_external_y_prediction("nngp")),
        ("NNMP", *load_external_y_prediction("nnmp")),
        ("NNnGP", *load_nnngp_y_prediction()),
    )

    grids = []
    for method_name, coordinates, point_indices, prediction_mean, prediction_std in method_data:
        x_values, y_values, mean_grid = values_to_grid(
            point_indices,
            prediction_mean,
        )
        _, _, std_grid = values_to_grid(point_indices, prediction_std)
        grids.append((method_name, x_values, y_values, mean_grid, std_grid))

    all_std_values = np.concatenate(
        [item[4][np.isfinite(item[4])] for item in grids]
    )
    mean_limit = STRONG_Y_COLOR_LIMIT
    std_limit = float(np.quantile(all_std_values, 0.98))
    mean_norm = colors.TwoSlopeNorm(
        vmin=-mean_limit,
        vcenter=0.0,
        vmax=mean_limit,
    )
    std_norm = colors.Normalize(vmin=0.0, vmax=std_limit)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    mean_image = None
    std_image = None
    for column, (method_name, x_values, y_values, mean_grid, std_grid) in enumerate(grids):
        mean_image = axes[0, column].pcolormesh(
            x_values,
            y_values,
            mean_grid,
            cmap="Spectral_r",
            norm=mean_norm,
            shading="auto",
            rasterized=True,
        )
        std_image = axes[1, column].pcolormesh(
            x_values,
            y_values,
            std_grid,
            cmap=STD_COLORMAP,
            norm=std_norm,
            shading="auto",
            rasterized=True,
        )
        axes[0, column].set_title(
            method_name,
            fontsize=20,
            fontweight="bold",
            color="black",
        )

    for ax in axes.flat:
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[-1, :]:
        ax.set_xlabel("Easting")
    for ax in axes[:, 0]:
        ax.set_ylabel("Northing")

    mean_colorbar = fig.colorbar(
        mean_image,
        ax=axes[0, :],
        location="right",
        shrink=0.88,
        pad=0.02,
    )
    mean_colorbar.set_ticks(STRONG_Y_COLOR_TICKS)
    fig.colorbar(
        std_image,
        ax=axes[1, :],
        location="right",
        shrink=0.88,
        pad=0.02,
    )

    pdf_path = os.path.join(
        VISUALIZATION_DIR,
        "y_prediction_mean_std_heatmaps.pdf",
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
    plot_y_prediction_heatmaps()


if __name__ == "__main__":
    main()
