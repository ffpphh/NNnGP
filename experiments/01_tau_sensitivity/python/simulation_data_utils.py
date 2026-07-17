"""Simulation data generators for multiple spatial experiments.

This module keeps experiment-specific latent-field generation separate from
shared spatial design, covariates, train/prediction split, and npz IO.
"""

import os
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, norm
from sklearn.gaussian_process.kernels import Matern


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = PACKAGE_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
RESULTS_DIR = str(EXPERIMENT_DIR / "outputs")
SIMULATION_RESULTS_DIR = str(EXPERIMENT_DIR / "data")


# ==============================================================================
# Manual experiment configuration
# ==============================================================================
# Change values here, then run:
#   python NNnGP/simulation_data_utils.py
#
# RUN_KINDS options:
#   "matern_gp_nnngp" : NNnGP simulation with g(v) drawn from a Matern 3/2 GP
RUN_KINDS = ("matern_gp_nnngp",)
SAVE_PLOTS = True
density_repeats = 200
density_point_indices = None  # None means randomly choose two grid locations.

domain_size = 5.0
k = 500
random_seed = 43
beta = (0.5, 0.5, -0.5)
sigma_epsilon = 0.2
tau_params = (5.0, 0.3)

m = 10
m_tilde = 50
matern_params = (1.0, 0.2)
g_params = (0.0, -2.0, 1.0)

# NNnGP with latent g(v) ~ GP(0, Matern 3/2)
matern_gp_grid_size = 100
matern_gp_include_residual = True


def _ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _make_grid_points(grid_size=100, domain_size=5.0):
    x = np.linspace(0.0, domain_size, grid_size)
    y = np.linspace(0.0, domain_size, grid_size)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _maximin_ordering(points, rng):
    n = len(points)
    order = [int(rng.integers(n))]
    remaining = np.ones(n, dtype=bool)
    remaining[order[0]] = False

    min_dist = cdist(points, points[order[:1]]).ravel()
    for _ in range(n - 1):
        masked = np.where(remaining, min_dist, -np.inf)
        next_idx = int(np.argmax(masked))
        order.append(next_idx)
        remaining[next_idx] = False
        min_dist = np.minimum(min_dist, cdist(points, points[next_idx : next_idx + 1]).ravel())

    return np.asarray(order, dtype=int)


def make_spatial_design(
    k=500,
    grid_size=100,
    domain_size=5.0,
    random_seed=42,
    order_s=True,
):
    """Build common grid, S/U split, and linear covariate matrix."""
    rng = np.random.default_rng(random_seed)
    all_points = _make_grid_points(grid_size=grid_size, domain_size=domain_size)
    n_total = len(all_points)
    if k > n_total:
        raise ValueError(f"k={k} cannot exceed grid_size^2={n_total}.")

    s_indices = rng.choice(n_total, size=k, replace=False)
    S = all_points[s_indices]
    if order_s:
        order = _maximin_ordering(S, rng)
        S = S[order]
        s_indices = s_indices[order]

    u_indices = np.setdiff1d(np.arange(n_total), s_indices)
    U = all_points[u_indices]
    X_all = np.column_stack((np.ones(n_total), all_points[:, 0], all_points[:, 1]))

    return {
        "all_points": all_points,
        "S": S,
        "U": U,
        "s_indices": s_indices,
        "u_indices": u_indices,
        "X_all": X_all,
        "X_S": X_all[s_indices],
        "X_U": X_all[u_indices],
    }


def _save_simulation_npz(save_path, design, w_all, y_all, true_params, extra_arrays=None):
    extra_arrays = {} if extra_arrays is None else dict(extra_arrays)
    s_indices = design["s_indices"]
    u_indices = design["u_indices"]

    _ensure_parent_dir(save_path)
    np.savez_compressed(
        save_path,
        all_points=design["all_points"],
        S=design["S"],
        s_indices=s_indices,
        w_S=w_all[s_indices],
        y_S=y_all[s_indices],
        U=design["U"],
        u_indices=u_indices,
        w_U=w_all[u_indices],
        y_U=y_all[u_indices],
        X_all=design["X_all"],
        X_S=design["X_S"],
        X_U=design["X_U"],
        true_params=true_params,
        **extra_arrays,
    )
    _save_simulation_csv(
        os.path.splitext(save_path)[0] + ".csv",
        design["all_points"],
        s_indices,
        w_all[s_indices],
        y_all[s_indices],
        u_indices,
        w_all[u_indices],
        y_all[u_indices],
        design["X_all"],
    )
    return save_path


def _save_simulation_csv(csv_path, all_points, s_indices, w_S, y_S, u_indices, w_U, y_U, X_all):
    """Save one combined S/U table for R and other external tools."""
    all_points = np.asarray(all_points, dtype=np.float64)
    s_indices = np.asarray(s_indices, dtype=int)
    u_indices = np.asarray(u_indices, dtype=int)
    X_all = np.asarray(X_all, dtype=np.float64)

    w_all = np.full(len(all_points), np.nan, dtype=np.float64)
    y_all = np.full(len(all_points), np.nan, dtype=np.float64)
    split = np.full(len(all_points), "U", dtype=object)
    w_all[s_indices] = np.asarray(w_S, dtype=np.float64)
    y_all[s_indices] = np.asarray(y_S, dtype=np.float64)
    split[s_indices] = "S"
    w_all[u_indices] = np.asarray(w_U, dtype=np.float64)
    y_all[u_indices] = np.asarray(y_U, dtype=np.float64)

    _ensure_parent_dir(csv_path)
    ordered_indices = np.concatenate([s_indices, u_indices])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_order", "point_index", "split", "x", "y", "w", "y_obs", "x0", "x1", "x2"])
        for row_order, idx in enumerate(ordered_indices):
            writer.writerow(
                [
                    row_order,
                    idx,
                    split[idx],
                    f"{all_points[idx, 0]:.17g}",
                    f"{all_points[idx, 1]:.17g}",
                    f"{w_all[idx]:.17g}",
                    f"{y_all[idx]:.17g}",
                    f"{X_all[idx, 0]:.17g}",
                    f"{X_all[idx, 1]:.17g}",
                    f"{X_all[idx, 2]:.17g}",
                ]
            )
    return csv_path


def _save_simulation_csv_from_npz(npz_path):
    data = load_simulation_data(npz_path)
    csv_path = os.path.splitext(npz_path)[0] + ".csv"
    return _save_simulation_csv(
        csv_path,
        data["all_points"],
        data["s_indices"],
        data["w_S"],
        data["y_S"],
        data["u_indices"],
        data["w_U"],
        data["y_U"],
        data["X_all"],
    )


def _red_high_value_norm(values, vmin=None, vmax=None):
    values = np.asarray(values)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmin < 0.0 < vmax:
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _scatter_field(
    points,
    values,
    title,
    save_path,
    s=20,
    marked_points=None,
    marked_labels=None,
):
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    is_regular_grid = len(x_unique) * len(y_unique) == len(points)

    if is_regular_grid:
        order = np.lexsort((points[:, 0], points[:, 1]))
        grid_values = values[order].reshape(len(y_unique), len(x_unique))
        mesh = ax.pcolormesh(
            x_unique,
            y_unique,
            grid_values,
            shading="nearest",
            cmap="Spectral_r",
            norm=_red_high_value_norm(values),
            rasterized=True,
        )
        fig.colorbar(mesh, ax=ax, label="Value")
    else:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=values,
            s=s,
            marker="s",
            linewidths=0,
            cmap="Spectral_r",
            norm=_red_high_value_norm(values),
            rasterized=True,
        )
        fig.colorbar(scatter, ax=ax, label="Value")

    if marked_points is not None:
        marked_points = np.asarray(marked_points, dtype=np.float64)
        if marked_points.ndim == 1:
            marked_points = marked_points.reshape(1, -1)
        if marked_labels is None:
            marked_labels = [str(i + 1) for i in range(len(marked_points))]

        ax.scatter(
            marked_points[:, 0],
            marked_points[:, 1],
            marker="X",
            s=240,
            c="black",
            edgecolors="white",
            linewidths=1.8,
            zorder=5,
        )
        ax.scatter(
            marked_points[:, 0],
            marked_points[:, 1],
            marker="x",
            s=260,
            c="white",
            linewidths=1.4,
            zorder=6,
        )
        for label, (x, y) in zip(marked_labels, marked_points):
            ax.text(
                x + 0.018,
                y + 0.018,
                str(label),
                color="black",
                fontsize=13,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.85,
                    boxstyle="round,pad=0.2",
                ),
                zorder=7,
            )

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect("equal", adjustable="box")
    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(values, title, xlabel, save_path, bins=40):
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(np.asarray(values, dtype=np.float64), bins=bins, density=True, alpha=0.78)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_two_point_density(
    draws,
    point_indices,
    point_locations,
    repeats,
    save_path,
    title_prefix="Fixed-location",
    variable_label="y",
):
    """Plot two empirical pointwise densities with matched Gaussian references."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    palette = ["#1f77b4", "#d62728"]

    for j, ax in enumerate(axes):
        samples = np.asarray(draws[:, j], dtype=np.float64)
        mean = float(np.mean(samples))
        std = max(float(np.std(samples, ddof=1)), 1e-8)
        xs = np.linspace(np.min(samples) - 0.5 * std, np.max(samples) + 0.5 * std, 300)
        kde = gaussian_kde(samples)

        ax.hist(samples, bins=24, density=True, alpha=0.30, color=palette[j], edgecolor="white")
        ax.plot(xs, kde(xs), color=palette[j], linewidth=2.2, label="Empirical KDE")
        ax.plot(
            xs,
            norm.pdf(xs, loc=mean, scale=std),
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Matched Gaussian",
        )
        ax.axvline(mean, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title(
            f"Point {j + 1}: index={point_indices[j]}, "
            f"loc=({point_locations[j, 0]:.3f}, {point_locations[j, 1]:.3f})"
        )
        ax.set_xlabel(f"Repeated {variable_label} value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(f"{title_prefix} {variable_label} density over {repeats} repeated simulations", fontsize=13)
    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_two_point_joint_density(draws, point_indices, point_locations, repeats, save_path, variable_label="y"):
    """Plot empirical joint density of two repeated fixed-location values."""
    draws = np.asarray(draws, dtype=np.float64)
    y1 = draws[:, 0]
    y2 = draws[:, 1]
    corr = float(np.corrcoef(y1, y2)[0, 1])

    pad1 = max(0.15 * (np.max(y1) - np.min(y1)), 1e-6)
    pad2 = max(0.15 * (np.max(y2) - np.min(y2)), 1e-6)
    x_grid = np.linspace(np.min(y1) - pad1, np.max(y1) + pad1, 160)
    y_grid = np.linspace(np.min(y2) - pad2, np.max(y2) + pad2, 160)
    xx, yy = np.meshgrid(x_grid, y_grid)

    kde = gaussian_kde(draws.T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    contour = ax.contourf(xx, yy, zz, levels=24, cmap="Spectral_r")
    ax.contour(xx, yy, zz, levels=8, colors="black", linewidths=0.6, alpha=0.45)
    ax.axvline(np.mean(y1), color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(np.mean(y2), color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel(
        f"Point 1 {variable_label}, index={point_indices[0]}, "
        f"loc=({point_locations[0, 0]:.3f}, {point_locations[0, 1]:.3f})"
    )
    ax.set_ylabel(
        f"Point 2 {variable_label}, index={point_indices[1]}, "
        f"loc=({point_locations[1, 0]:.3f}, {point_locations[1, 1]:.3f})"
    )
    ax.set_title(f"Joint density of two fixed-location {variable_label} values ({repeats} repeats), corr={corr:.3f}")
    fig.colorbar(contour, ax=ax, label="Joint density")

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _choose_two_grid_points(all_points, point_indices, random_seed):
    rng = np.random.default_rng(random_seed)
    if point_indices is None:
        point_indices = np.sort(rng.choice(np.arange(len(all_points)), size=2, replace=False))
    else:
        point_indices = np.asarray(point_indices, dtype=int)
        if point_indices.shape != (2,):
            raise ValueError("density_point_indices must contain exactly two indices.")
    return point_indices, np.asarray(all_points, dtype=np.float64)[point_indices]


def _simulate_w_at_locations_with_g(locations, S, w_S, true_params, g_fn, rng):
    locations = np.asarray(locations, dtype=np.float64)
    m_value = int(true_params["m"])
    theta_tau1, theta_tau2 = true_params["tau_params"]
    theta_g1, theta_g2, _ = true_params["g_params"]
    matern_kernel = _matern_kernel_from_params(true_params["matern_params"])

    s_lookup = {tuple(point): idx for idx, point in enumerate(np.asarray(S, dtype=np.float64))}
    w_values = np.zeros(len(locations), dtype=np.float64)
    for j, loc in enumerate(locations):
        s_idx = s_lookup.get(tuple(loc))
        if s_idx is not None:
            w_values[j] = w_S[s_idx]
            continue

        distances = cdist(loc.reshape(1, -1), S)[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        S_N = S[neighbor_indices]
        w_N = w_S[neighbor_indices]
        dists = distances[neighbor_indices]

        C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m_value)
        C_uN = matern_kernel(loc.reshape(1, -1), S_N)[0]
        inv_C_NN = np.linalg.inv(C_NN)
        h_u = C_uN @ inv_C_NN @ w_N
        F_u = matern_kernel(loc.reshape(1, -1))[0, 0] - C_uN @ inv_C_NN @ C_uN.T
        F_u = max(float(F_u), 1e-6)
        tau_u = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
        v_u = sqrt_lambda * w_N
        w_values[j] = rng.normal(h_u + tau_u * g_fn(v_u), np.sqrt(F_u))

    return w_values


def _simulate_matern_gp_two_point_repeats(data, point_locations, repeats, random_seed):
    S = np.asarray(data["S"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    rho = float(true_params["g_params"][2])
    include_residual = bool(true_params.get("g_include_residual", True))
    R_Z = _g_matern32_correlation(Z, Z, rho=rho) + 1e-6 * np.eye(len(Z))
    R_Z_inv = np.linalg.inv(R_Z)
    rng = np.random.default_rng(random_seed)

    w_draws = np.zeros((repeats, 2), dtype=np.float64)
    y_draws = np.zeros((repeats, 2), dtype=np.float64)
    X = np.column_stack((np.ones(len(point_locations)), point_locations[:, 0], point_locations[:, 1]))
    beta_value = np.asarray(true_params["beta"], dtype=np.float64)
    sigma_eps = float(true_params["sigma_epsilon"])

    for r in range(repeats):
        g_Z = rng.multivariate_normal(np.zeros(len(Z)), R_Z)
        w_S, _ = _simulate_nnngp_field_with_matern_gp_g(
            S,
            np.empty((0, S.shape[1]), dtype=np.float64),
            int(true_params["m"]),
            true_params["matern_params"],
            true_params["tau_params"],
            true_params["g_params"],
            Z,
            g_Z,
            R_Z_inv,
            random_seed=int(random_seed + r + 1),
            include_residual=include_residual,
        )
        g_fn = lambda v, g_Z_value=g_Z: _sample_g_matern32_at_v(
            v,
            Z,
            g_Z_value,
            R_Z_inv,
            rho,
            rng,
            include_residual=include_residual,
        )
        w_draws[r] = _simulate_w_at_locations_with_g(point_locations, S, w_S, true_params, g_fn, rng)
        y_draws[r] = X @ beta_value + w_draws[r] + rng.normal(0.0, sigma_eps, 2)

    return w_draws, y_draws


def plot_two_point_density_repeats(
    data_path,
    output_dir,
    repeats=200,
    point_indices=None,
    random_seed=2026,
):
    """Randomly choose two grid locations and save repeated-simulation density plots."""
    data = load_simulation_data(data_path)
    os.makedirs(output_dir, exist_ok=True)

    all_points = np.asarray(data["all_points"], dtype=np.float64)
    selected_indices, point_locations = _choose_two_grid_points(all_points, point_indices, random_seed)
    simulation = data["true_params"].get("simulation", "matern_gp_nnngp")
    if simulation != "matern_gp_nnngp":
        raise ValueError(f"Unsupported simulation type after cleanup: {simulation!r}")

    w_draws, y_draws = _simulate_matern_gp_two_point_repeats(
        data,
        point_locations,
        repeats,
        random_seed=random_seed,
    )
    title_prefix = "Matern-GP NNnGP fixed-location"

    _plot_two_point_density(
        w_draws,
        selected_indices,
        point_locations,
        repeats,
        os.path.join(output_dir, "two_point_w_density.png"),
        title_prefix=title_prefix,
        variable_label="w",
    )
    _plot_two_point_joint_density(
        w_draws,
        selected_indices,
        point_locations,
        repeats,
        os.path.join(output_dir, "two_point_w_joint_density.png"),
        variable_label="w",
    )
    _plot_two_point_density(
        y_draws,
        selected_indices,
        point_locations,
        repeats,
        os.path.join(output_dir, "two_point_y_density.png"),
        title_prefix=title_prefix,
        variable_label="y",
    )
    _plot_two_point_joint_density(
        y_draws,
        selected_indices,
        point_locations,
        repeats,
        os.path.join(output_dir, "two_point_y_joint_density.png"),
        variable_label="y",
    )
    np.savez_compressed(
        os.path.join(output_dir, "two_point_density_repeats.npz"),
        w_draws=w_draws,
        y_draws=y_draws,
        point_indices=selected_indices,
        point_locations=point_locations,
        repeats=repeats,
    )

    print("两点重复模拟密度图已保存")
    print(f"  采样点索引: {selected_indices.tolist()}")
    print(f"  采样点坐标: {point_locations.tolist()}")
    print(f"  输出目录: {output_dir}")
    return output_dir


def plot_simulation_outputs(data_path, output_dir=None, repeats=200, point_indices=None, random_seed=2026):
    """Save field maps and two-point repeated-simulation density diagnostics."""
    data = load_simulation_data(data_path)
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(data_path))[0]
        output_dir = os.path.join(SIMULATION_RESULTS_DIR, stem)
    os.makedirs(output_dir, exist_ok=True)

    all_points = np.asarray(data["all_points"], dtype=np.float64)
    s_indices = np.asarray(data["s_indices"], dtype=int)
    u_indices = np.asarray(data["u_indices"], dtype=int)
    w_all = np.empty(len(all_points), dtype=np.float64)
    y_all = np.empty(len(all_points), dtype=np.float64)
    w_all[s_indices] = np.asarray(data["w_S"], dtype=np.float64)
    y_all[s_indices] = np.asarray(data["y_S"], dtype=np.float64)
    w_all[u_indices] = np.asarray(data["w_U"], dtype=np.float64)
    y_all[u_indices] = np.asarray(data["y_U"], dtype=np.float64)
    selected_indices, selected_locations = _choose_two_grid_points(all_points, point_indices, random_seed)

    _scatter_field(
        all_points,
        w_all,
        "Latent field w(s)",
        os.path.join(output_dir, "latent_w_field.png"),
        marked_points=selected_locations,
        marked_labels=["1", "2"],
    )
    _scatter_field(all_points, y_all, "Observed response y(s)", os.path.join(output_dir, "observed_y_field.png"))
    if "Z1_all" in data:
        _scatter_field(all_points, data["Z1_all"], "Matern GP Z1(s)", os.path.join(output_dir, "z1_field.png"))
    if "Z2_all" in data:
        _scatter_field(all_points, data["Z2_all"], "Matern GP Z2(s)", os.path.join(output_dir, "z2_field.png"))
    if "signal_all" in data:
        _scatter_field(
            all_points,
            data["signal_all"],
            "Nonlinear signal exp(lambda Z1) Z2",
            os.path.join(output_dir, "signal_field.png"),
        )
    _scatter_field(
        np.asarray(data["S"], dtype=np.float64),
        np.arange(len(data["S"])),
        "Reference locations S",
        os.path.join(output_dir, "reference_locations.png"),
        s=28,
    )
    plot_two_point_density_repeats(
        data_path,
        output_dir,
        repeats=repeats,
        point_indices=selected_indices,
        random_seed=random_seed,
    )
    print(f"模拟可视化结果已保存至: {output_dir}")
    return output_dir


def load_simulation_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    out = {key: data[key] for key in data.files}
    if isinstance(out.get("true_params"), np.ndarray):
        out["true_params"] = out["true_params"].item()
    return out


def _matern_kernel_from_params(matern_params):
    sigma_f, length_scale = matern_params
    return (float(sigma_f) ** 2) * Matern(length_scale=float(length_scale), nu=1.5)


def _g_matern32_correlation(X, Y=None, rho=1.0):
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    dists = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1) + 1e-12)
    scaled = np.sqrt(3.0) * dists / (float(rho) + 1e-12)
    return (1.0 + scaled) * np.exp(-scaled)


def _sample_g_matern32_at_v(v, Z, g_Z, R_Z_inv, rho, rng, include_residual=True):
    v = np.asarray(v, dtype=np.float64).reshape(1, -1)
    R_vZ = _g_matern32_correlation(v, Z, rho=rho)[0]
    mean = float(R_vZ @ R_Z_inv @ g_Z)
    if not include_residual:
        return mean
    var = 1.0 - float(R_vZ @ R_Z_inv @ R_vZ.T)
    var = max(var, 1e-8)
    return float(rng.normal(mean, np.sqrt(var)))


def _select_inducing_Z_from_nngp(S, matern_kernel, m_value, m_tilde_value, g_params_value, random_seed):
    """Select inducing inputs in neighbor-value space."""
    from data_utils import generate_nngp_ws

    rng = np.random.default_rng(random_seed)
    w_S_nngp = generate_nngp_ws(S, matern_kernel, m_value, random_seed=random_seed)
    v_samples = []
    for i in range(m_value, len(S)):
        distances = cdist(S[i : i + 1], S[:i])[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        w_N = w_S_nngp[neighbor_indices]
        dists = cdist(S[i : i + 1], S[neighbor_indices])[0]
        sqrt_lambda = np.sqrt(np.exp(g_params_value[0] + g_params_value[1] * dists))
        v_samples.append(sqrt_lambda * w_N)

    v_samples = np.asarray(v_samples, dtype=np.float64)
    if m_tilde_value <= len(v_samples):
        return v_samples[rng.choice(len(v_samples), m_tilde_value, replace=False)]
    return np.vstack(
        [
            v_samples,
            rng.normal(0.0, np.std(v_samples, axis=0), size=(m_tilde_value - len(v_samples), m_value)),
        ]
    )


def _simulate_nnngp_field_with_matern_gp_g(
    S,
    U,
    m_value,
    matern_params_value,
    tau_params_value,
    g_params_value,
    Z,
    g_Z,
    R_Z_inv,
    random_seed,
    include_residual=True,
):
    rng = np.random.default_rng(random_seed)
    matern_kernel = _matern_kernel_from_params(matern_params_value)
    theta_tau1, theta_tau2 = tau_params_value
    theta_g1, theta_g2, rho = g_params_value

    def sample_g(v):
        return _sample_g_matern32_at_v(
            v,
            Z,
            g_Z,
            R_Z_inv,
            rho,
            rng,
            include_residual=include_residual,
        )

    w_S = np.zeros(len(S), dtype=np.float64)
    if m_value > 0:
        cov0 = matern_kernel(S[:m_value]) + 1e-4 * np.eye(m_value)
        w_S[:m_value] = rng.multivariate_normal(np.zeros(m_value), cov0)

    for i in range(m_value, len(S)):
        distances = cdist(S[i : i + 1], S[:i])[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        S_N = S[neighbor_indices]
        w_N = w_S[neighbor_indices]
        dists = distances[neighbor_indices]

        C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m_value)
        C_iN = matern_kernel(S[i : i + 1], S_N)[0]
        inv_C_NN = np.linalg.inv(C_NN)
        h_i = C_iN @ inv_C_NN @ w_N
        F_i = matern_kernel(S[i : i + 1])[0, 0] - C_iN @ inv_C_NN @ C_iN.T
        F_i = max(float(F_i), 1e-6)

        tau_i = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
        v_i = sqrt_lambda * w_N
        w_S[i] = rng.normal(h_i + tau_i * sample_g(v_i), np.sqrt(F_i))

    w_U = np.zeros(len(U), dtype=np.float64)
    for j, u in enumerate(U):
        distances = cdist(u.reshape(1, -1), S)[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        S_N = S[neighbor_indices]
        w_N = w_S[neighbor_indices]
        dists = distances[neighbor_indices]

        C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m_value)
        C_uN = matern_kernel(u.reshape(1, -1), S_N)[0]
        inv_C_NN = np.linalg.inv(C_NN)
        h_u = C_uN @ inv_C_NN @ w_N
        F_u = matern_kernel(u.reshape(1, -1))[0, 0] - C_uN @ inv_C_NN @ C_uN.T
        F_u = max(float(F_u), 1e-6)

        tau_u = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
        v_u = sqrt_lambda * w_N
        w_U[j] = rng.normal(h_u + tau_u * sample_g(v_u), np.sqrt(F_u))

    return w_S, w_U


def generate_matern_gp_nnngp_data(
    k=500,
    m=10,
    m_tilde=50,
    grid_size=100,
    domain_size=5.0,
    matern_params=(1.0, 0.2),
    tau_params=(0.0, 1.0),
    g_params=(0.0, -2.0, 1.0),
    beta=(0.5, 1.0, -1.0),
    sigma_epsilon=0.1,
    random_seed=42,
    save_path=os.path.join(SIMULATION_RESULTS_DIR, "matern_gp_nnngp", "matern_gp_nnngp_data.npz"),
    plot_dir=None,
    save_plots=True,
    repeats=200,
    point_indices=None,
    include_residual=True,
):
    """Generate NNnGP data with a fixed latent g(v) drawn from a Matern 3/2 GP.

    The spatial field generation follows the same NNnGP recursion as the other
    simulations. The nonlinear function is represented by inducing inputs Z in
    v-space, with g_Z ~ N(0, R_Z) and location-wise Matern 3/2 conditional draws.
    """
    design = make_spatial_design(
        k=k,
        grid_size=grid_size,
        domain_size=domain_size,
        random_seed=random_seed,
        order_s=True,
    )
    S = design["S"]
    U = design["U"]
    matern_kernel = _matern_kernel_from_params(matern_params)
    Z = _select_inducing_Z_from_nngp(S, matern_kernel, m, m_tilde, g_params, random_seed)

    rho = float(g_params[2])
    R_Z = _g_matern32_correlation(Z, Z, rho=rho) + 1e-6 * np.eye(len(Z))
    R_Z_inv = np.linalg.inv(R_Z)
    rng = np.random.default_rng(random_seed + 77)
    g_Z = rng.multivariate_normal(np.zeros(len(Z)), R_Z)

    w_S, w_U = _simulate_nnngp_field_with_matern_gp_g(
        S,
        U,
        m,
        matern_params,
        tau_params,
        g_params,
        Z,
        g_Z,
        R_Z_inv,
        random_seed=random_seed,
        include_residual=include_residual,
    )

    w_all = np.zeros(len(design["all_points"]), dtype=np.float64)
    w_all[design["s_indices"]] = w_S
    w_all[design["u_indices"]] = w_U
    beta = np.asarray(beta, dtype=np.float64)
    obs_rng = np.random.default_rng(random_seed + 1)
    y_all = design["X_all"] @ beta + w_all + obs_rng.normal(0.0, sigma_epsilon, len(w_all))

    true_params = {
        "simulation": "matern_gp_nnngp",
        "matern_params": matern_params,
        "tau_params": tau_params,
        "g_params": g_params,
        "g_type": "matern32_gp",
        "g_include_residual": bool(include_residual),
        "g_standardized": False,
        "g_standardization": (0.0, 1.0),
        "beta": beta,
        "tau2": float(sigma_epsilon**2),
        "sigma_epsilon": float(sigma_epsilon),
        "m": int(m),
        "m_tilde": int(m_tilde),
        "domain_size": float(domain_size),
    }

    path = _save_simulation_npz(
        save_path,
        design,
        w_all,
        y_all,
        true_params,
        extra_arrays={"Z": Z, "g_Z": g_Z},
    )
    print(f"Matern-GP-NNnGP 模拟数据已保存至: {path}")
    print(
        f"参考点数={k}, 预测点数={len(U)}, beta={beta}, tau2={sigma_epsilon**2:.6g}, "
        f"g_rho={rho:.6g}, include_residual={include_residual}"
    )
    if save_plots:
        plot_simulation_outputs(
            path,
            output_dir=plot_dir,
            repeats=repeats,
            point_indices=point_indices,
            random_seed=random_seed + 2026,
        )
    return path


def generate_simulation(kind, **kwargs):
    if kind in {"matern_gp", "matern_gp_nnngp", "nnngp_matern_gp"}:
        return generate_matern_gp_nnngp_data(**kwargs)
    raise ValueError(f"Unknown simulation kind: {kind}")


def _generate_one_from_manual_config(run_kind):
    output_dir = os.path.join(SIMULATION_RESULTS_DIR, run_kind)
    os.makedirs(output_dir, exist_ok=True)

    if run_kind != "matern_gp_nnngp":
        raise ValueError(f"Unknown run_kind: {run_kind}")

    name = "matern_gp_nnngp_data.npz"
    save_path = os.path.join(output_dir, name)
    plot_dir = os.path.join(output_dir, "plots")

    print("=" * 70)
    print("模拟数据生成")
    print("=" * 70)
    print(f"当前模拟类型: {run_kind}")
    print(f"输出目录: {output_dir}")

    generate_matern_gp_nnngp_data(
        k=k,
        m=m,
        m_tilde=m_tilde,
        grid_size=matern_gp_grid_size,
        domain_size=domain_size,
        matern_params=matern_params,
        tau_params=tau_params,
        g_params=g_params,
        beta=beta,
        sigma_epsilon=sigma_epsilon,
        random_seed=random_seed,
        save_path=save_path,
        plot_dir=plot_dir,
        save_plots=SAVE_PLOTS,
        repeats=density_repeats,
        point_indices=density_point_indices,
        include_residual=matern_gp_include_residual,
    )


def main():
    for run_kind in RUN_KINDS:
        _generate_one_from_manual_config(run_kind)


if __name__ == "__main__":
    main()
