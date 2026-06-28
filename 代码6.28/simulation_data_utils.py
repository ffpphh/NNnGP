"""Simulation data generators for multiple spatial experiments.

This module keeps experiment-specific latent-field generation separate from
shared spatial design, covariates, train/prediction split, and npz IO.
"""

import os
import csv

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, gaussian_kde, norm, t
from sklearn.gaussian_process.kernels import Matern

from data_utils import (
    generate_nnngp_data,
    _precompute_fixed_s_terms,
    _simulate_fixed_s_state,
    _simulate_w_at_locations,
    _simulate_y_at_locations,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SIMULATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "simulation_outputs")


# ==============================================================================
# Manual experiment configuration
# ==============================================================================
# Change values here, then run:
#   python NNnGP/simulation_data_utils.py
#
# RUN_KINDS options:
#   "tanh_nnngp" : existing NNnGP simulation with parametric tanh g(v)
#   "mlp_nnngp"  : NNnGP simulation with a neural-network g(v)
#   "matern_gp_nnngp" : NNnGP simulation with g(v) drawn from a Matern 3/2 GP
#   "t_copula"  : Student-t copula field with Gaussian margins
# Use both by default. To run only one simulation, set e.g.
# RUN_KINDS = ("tanh_nnngp", "mlp_nnngp", "t_copula","matern_gp_nnngp")
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

# Simulation 1: tanh NNnGP parameters
tanh_grid_size = 100
m = 10
m_tilde = 50
matern_params = (1.0, 0.2)
g_params = (0.0, -2.0, 1.0)
# Tanh-style g(v): amplitude * (z + gamma * tanh(z)),
# z = slope * mean(v) + bias. The linear z term makes g unbounded.
parametric_g_params = (0.3, 0.5, 0.0, 0.5)  # (amplitude, slope, bias, gamma)
tanh_standardize_g = True  # Standardize raw tanh-style g(v) to mean 0 and std 1.

# Simulation 3: NNnGP with neural-network g(v)
# For hidden/output layers:
#   W_l ~ N(0, mlp_weight_scale / sqrt(fan_in_l))
#   b_l ~ N(0, mlp_bias_scale)
# With mlp_hidden_dims=(16, 16) and m=10:
#   W1: 10x16, b1: 16; W2: 16x16, b2: 16; W3: 16x1, b3: 1.
mlp_grid_size = 100
mlp_hidden_dims = (16, 16)
mlp_weight_scale = 0.35
mlp_bias_scale = 0.05

# Simulation 4: NNnGP with latent g(v) ~ GP(0, Matern 3/2)
matern_gp_grid_size = 100
matern_gp_include_residual = True

# Simulation 2: t-copula Gaussian-margin parameters
# This simulation uses a dense Cholesky factorization on the full grid.
# Keep t_copula_grid_size moderate unless you replace it with an approximation.
# Its field maps look coarser than tanh_grid_size=100 because the default grid is 50x50.
t_copula_grid_size = 50
t_copula_tau2 = float(sigma_epsilon**2)
student_t_df = 4.0
correlation_range = 0.5
max_dense_points = 4000


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


def _simulate_tanh_two_point_repeats(data, point_locations, repeats, random_seed):
    S = np.asarray(data["S"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    m_value = int(true_params["m"])
    if len(S) <= m_value:
        raise ValueError("Need len(S) > m to simulate NNnGP reference locations.")

    rng = np.random.default_rng(random_seed)
    precomputed = _precompute_fixed_s_terms(S, m_value, true_params)
    w_draws = np.zeros((repeats, 2), dtype=np.float64)
    y_draws = np.zeros((repeats, 2), dtype=np.float64)
    for r in range(repeats):
        w_S, g_Z, R_Z_inv, _, g_kernel = _simulate_fixed_s_state(S, Z, true_params, precomputed, rng)
        w_draws[r] = _simulate_w_at_locations(
            point_locations,
            S,
            w_S,
            Z,
            true_params,
            precomputed,
            g_Z,
            R_Z_inv,
            g_kernel,
            rng,
        )
        y_draws[r] = _simulate_y_at_locations(
            point_locations,
            S,
            w_S,
            Z,
            true_params,
            precomputed,
            g_Z,
            R_Z_inv,
            g_kernel,
            rng,
        )
    return w_draws, y_draws


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


def _simulate_mlp_two_point_repeats(data, point_locations, repeats, random_seed):
    S = np.asarray(data["S"], dtype=np.float64)
    true_params = data["true_params"]
    mlp_params_value = true_params["mlp_params"]
    g_fn = lambda v: mlp_g(v, mlp_params_value)
    rng = np.random.default_rng(random_seed)

    w_draws = np.zeros((repeats, 2), dtype=np.float64)
    y_draws = np.zeros((repeats, 2), dtype=np.float64)
    X = np.column_stack((np.ones(len(point_locations)), point_locations[:, 0], point_locations[:, 1]))
    beta_value = np.asarray(true_params["beta"], dtype=np.float64)
    sigma_eps = float(true_params["sigma_epsilon"])

    for r in range(repeats):
        w_S, _ = _simulate_nnngp_field_with_g(
            S,
            np.empty((0, S.shape[1]), dtype=np.float64),
            int(true_params["m"]),
            true_params["matern_params"],
            true_params["tau_params"],
            true_params["g_params"],
            g_fn,
            int(random_seed + r + 1),
        )
        w_draws[r] = _simulate_w_at_locations_with_g(point_locations, S, w_S, true_params, g_fn, rng)
        y_draws[r] = X @ beta_value + w_draws[r] + rng.normal(0.0, sigma_eps, 2)

    return w_draws, y_draws


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


def _simulate_t_copula_two_point_repeats(true_params, point_locations, repeats, random_seed):
    rng = np.random.default_rng(random_seed)
    df = float(true_params["df"])
    tau2_value = float(true_params["tau2"])
    beta_value = np.asarray(true_params["beta"], dtype=np.float64)
    R2 = exponential_correlation(
        point_locations,
        correlation_range=float(true_params["correlation_range"]),
        nugget=1e-8,
    )
    L2 = np.linalg.cholesky(R2)
    X = np.column_stack((np.ones(len(point_locations)), point_locations[:, 0], point_locations[:, 1]))

    w_draws = np.zeros((repeats, 2), dtype=np.float64)
    y_draws = np.zeros((repeats, 2), dtype=np.float64)
    for r in range(repeats):
        z = L2 @ rng.normal(size=2)
        scale = np.sqrt(rng.chisquare(df) / df)
        w0 = z / scale
        u = np.clip(t.cdf(w0, df=df), np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        w = norm.ppf(u)
        w_draws[r] = w
        y_draws[r] = X @ beta_value + w + rng.normal(0.0, np.sqrt(tau2_value), 2)
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
    simulation = data["true_params"].get("simulation", "tanh_nnngp")

    if simulation == "t_copula_gaussian_margin":
        w_draws, y_draws = _simulate_t_copula_two_point_repeats(
            data["true_params"],
            point_locations,
            repeats,
            random_seed=random_seed,
        )
        title_prefix = "t-copula fixed-location"
    elif simulation == "mlp_nnngp":
        w_draws, y_draws = _simulate_mlp_two_point_repeats(
            data,
            point_locations,
            repeats,
            random_seed=random_seed,
        )
        title_prefix = "MLP NNnGP fixed-location"
    elif simulation == "matern_gp_nnngp":
        w_draws, y_draws = _simulate_matern_gp_two_point_repeats(
            data,
            point_locations,
            repeats,
            random_seed=random_seed,
        )
        title_prefix = "Matern-GP NNnGP fixed-location"
    else:
        w_draws, y_draws = _simulate_tanh_two_point_repeats(
            data,
            point_locations,
            repeats,
            random_seed=random_seed,
        )
        title_prefix = "tanh NNnGP fixed-location"

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
    if "w0_all" in data:
        _scatter_field(all_points, data["w0_all"], "Student-t process w0(s)", os.path.join(output_dir, "student_t_w0_field.png"))

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


def make_mlp_params(input_dim, hidden_dims=(16, 16), weight_scale=0.35, bias_scale=0.05, random_seed=42):
    """Create fixed true MLP weights for g(v): R^m -> R."""
    rng = np.random.default_rng(random_seed)
    dims = (int(input_dim),) + tuple(int(x) for x in hidden_dims) + (1,)
    weights = []
    biases = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        weights.append(rng.normal(0.0, weight_scale / np.sqrt(max(fan_in, 1)), size=(fan_in, fan_out)))
        biases.append(rng.normal(0.0, bias_scale, size=(fan_out,)))
    return {"weights": weights, "biases": biases, "hidden_dims": tuple(hidden_dims)}


def mlp_g(v, mlp_params):
    """Two-layer-style neural network g(v), with linear output and unbounded range."""
    h = np.asarray(v, dtype=np.float64)
    weights = mlp_params["weights"]
    biases = mlp_params["biases"]
    for W, b in zip(weights[:-1], biases[:-1]):
        h = np.maximum(h @ W + b, 0.0)
    out = h @ weights[-1] + biases[-1]
    return float(np.ravel(out)[0])


def _estimate_g_standardization(S, m_value, matern_params_value, g_params_value, raw_g_fn, random_seed):
    """Estimate mean/std of a deterministic g(v) over NNGP warmup v samples."""
    from data_utils import generate_nngp_ws

    S = np.asarray(S, dtype=np.float64)
    m_value = int(m_value)
    matern_kernel = _matern_kernel_from_params(matern_params_value)
    w_S_nngp = generate_nngp_ws(S, matern_kernel, m_value, random_seed=random_seed)
    theta_g1, theta_g2, _ = g_params_value

    g_values = []
    for i in range(m_value, len(S)):
        distances = cdist(S[i : i + 1], S[:i])[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        w_N = w_S_nngp[neighbor_indices]
        dists = distances[neighbor_indices]
        sqrt_lambda = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
        g_values.append(raw_g_fn(sqrt_lambda * w_N))

    g_values = np.asarray(g_values, dtype=np.float64)
    if len(g_values) == 0:
        return 0.0, 1.0
    g_mean = float(np.mean(g_values))
    g_std = float(np.std(g_values, ddof=0))
    if g_std <= 1e-12:
        return 0.0, 1.0
    return g_mean, g_std


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
    """Match the tanh simulation's diagnostic Z construction."""
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


def _simulate_nnngp_field_with_g(
    S,
    U,
    m_value,
    matern_params_value,
    tau_params_value,
    g_params_value,
    g_fn,
    random_seed,
):
    rng = np.random.default_rng(random_seed)
    matern_kernel = _matern_kernel_from_params(matern_params_value)

    w_S = np.zeros(len(S), dtype=np.float64)
    if m_value > 0:
        cov0 = matern_kernel(S[:m_value]) + 1e-4 * np.eye(m_value)
        w_S[:m_value] = rng.multivariate_normal(np.zeros(m_value), cov0)

    theta_tau1, theta_tau2 = tau_params_value
    theta_g1, theta_g2, _ = g_params_value

    for i in range(m_value, len(S)):
        distances = cdist(S[i : i + 1], S[:i])[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        S_N = S[neighbor_indices]
        w_N = w_S[neighbor_indices]
        dists = distances[neighbor_indices]

        C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m_value)
        C_iN = matern_kernel(S[i : i + 1], S_N)[0]
        B_i = C_iN @ np.linalg.inv(C_NN)
        h_i = B_i @ w_N
        F_i = matern_kernel(S[i : i + 1])[0, 0] - C_iN @ np.linalg.inv(C_NN) @ C_iN.T
        F_i = max(float(F_i), 1e-6)
        tau_i = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
        v_i = sqrt_lambda * w_N
        w_S[i] = rng.normal(h_i + tau_i * g_fn(v_i), np.sqrt(F_i))

    w_U = np.zeros(len(U), dtype=np.float64)
    for j, u in enumerate(U):
        distances = cdist(u.reshape(1, -1), S)[0]
        neighbor_indices = np.argsort(distances)[:m_value]
        S_N = S[neighbor_indices]
        w_N = w_S[neighbor_indices]
        dists = distances[neighbor_indices]

        C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m_value)
        C_uN = matern_kernel(u.reshape(1, -1), S_N)[0]
        B_u = C_uN @ np.linalg.inv(C_NN)
        h_u = B_u @ w_N
        F_u = matern_kernel(u.reshape(1, -1))[0, 0] - C_uN @ np.linalg.inv(C_NN) @ C_uN.T
        F_u = max(float(F_u), 1e-6)
        tau_u = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
        v_u = sqrt_lambda * w_N
        w_U[j] = rng.normal(h_u + tau_u * g_fn(v_u), np.sqrt(F_u))

    return w_S, w_U


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


def exponential_correlation(points, correlation_range=0.5, nugget=1e-8):
    """R_ij = exp(-||s_i - s_j|| / correlation_range)."""
    if correlation_range <= 0:
        raise ValueError("correlation_range must be positive.")
    dists = cdist(points, points)
    R = np.exp(-dists / correlation_range)
    if nugget > 0:
        R = R + nugget * np.eye(len(points))
    return R


def sample_t_copula_gaussian_marginal_field(
    points,
    df=4.0,
    correlation_range=0.5,
    random_seed=42,
    nugget=1e-8,
):
    """Sample w(s)=Phi^{-1}(T_df(w0(s))) from a t-copula field."""
    if df <= 0:
        raise ValueError("df must be positive.")

    rng = np.random.default_rng(random_seed)
    R = exponential_correlation(points, correlation_range=correlation_range, nugget=nugget)
    L = np.linalg.cholesky(R)

    z = L @ rng.normal(size=len(points))
    scale = np.sqrt(chi2.rvs(df, random_state=rng) / df)
    w0 = z / scale

    u = t.cdf(w0, df=df)
    u = np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    w = norm.ppf(u)
    return w, w0, R


def generate_t_copula_gaussian_margin_data(
    k=500,
    grid_size=50,
    domain_size=5.0,
    beta=(0.5, 1.0, -1.0),
    tau2=0.01,
    df=4.0,
    correlation_range=0.5,
    random_seed=42,
    save_path=os.path.join(SIMULATION_RESULTS_DIR, "t_copula", "t_copula_gaussian_margin_data.npz"),
    max_dense_points=4000,
    plot_dir=None,
    save_plots=True,
    repeats=200,
    point_indices=None,
):
    """
    Generate the second simulation:
      w0(s) follows a Student-t spatial process with exponential correlation R;
      w(s)=Phi^{-1}(T_df(w0(s))) has N(0,1) margins and t-copula dependence;
      y(s)=x(s)^T beta + w(s) + epsilon(s), epsilon~N(0,tau2).

    The t-copula field uses a dense Cholesky factorization of all grid points.
    Keep grid_size moderate, or raise max_dense_points deliberately.
    """
    design = make_spatial_design(
        k=k,
        grid_size=grid_size,
        domain_size=domain_size,
        random_seed=random_seed,
        order_s=False,
    )
    n_total = len(design["all_points"])
    if n_total > max_dense_points:
        raise ValueError(
            f"t-copula simulation needs a dense {n_total}x{n_total} correlation matrix. "
            f"Use a smaller grid_size or raise max_dense_points."
        )

    w_all, w0_all, R = sample_t_copula_gaussian_marginal_field(
        design["all_points"],
        df=df,
        correlation_range=correlation_range,
        random_seed=random_seed,
    )

    rng = np.random.default_rng(random_seed + 1)
    beta = np.asarray(beta, dtype=np.float64)
    sigma_epsilon = float(np.sqrt(tau2))
    y_all = (
        design["X_all"] @ beta
        + w_all
        + rng.normal(0.0, sigma_epsilon, n_total)
    )

    true_params = {
        "simulation": "t_copula_gaussian_margin",
        "correlation": "exponential",
        "correlation_range": float(correlation_range),
        "df": float(df),
        "beta": beta,
        "tau2": float(tau2),
        "sigma_epsilon": sigma_epsilon,
        "k": int(k),
        "grid_size": int(grid_size),
        "domain_size": float(domain_size),
    }

    extra_arrays = {
        "w0_all": w0_all,
        "w0_S": w0_all[design["s_indices"]],
        "w0_U": w0_all[design["u_indices"]],
    }
    if n_total <= 2000:
        extra_arrays["R"] = R

    path = _save_simulation_npz(save_path, design, w_all, y_all, true_params, extra_arrays)
    print(f"t-copula 高斯边缘模拟数据已保存至: {path}")
    print(f"总网格点数={n_total}, 参考点数={k}, beta={beta}, tau2={tau2}")
    if save_plots:
        plot_simulation_outputs(
            path,
            output_dir=plot_dir,
            repeats=repeats,
            point_indices=point_indices,
            random_seed=random_seed + 2026,
        )
    return path


def generate_tanh_nnngp_simulation(
    save_path=os.path.join(SIMULATION_RESULTS_DIR, "tanh_nnngp", "tanh_nnngp_data.npz"),
    plot_dir=None,
    save_plots=True,
    repeats=200,
    point_indices=None,
    **kwargs,
):
    """Wrapper for the existing NNnGP simulation with parametric tanh g(v)."""
    path = generate_nnngp_data(save_path=save_path, **kwargs)
    csv_path = _save_simulation_csv_from_npz(path)
    print(f"Tanh-NNnGP CSV 数据已保存至: {csv_path}")
    if save_plots:
        plot_simulation_outputs(
            path,
            output_dir=plot_dir,
            repeats=repeats,
            point_indices=point_indices,
            random_seed=int(kwargs.get("random_seed", 42)) + 2026,
        )
    return path


def generate_mlp_nnngp_data(
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
    mlp_hidden_dims=(16, 16),
    mlp_weight_scale=0.35,
    mlp_bias_scale=0.05,
    random_seed=42,
    save_path=os.path.join(SIMULATION_RESULTS_DIR, "mlp_nnngp", "mlp_nnngp_data.npz"),
    plot_dir=None,
    save_plots=True,
    repeats=200,
    point_indices=None,
    standardize_g=True,
):
    """Generate NNnGP data with a fixed neural-network g(v): R^m -> R."""
    design = make_spatial_design(
        k=k,
        grid_size=grid_size,
        domain_size=domain_size,
        random_seed=random_seed,
        order_s=True,
    )
    S = design["S"]
    U = design["U"]
    mlp_params_value = make_mlp_params(
        input_dim=m,
        hidden_dims=mlp_hidden_dims,
        weight_scale=mlp_weight_scale,
        bias_scale=mlp_bias_scale,
        random_seed=random_seed + 77,
    )
    raw_g_fn = lambda v: mlp_g(v, mlp_params_value)
    if standardize_g:
        g_mean, g_std = _estimate_g_standardization(
            S,
            m,
            matern_params,
            g_params,
            raw_g_fn,
            random_seed=random_seed,
        )
    else:
        g_mean, g_std = 0.0, 1.0
    g_fn = lambda v: (raw_g_fn(v) - g_mean) / g_std
    w_S, w_U = _simulate_nnngp_field_with_g(
        S,
        U,
        m,
        matern_params,
        tau_params,
        g_params,
        g_fn,
        random_seed=random_seed,
    )

    w_all = np.zeros(len(design["all_points"]), dtype=np.float64)
    w_all[design["s_indices"]] = w_S
    w_all[design["u_indices"]] = w_U
    beta = np.asarray(beta, dtype=np.float64)
    rng = np.random.default_rng(random_seed + 1)
    y_all = design["X_all"] @ beta + w_all + rng.normal(0.0, sigma_epsilon, len(w_all))

    matern_kernel = _matern_kernel_from_params(matern_params)
    Z = _select_inducing_Z_from_nngp(S, matern_kernel, m, m_tilde, g_params, random_seed)
    true_params = {
        "simulation": "mlp_nnngp",
        "matern_params": matern_params,
        "tau_params": tau_params,
        "g_params": g_params,
        "g_type": "mlp",
        "g_standardized": bool(standardize_g),
        "g_standardization": (float(g_mean), float(g_std)),
        "mlp_params": mlp_params_value,
        "mlp_hidden_dims": tuple(mlp_hidden_dims),
        "mlp_weight_scale": float(mlp_weight_scale),
        "mlp_bias_scale": float(mlp_bias_scale),
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
        extra_arrays={"Z": Z},
    )
    print(f"MLP-NNnGP 模拟数据已保存至: {path}")
    print(
        f"参考点数={k}, 预测点数={len(U)}, beta={beta}, tau2={sigma_epsilon**2:.6g}, "
        f"g_standardization=({g_mean:.6g}, {g_std:.6g})"
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
    if kind in {"tanh", "tanh_nnngp", "nnngp_tanh"}:
        return generate_tanh_nnngp_simulation(**kwargs)
    if kind in {"mlp", "mlp_nnngp", "nnngp_mlp"}:
        return generate_mlp_nnngp_data(**kwargs)
    if kind in {"matern_gp", "matern_gp_nnngp", "nnngp_matern_gp"}:
        return generate_matern_gp_nnngp_data(**kwargs)
    if kind in {"t_copula", "t_copula_gaussian_margin"}:
        return generate_t_copula_gaussian_margin_data(**kwargs)
    raise ValueError(f"Unknown simulation kind: {kind}")


def _generate_one_from_manual_config(run_kind):
    output_dir = os.path.join(SIMULATION_RESULTS_DIR, run_kind)
    os.makedirs(output_dir, exist_ok=True)

    output_names = {
        "tanh_nnngp": "tanh_nnngp_data.npz",
        "mlp_nnngp": "mlp_nnngp_data.npz",
        "matern_gp_nnngp": "matern_gp_nnngp_data.npz",
        "t_copula": "t_copula_gaussian_margin_data.npz",
    }
    name = output_names.get(run_kind, f"{run_kind}_data.npz")
    save_path = os.path.join(output_dir, name)
    plot_dir = os.path.join(output_dir, "plots")

    print("=" * 70)
    print("模拟数据生成")
    print("=" * 70)
    print(f"当前模拟类型: {run_kind}")
    print(f"输出目录: {output_dir}")

    if run_kind == "tanh_nnngp":
        generate_tanh_nnngp_simulation(
            k=k,
            m=m,
            m_tilde=m_tilde,
            grid_size=tanh_grid_size,
            domain_size=domain_size,
            matern_params=matern_params,
            tau_params=tau_params,
            g_params=g_params,
            parametric_g_params=parametric_g_params,
            beta=beta,
            sigma_epsilon=sigma_epsilon,
            random_seed=random_seed,
            standardize_g=tanh_standardize_g,
            save_path=save_path,
            plot_dir=plot_dir,
            save_plots=SAVE_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
        )
    elif run_kind == "mlp_nnngp":
        generate_mlp_nnngp_data(
            k=k,
            m=m,
            m_tilde=m_tilde,
            grid_size=mlp_grid_size,
            domain_size=domain_size,
            matern_params=matern_params,
            tau_params=tau_params,
            g_params=g_params,
            beta=beta,
            sigma_epsilon=sigma_epsilon,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_weight_scale=mlp_weight_scale,
            mlp_bias_scale=mlp_bias_scale,
            random_seed=random_seed,
            save_path=save_path,
            plot_dir=plot_dir,
            save_plots=SAVE_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
        )
    elif run_kind == "matern_gp_nnngp":
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
    elif run_kind == "t_copula":
        generate_t_copula_gaussian_margin_data(
            k=k,
            grid_size=t_copula_grid_size,
            domain_size=domain_size,
            beta=beta,
            tau2=t_copula_tau2,
            df=student_t_df,
            correlation_range=correlation_range,
            random_seed=random_seed,
            save_path=save_path,
            max_dense_points=max_dense_points,
            plot_dir=plot_dir,
            save_plots=SAVE_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
        )
    else:
        raise ValueError(f"Unknown run_kind: {run_kind}")


def main():
    for run_kind in RUN_KINDS:
        _generate_one_from_manual_config(run_kind)


if __name__ == "__main__":
    main()
