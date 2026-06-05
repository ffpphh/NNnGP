"""Simulation data generators for multiple spatial experiments.

This module keeps experiment-specific latent-field generation separate from
shared spatial design, covariates, train/prediction split, and npz IO.
"""

import os

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, gaussian_kde, norm, t

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
#   "t_copula"  : Student-t copula field with Gaussian margins
# Use both by default. To run only one simulation, set e.g.
# RUN_KINDS = ("tanh_nnngp",)
RUN_KINDS = ("tanh_nnngp", "t_copula")
SAVE_PLOTS = True
density_repeats = 200
density_point_indices = None  # None means randomly choose two grid locations.

domain_size = 5.0
k = 500
random_seed = 42
beta = (0.5, 0.5, -0.5)

# Simulation 1: tanh NNnGP parameters
tanh_grid_size = 100
m = 10
m_tilde = 50
matern_params = (1.0, 0.2)
tau_params = (1.0, 0.5)
g_params = (0.0, -2.0, 1.0)
parametric_g_params = (0.3, 0.5, 0.0)
sigma_epsilon = 0.1

# Simulation 2: t-copula Gaussian-margin parameters
# This simulation uses a dense Cholesky factorization on the full grid.
# Keep t_copula_grid_size moderate unless you replace it with an approximation.
# Its field maps look coarser than tanh_grid_size=100 because the default grid is 50x50.
t_copula_grid_size = 50
sigma_w = 1.0
tau2 = 0.01
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
    return save_path


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


def _simulate_t_copula_two_point_repeats(true_params, point_locations, repeats, random_seed):
    rng = np.random.default_rng(random_seed)
    df = float(true_params["df"])
    sigma_w_value = float(true_params["sigma_w"])
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
        y_draws[r] = X @ beta_value + sigma_w_value * w + rng.normal(0.0, np.sqrt(tau2_value), 2)
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
    sigma_w=1.0,
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
      y(s)=x(s)^T beta + sigma_w w(s) + epsilon(s), epsilon~N(0,tau2).

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
        + float(sigma_w) * w_all
        + rng.normal(0.0, sigma_epsilon, n_total)
    )

    true_params = {
        "simulation": "t_copula_gaussian_margin",
        "correlation": "exponential",
        "correlation_range": float(correlation_range),
        "df": float(df),
        "beta": beta,
        "sigma_w": float(sigma_w),
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
    print(f"总网格点数={n_total}, 参考点数={k}, beta={beta}, sigma_w={sigma_w}, tau2={tau2}")
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
    if save_plots:
        plot_simulation_outputs(
            path,
            output_dir=plot_dir,
            repeats=repeats,
            point_indices=point_indices,
            random_seed=int(kwargs.get("random_seed", 42)) + 2026,
        )
    return path


def generate_simulation(kind, **kwargs):
    if kind in {"tanh", "tanh_nnngp", "nnngp_tanh"}:
        return generate_tanh_nnngp_simulation(**kwargs)
    if kind in {"t_copula", "t_copula_gaussian_margin"}:
        return generate_t_copula_gaussian_margin_data(**kwargs)
    raise ValueError(f"Unknown simulation kind: {kind}")


def _generate_one_from_manual_config(run_kind):
    output_dir = os.path.join(SIMULATION_RESULTS_DIR, run_kind)
    os.makedirs(output_dir, exist_ok=True)

    name = "tanh_nnngp_data.npz" if run_kind == "tanh_nnngp" else "t_copula_gaussian_margin_data.npz"
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
            save_path=save_path,
            plot_dir=plot_dir,
            save_plots=SAVE_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
        )
    elif run_kind == "t_copula":
        generate_t_copula_gaussian_margin_data(
            k=k,
            grid_size=t_copula_grid_size,
            domain_size=domain_size,
            beta=beta,
            sigma_w=sigma_w,
            tau2=tau2,
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
