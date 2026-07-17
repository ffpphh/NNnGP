"""Visualization and reporting helpers for NNnGP inference experiments."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

STD_COLORMAP = colors.LinearSegmentedColormap.from_list(
    "std_green_to_red",
    ["#c7e9b4", "#fff7bc", "#fdae61", "#d7191c"],
)


def _red_high_value_norm(values, vmin=None, vmax=None):
    """Use the same low-purple/green/high-red scale as data_utils.py."""
    values = np.asarray(values)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmin < 0.0 < vmax:
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _grid_edges(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _regular_grid_for_plot(points, values):
    """Return pcolormesh inputs when points form a complete regular grid."""
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).flatten()
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(values):
        return None

    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    if len(x_values) * len(y_values) != len(points):
        return None

    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for point, value in zip(points, values):
        grid[y_lookup[point[1]], x_lookup[point[0]]] = value
    if np.isnan(grid).any():
        return None
    return _grid_edges(x_values), _grid_edges(y_values), grid


def _plot_spatial_values(ax, points, values, cmap, norm, scatter_kwargs):
    grid_data = _regular_grid_for_plot(points, values)
    if grid_data is not None:
        x_edges, y_edges, value_grid = grid_data
        return ax.imshow(
            value_grid,
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
    return ax.scatter(
        points[:, 0],
        points[:, 1],
        c=values,
        cmap=cmap,
        norm=norm,
        **scatter_kwargs,
    )


def evaluate_predictions(y_true, y_pred_mean, y_pred_std, label=""):
    y_true = np.asarray(y_true).flatten()
    y_pred_mean = np.asarray(y_pred_mean).flatten()
    y_pred_std = np.asarray(y_pred_std).flatten()
    rmse = float(np.sqrt(np.mean((y_true - y_pred_mean) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred_mean)))
    lower = y_pred_mean - 1.96 * y_pred_std
    upper = y_pred_mean + 1.96 * y_pred_std
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    prefix = f"{label} " if label else ""
    print(f"\n{prefix}预测性能评估:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  95% CI Coverage: {coverage:.4f} (理想值: 0.95)")
    return {"rmse": rmse, "mae": mae, "coverage": coverage}


def print_parameter_comparison(
    true_params,
    est_params,
    title="Empirical-Bayes 参数估计结果（VI训练后参数）",
    est_label="est",
):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    rows = [
        ("sigma_f", true_params["matern_params"][0], est_params["matern_params"][0]),
        ("length_scale", true_params["matern_params"][1], est_params["matern_params"][1]),
        ("theta_tau1", true_params["tau_params"][0], est_params["tau_params"][0]),
        ("theta_tau2", true_params["tau_params"][1], est_params["tau_params"][1]),
        ("theta_g1", true_params["g_params"][0], est_params["g_params"][0]),
        ("theta_g2", true_params["g_params"][1], est_params["g_params"][1]),
    ]
    if len(true_params["g_params"]) > 2 and len(est_params["g_params"]) > 2:
        rows.append(("rho", true_params["g_params"][2], est_params["g_params"][2]))
    rows.append(("sigma_epsilon", true_params["sigma_epsilon"], est_params["sigma_epsilon"]))

    for name, truth, est in rows:
        print(f"  {name:15s} true={float(truth): .6f}   {est_label}={float(est): .6f}")
    print(f"  beta true={np.asarray(true_params['beta'])}")
    print(f"  beta {est_label:<4s}={np.asarray(est_params['beta'])}")
    print("=" * 70)


def compare_posteriors(
    hmc_samples,
    vi_samples,
    title="Posterior Comparison",
    save_path=None,
    vi_label="VI",
):
    hmc_samples = np.asarray(hmc_samples)
    vi_samples = np.asarray(vi_samples)
    min_samples = min(len(hmc_samples), len(vi_samples))
    hmc_samples = hmc_samples[:min_samples]
    vi_samples = vi_samples[:min_samples]
    n_dims = min(hmc_samples.shape[1], vi_samples.shape[1])
    rng = np.random.default_rng(123)
    indices = rng.choice(n_dims, min(12, n_dims), replace=False)

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        plt.hist(hmc_samples[:, idx], bins=30, alpha=0.5, label="HMC", density=True)
        plt.hist(vi_samples[:, idx], bins=30, alpha=0.5, label=vi_label, density=True)
        plt.title(f"w_{idx}")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(title, fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ws_true_vs_inferred(w_true, w_samples, method_name="VI", save_path=None):
    w_true = np.asarray(w_true).flatten()
    w_mean = np.mean(np.asarray(w_samples), axis=0).flatten()
    rmse = float(np.sqrt(np.mean((w_true - w_mean) ** 2)))
    corr = float(np.corrcoef(w_true, w_mean)[0, 1])
    lo = min(np.min(w_true), np.min(w_mean))
    hi = max(np.max(w_true), np.max(w_mean))

    plt.figure(figsize=(6, 6))
    plt.scatter(w_true, w_mean, s=12, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    plt.xlabel("True $w_S$")
    plt.ylabel(f"{method_name} posterior mean of $w_S$")
    plt.title(f"{method_name}: True vs Inferred $w_S$\nRMSE={rmse:.4f}, Corr={corr:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return rmse, corr


def plot_training_y_true_vs_fitted(
    y_true,
    X_S,
    w_S_samples,
    beta_samples,
    method_name="VI",
    save_path=None,
):
    y_true = np.asarray(y_true).flatten()
    X_S = np.asarray(X_S)
    w_mean = np.mean(np.asarray(w_S_samples), axis=0)
    beta_mean = np.mean(np.asarray(beta_samples), axis=0)
    y_fitted = X_S @ beta_mean + w_mean
    rmse = float(np.sqrt(np.mean((y_true - y_fitted) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_fitted)))
    corr = float(np.corrcoef(y_true, y_fitted)[0, 1])
    lo = min(np.min(y_true), np.min(y_fitted))
    hi = max(np.max(y_true), np.max(y_fitted))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_fitted, s=14, alpha=0.65)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    plt.xlabel("True $y_S$")
    plt.ylabel(f"{method_name} fitted $\\hat y_S$")
    plt.title(f"{method_name}: True vs Fitted $y_S$\nRMSE={rmse:.4f}, MAE={mae:.4f}, Corr={corr:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return rmse, mae, corr


def plot_prediction_maps(S, y_S, U, y_U, pred_mean, pred_std, title, save_path=None, include_S=True):
    """Plot true values, predicted mean, and predicted standard deviation."""
    if include_S:
        points = np.vstack([S, U])
        true_full = np.concatenate([y_S, y_U])
        pred_full = np.concatenate([y_S, pred_mean])
        std_points = points
        std_full = np.concatenate([np.zeros_like(y_S), pred_std])
    else:
        points = U
        true_full = y_U
        pred_full = pred_mean
        std_points = U
        std_full = pred_std

    vmin = float(np.nanmin(true_full))
    vmax = float(np.nanmax(true_full))
    mean_norm = _red_high_value_norm(true_full, vmin=vmin, vmax=vmax)
    std_norm = colors.Normalize(vmin=float(np.nanmin(std_full)), vmax=float(np.nanmax(std_full)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    kwargs = dict(s=18, marker="s", linewidths=0, rasterized=True)
    im0 = _plot_spatial_values(axes[0], points, true_full, "Spectral_r", mean_norm, kwargs)
    axes[0].set_title("True Values")
    plt.colorbar(im0, ax=axes[0])
    im1 = _plot_spatial_values(axes[1], points, pred_full, "Spectral_r", mean_norm, kwargs)
    axes[1].set_title("Predicted Mean")
    plt.colorbar(im1, ax=axes[1])
    im2 = _plot_spatial_values(axes[2], std_points, std_full, STD_COLORMAP, std_norm, kwargs)
    axes[2].set_title("Predicted Standard Deviation")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title, fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_true_vs_draw_map(U, true_values, draw_values, title, save_path=None, true_title="True Values", draw_title="Posterior Predictive Draw"):
    """Plot true values against one posterior predictive draw on prediction locations."""
    U = np.asarray(U, dtype=np.float64)
    true_values = np.asarray(true_values).flatten()
    draw_values = np.asarray(draw_values).flatten()

    vmin = min(float(np.nanmin(true_values)), float(np.nanmin(draw_values)))
    vmax = max(float(np.nanmax(true_values)), float(np.nanmax(draw_values)))
    norm = _red_high_value_norm(np.concatenate([true_values, draw_values]), vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    kwargs = dict(s=18, marker="s", linewidths=0, rasterized=True)
    im0 = axes[0].scatter(U[:, 0], U[:, 1], c=true_values, cmap="Spectral_r", norm=norm, **kwargs)
    axes[0].set_title(true_title)
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].scatter(U[:, 0], U[:, 1], c=draw_values, cmap="Spectral_r", norm=norm, **kwargs)
    axes[1].set_title(draw_title)
    plt.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        x_span = max(float(np.nanmax(U[:, 0]) - np.nanmin(U[:, 0])), 1e-12)
        y_span = max(float(np.nanmax(U[:, 1]) - np.nanmin(U[:, 1])), 1e-12)
        ax.set_xlim(float(np.nanmin(U[:, 0])) - 0.02 * x_span, float(np.nanmax(U[:, 0])) + 0.02 * x_span)
        ax.set_ylim(float(np.nanmin(U[:, 1])) - 0.02 * y_span, float(np.nanmax(U[:, 1])) + 0.02 * y_span)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(y_true, pred_mean, method_name="VI", save_path=None):
    y_true = np.asarray(y_true).flatten()
    pred_mean = np.asarray(pred_mean).flatten()
    rmse = np.sqrt(np.mean((y_true - pred_mean) ** 2))
    lo = min(np.min(y_true), np.min(pred_mean))
    hi = max(np.max(y_true), np.max(pred_mean))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, pred_mean, alpha=0.5, s=5)
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.title(f"{method_name}: True vs Predicted on U\nRMSE={rmse:.4f}")
    plt.xlabel("True y_U")
    plt.ylabel("Predicted mean")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_elbo_loss_curve(losses, title, save_path, window=None, start_iter=0):
    losses = np.asarray(losses, dtype=np.float64).reshape(-1)
    x = np.arange(len(losses))
    mask = x >= start_iter
    if window is None:
        window = max(100, len(losses) // 100)

    plt.figure(figsize=(10, 5))
    plt.plot(x[mask], losses[mask], alpha=0.35, label="Raw loss")
    if len(losses) > window:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        smooth_x = np.arange(window - 1, len(losses))
        smooth_mask = smooth_x >= start_iter
        plt.plot(smooth_x[smooth_mask], smoothed[smooth_mask], linewidth=2, label=f"Smoothed ({window})")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO loss")
    plt.xlim(left=start_iter)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_prediction_performance_comparison(
    y_true,
    hmc_pred_mean,
    eb_pred_mean,
    hmc_metrics,
    eb_metrics,
    save_path,
):
    y_true = np.asarray(y_true).flatten()
    hmc_pred_mean = np.asarray(hmc_pred_mean).flatten()
    eb_pred_mean = np.asarray(eb_pred_mean).flatten()
    lo = float(np.min(y_true))
    hi = float(np.max(y_true))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, hmc_pred_mean, alpha=0.5, s=5)
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.title(f'HMC: True vs Predicted\nRMSE={hmc_metrics["rmse"]:.4f}')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_true, eb_pred_mean, alpha=0.5, s=5)
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.title(f'EB-VI: True vs Predicted\nRMSE={eb_metrics["rmse"]:.4f}')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.grid(True, alpha=0.3)
    plt.suptitle("Prediction Performance Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_saved_elbo_curve(label, source, output, start_iter=500):
    data = np.load(source, allow_pickle=True)
    if "losses" not in data.files:
        raise KeyError(f"{source} does not contain a 'losses' array")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plot_elbo_loss_curve(
        data["losses"],
        title=f"{label} ELBO Loss Curve from Iteration {start_iter}",
        save_path=output,
        start_iter=start_iter,
    )
