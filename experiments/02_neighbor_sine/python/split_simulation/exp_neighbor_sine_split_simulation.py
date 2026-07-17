"""Archived reference-split neighbor-sine simulation.

This file preserves the earlier data-generating design:

1. Build a 50 x 50 regular grid on [0, 5]^2.
2. Sample 500 reference locations S uniformly without replacement.
3. Reorder those 500 S locations by maximin ordering.
4. Generate y_S recursively over the ordered S locations.
5. Generate the remaining 2000 prediction locations U conditionally on all S.

This archive is intentionally separate from exp_neighbor_sine_simulation.py so
the current active experiment can keep evolving without losing this version.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Matern


Array = np.ndarray

GRID_SIZE = 50
DOMAIN_SIZE = 5.0
NUM_REFERENCE_POINTS = 500
RANDOM_SEED = 43

M = 10
SINE_NEIGHBORS = 2
SINE_AMPLITUDE = 1.0
SINE_FREQUENCY = 4.0
MATERN_PARAMS = (1.0, 0.2)
JITTER = 1e-6

NNNGP_FIT_M_TILDE = 50
DEFAULT_TAU_PARAMS = (0.0, 1.0)
DEFAULT_G_PARAMS = (0.0, -2.0, 1.0)

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "split"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_NPZ = OUTPUT_DIR / "exp_neighbor_sine_reference_split_data.npz"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_grid_points(grid_size: int = GRID_SIZE, domain_size: float = DOMAIN_SIZE) -> Array:
    axis = np.linspace(0.0, domain_size, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack((xx.ravel(), yy.ravel()))


def maximin_ordering(points: Array, rng: np.random.Generator) -> Array:
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
    num_reference_points: int = NUM_REFERENCE_POINTS,
    grid_size: int = GRID_SIZE,
    domain_size: float = DOMAIN_SIZE,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Array]:
    rng = np.random.default_rng(random_seed)
    all_points = make_grid_points(grid_size=grid_size, domain_size=domain_size)
    n_total = len(all_points)
    if num_reference_points > n_total:
        raise ValueError(
            f"num_reference_points={num_reference_points} cannot exceed grid_size^2={n_total}."
        )

    s_indices = rng.choice(n_total, size=num_reference_points, replace=False)
    order = maximin_ordering(all_points[s_indices], rng)
    s_indices = s_indices[order]
    u_indices = np.setdiff1d(np.arange(n_total), s_indices)

    X_all = np.column_stack((np.ones(n_total), all_points[:, 0], all_points[:, 1]))
    return {
        "all_points": all_points,
        "S": all_points[s_indices],
        "U": all_points[u_indices],
        "s_indices": s_indices,
        "u_indices": u_indices,
        "X_all": X_all,
        "X_S": X_all[s_indices],
        "X_U": X_all[u_indices],
    }


def matern32_kernel(matern_params: tuple[float, float]) -> Matern:
    sigma, length_scale = matern_params
    return (float(sigma) ** 2) * Matern(length_scale=float(length_scale), nu=1.5)


def conditional_coefficients(
    point: Array,
    neighbor_points: Array,
    matern_params: tuple[float, float],
    jitter: float,
) -> tuple[Array, float]:
    sigma, _ = matern_params
    if len(neighbor_points) == 0:
        return np.zeros(0, dtype=np.float64), float(sigma) ** 2

    kernel = matern32_kernel(matern_params)
    C_NN = np.asarray(kernel(neighbor_points), dtype=np.float64)
    C_NN.flat[:: len(neighbor_points) + 1] += float(jitter)
    C_iN = np.asarray(kernel(point.reshape(1, -1), neighbor_points)[0], dtype=np.float64)
    B_i = C_iN @ np.linalg.inv(C_NN)
    C_ii = float(kernel(point.reshape(1, -1))[0, 0])
    F_i = C_ii - float(B_i @ C_iN.T)
    return B_i, max(F_i, float(jitter))


def previous_neighbor_indices(points: Array, i: int, m: int) -> Array:
    if i <= 0:
        return np.zeros(0, dtype=int)
    distances = cdist(points[i : i + 1], points[:i])[0]
    return np.argsort(distances)[: min(int(m), i)].astype(int)


def reference_neighbor_matrix(points: Array, m: int) -> Array:
    neighbors = np.full((len(points), int(m)), 0, dtype=int)
    for i in range(len(points)):
        idx = previous_neighbor_indices(points, i, m)
        if len(idx) == 0:
            continue
        neighbors[i, : len(idx)] = idx
        if len(idx) < m:
            neighbors[i, len(idx) :] = idx[-1]
    return neighbors


def simulate_reference_field(
    S: Array,
    m: int,
    sine_neighbors: int,
    sine_amplitude: float,
    sine_frequency: float,
    matern_params: tuple[float, float],
    jitter: float,
    rng: np.random.Generator,
) -> dict[str, Array]:
    y_S = np.zeros(len(S), dtype=np.float64)
    linear_mean_S = np.zeros(len(S), dtype=np.float64)
    sine_input_S = np.zeros(len(S), dtype=np.float64)
    nonlinear_mean_S = np.zeros(len(S), dtype=np.float64)
    conditional_var_S = np.zeros(len(S), dtype=np.float64)

    for i in range(len(S)):
        N10 = previous_neighbor_indices(S, i, m)
        B10, F_i = conditional_coefficients(S[i], S[N10], matern_params, jitter)
        linear_mean = float(B10 @ y_S[N10]) if len(N10) else 0.0

        n_sine = min(int(sine_neighbors), len(N10))
        if n_sine > 0:
            N2 = N10[:n_sine]
            B2, _ = conditional_coefficients(S[i], S[N2], matern_params, jitter)
            sine_input = float(B2 @ y_S[N2])
        else:
            sine_input = 0.0

        nonlinear_mean = float(sine_amplitude) * np.sin(float(sine_frequency) * sine_input)
        y_S[i] = rng.normal(linear_mean + nonlinear_mean, np.sqrt(F_i))

        linear_mean_S[i] = linear_mean
        sine_input_S[i] = sine_input
        nonlinear_mean_S[i] = nonlinear_mean
        conditional_var_S[i] = F_i

    return {
        "y_S": y_S,
        "linear_mean_S": linear_mean_S,
        "sine_input_S": sine_input_S,
        "nonlinear_mean_S": nonlinear_mean_S,
        "conditional_var_S": conditional_var_S,
    }


def simulate_prediction_field(
    U: Array,
    S: Array,
    y_S: Array,
    m: int,
    sine_neighbors: int,
    sine_amplitude: float,
    sine_frequency: float,
    matern_params: tuple[float, float],
    jitter: float,
    rng: np.random.Generator,
) -> dict[str, Array]:
    y_U = np.zeros(len(U), dtype=np.float64)
    linear_mean_U = np.zeros(len(U), dtype=np.float64)
    sine_input_U = np.zeros(len(U), dtype=np.float64)
    nonlinear_mean_U = np.zeros(len(U), dtype=np.float64)
    conditional_var_U = np.zeros(len(U), dtype=np.float64)
    neighbor_indices_U = np.full((len(U), int(m)), 0, dtype=int)

    for j, point in enumerate(U):
        distances = cdist(point.reshape(1, -1), S)[0]
        N10 = np.argsort(distances)[: int(m)].astype(int)
        neighbor_indices_U[j] = N10

        B10, F_j = conditional_coefficients(point, S[N10], matern_params, jitter)
        linear_mean = float(B10 @ y_S[N10])

        n_sine = min(int(sine_neighbors), len(N10))
        N2 = N10[:n_sine]
        B2, _ = conditional_coefficients(point, S[N2], matern_params, jitter)
        sine_input = float(B2 @ y_S[N2])

        nonlinear_mean = float(sine_amplitude) * np.sin(float(sine_frequency) * sine_input)
        y_U[j] = rng.normal(linear_mean + nonlinear_mean, np.sqrt(F_j))

        linear_mean_U[j] = linear_mean
        sine_input_U[j] = sine_input
        nonlinear_mean_U[j] = nonlinear_mean
        conditional_var_U[j] = F_j

    return {
        "y_U": y_U,
        "linear_mean_U": linear_mean_U,
        "sine_input_U": sine_input_U,
        "nonlinear_mean_U": nonlinear_mean_U,
        "conditional_var_U": conditional_var_U,
        "neighbor_indices_U": neighbor_indices_U,
    }


def save_csv(path: Path, data: dict[str, Array]) -> None:
    all_points = data["all_points"]
    X_all = data["X_all"]
    y_all = data["y_all"]
    s_indices = data["s_indices"].astype(int)
    u_indices = data["u_indices"].astype(int)

    split = np.full(len(all_points), "U", dtype=object)
    split[s_indices] = "S"

    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "split", "x", "y", "y_obs", "x0", "x1", "x2"])
        for idx in np.concatenate([s_indices, u_indices]):
            writer.writerow(
                [
                    int(idx),
                    split[idx],
                    f"{all_points[idx, 0]:.17g}",
                    f"{all_points[idx, 1]:.17g}",
                    f"{y_all[idx]:.17g}",
                    f"{X_all[idx, 0]:.17g}",
                    f"{X_all[idx, 1]:.17g}",
                    f"{X_all[idx, 2]:.17g}",
                ]
            )


def grid_field(points: Array, values: Array) -> tuple[Array, Array, Array]:
    x_unique = np.unique(points[:, 0])
    y_unique = np.unique(points[:, 1])
    order = np.lexsort((points[:, 0], points[:, 1]))
    grid_values = values[order].reshape(len(y_unique), len(x_unique))
    return x_unique, y_unique, grid_values


def plot_field(points: Array, values: Array, title: str, save_path: Path) -> None:
    x_unique, y_unique, grid_values = grid_field(points, values)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(x_unique, y_unique, grid_values, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="value")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect("equal")
    ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_points(S: Array, U: Array, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(U[:, 0], U[:, 1], s=8, c="#d1d5db", label=f"Prediction points U ({len(U)})")
    ax.scatter(S[:, 0], S[:, 1], s=14, c="#1f77b4", label=f"Reference points S ({len(S)})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=True)
    ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_reference_split_data(
    grid_size: int = GRID_SIZE,
    domain_size: float = DOMAIN_SIZE,
    num_reference_points: int = NUM_REFERENCE_POINTS,
    random_seed: int = RANDOM_SEED,
    m: int = M,
    sine_neighbors: int = SINE_NEIGHBORS,
    sine_amplitude: float = SINE_AMPLITUDE,
    sine_frequency: float = SINE_FREQUENCY,
    matern_params: tuple[float, float] = MATERN_PARAMS,
    jitter: float = JITTER,
    output_npz: Path = OUTPUT_NPZ,
    save_plots: bool = True,
) -> Path:
    if m < 1:
        raise ValueError(f"m must be positive, got {m}.")
    if sine_neighbors < 1 or sine_neighbors > m:
        raise ValueError(f"sine_neighbors must be in [1, m], got {sine_neighbors}.")
    if jitter <= 0:
        raise ValueError(f"jitter must be positive, got {jitter}.")

    design = make_spatial_design(
        num_reference_points=num_reference_points,
        grid_size=grid_size,
        domain_size=domain_size,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(random_seed)

    ref = simulate_reference_field(
        design["S"],
        m=m,
        sine_neighbors=sine_neighbors,
        sine_amplitude=sine_amplitude,
        sine_frequency=sine_frequency,
        matern_params=matern_params,
        jitter=jitter,
        rng=rng,
    )
    pred = simulate_prediction_field(
        design["U"],
        design["S"],
        ref["y_S"],
        m=m,
        sine_neighbors=sine_neighbors,
        sine_amplitude=sine_amplitude,
        sine_frequency=sine_frequency,
        matern_params=matern_params,
        jitter=jitter,
        rng=rng,
    )

    all_points = design["all_points"]
    s_indices = design["s_indices"]
    u_indices = design["u_indices"]

    y_all = np.zeros(len(all_points), dtype=np.float64)
    y_all[s_indices] = ref["y_S"]
    y_all[u_indices] = pred["y_U"]

    linear_mean_all = np.zeros(len(all_points), dtype=np.float64)
    sine_input_all = np.zeros(len(all_points), dtype=np.float64)
    nonlinear_mean_all = np.zeros(len(all_points), dtype=np.float64)
    conditional_var_all = np.zeros(len(all_points), dtype=np.float64)
    linear_mean_all[s_indices] = ref["linear_mean_S"]
    sine_input_all[s_indices] = ref["sine_input_S"]
    nonlinear_mean_all[s_indices] = ref["nonlinear_mean_S"]
    conditional_var_all[s_indices] = ref["conditional_var_S"]
    linear_mean_all[u_indices] = pred["linear_mean_U"]
    sine_input_all[u_indices] = pred["sine_input_U"]
    nonlinear_mean_all[u_indices] = pred["nonlinear_mean_U"]
    conditional_var_all[u_indices] = pred["conditional_var_U"]

    true_params = {
        "simulation": "exp_neighbor_sine_reference_split_archive",
        "data_generating_process": "reference_split_neighbor_sine_nngp",
        "direct_y_process": True,
        "matern_params": tuple(float(x) for x in matern_params),
        "beta": np.zeros(3, dtype=np.float64),
        "tau2": 0.0,
        "sigma_epsilon": float(jitter),
        "dgp_sigma_epsilon": 0.0,
        "m": int(m),
        "m_tilde": int(NNNGP_FIT_M_TILDE),
        "sine_neighbors": int(sine_neighbors),
        "sine_amplitude": float(sine_amplitude),
        "sine_frequency": float(sine_frequency),
        "tau_params": DEFAULT_TAU_PARAMS,
        "g_params": DEFAULT_G_PARAMS,
        "domain_size": float(domain_size),
        "grid_size": int(grid_size),
        "random_seed": int(random_seed),
        "jitter": float(jitter),
    }

    ensure_parent_dir(output_npz)
    np.savez_compressed(
        output_npz,
        all_points=all_points,
        S=design["S"],
        s_indices=s_indices,
        y_S=ref["y_S"],
        U=design["U"],
        u_indices=u_indices,
        y_U=pred["y_U"],
        X_all=design["X_all"],
        X_S=design["X_S"],
        X_U=design["X_U"],
        true_params=true_params,
        y_all=y_all,
        linear_mean_all=linear_mean_all,
        sine_input_all=sine_input_all,
        nonlinear_mean_all=nonlinear_mean_all,
        conditional_var_all=conditional_var_all,
        neighbors_S=reference_neighbor_matrix(design["S"], m),
        neighbors_U=pred["neighbor_indices_U"],
    )
    save_csv(output_npz.with_suffix(".csv"), {**design, "y_all": y_all})

    if save_plots:
        plot_reference_points(
            design["S"],
            design["U"],
            "Reference-split archive locations",
            PLOTS_DIR / "reference_points.png",
        )
        plot_field(all_points, y_all, "Reference-split neighbor-sine field", PLOTS_DIR / "y_field.png")
        plot_field(
            all_points,
            nonlinear_mean_all,
            f"Nonlinear mean {sine_amplitude:.3g} sin({sine_frequency:.3g} B2 y_N2)",
            PLOTS_DIR / "nonlinear_mean_field.png",
        )

    print(f"Reference-split archive data saved to: {output_npz}")
    print(
        f"reference points={num_reference_points}, prediction points={len(design['U'])}, "
        f"m={m}, sine_neighbors={sine_neighbors}, sine_amplitude={sine_amplitude:.6g}, "
        f"sine_frequency={sine_frequency:.6g}, matern_params={matern_params}, "
        f"random_seed={random_seed}"
    )
    return output_npz


def main() -> None:
    generate_reference_split_data()


if __name__ == "__main__":
    main()
