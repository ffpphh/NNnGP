"""Run VI tau-sensitivity experiments.

For each tau_params value, this script generates one dataset while keeping the
rest of the simulation settings fixed, then runs the standard VI inference
pipeline under multiple variational guides and saves results in the same format
as vi_inference.py.
"""

import argparse
import csv
import os
import warnings
from types import SimpleNamespace

import numpy as np

warnings.filterwarnings("ignore")

from vi_inference import (
    RESULTS_DIR,
    build_param_mode,
    load_npz_data,
    parse_csv_list,
    prepare_fit_data,
    run_one_simulation,
    save_vi_metric_summary,
)
from hmc_inference import run_hmc_inference
from inference_utils import make_parent_matern_kernel, precompute_RZ
from model import (
    build_neighbor_indices,
    compute_sparse_reverse_cholesky,
    predict,
)
from simulation_data_utils import (
    _save_simulation_csv_from_npz,
    density_point_indices,
    density_repeats,
    domain_size,
    g_params,
    generate_matern_gp_nnngp_data,
    generate_mlp_nnngp_data,
    generate_tanh_nnngp_simulation,
    k,
    matern_gp_grid_size,
    matern_gp_include_residual,
    matern_params,
    mlp_bias_scale,
    mlp_grid_size,
    mlp_hidden_dims,
    mlp_weight_scale,
    parametric_g_params,
    random_seed,
    sigma_epsilon,
    tanh_grid_size,
    beta,
)


# ==============================================================================
# Manual experiment configuration
# ==============================================================================
# Change these values, then run:
#   python NNnGP/vi_tau_sensitivity.py
#
# RUN_KINDS options:
#   "tanh_nnngp", "mlp_nnngp", "matern_gp_nnngp"
RUN_KINDS = ("matern_gp_nnngp",)

# Generate one dataset for each of these tau_params values.
TAU_VALUES = (
    (-5.0, 0.1),
    (0.0, 0.1),
    (5.0, 0.1),
)

NONLINEARITY_LABELS = dict(zip(TAU_VALUES, ("weak", "median", "strong")))

# Compare these VI inference methods for every tau_params value.
# FlowJAX MAF is labeled as vi_nf; diagonal AutoNormal is mean_field.
GUIDE_METHODS = (
    ("flow", "vi_nf"),
    ("diagonal", "mean_field"),
    ("lowrank", "lowrank"),
)

VI_TABLE_LABELS = {
    "hmc": "HMC",
    "mean_field": "VI-MF",
    "lowrank": "VI-LR",
    "vi_nf": "VI-NF",
}

# Keep all NNnGP fitting/data neighbor settings fixed.
M = 10
M_TILDE = 50

SAVE_DIR = os.path.join(RESULTS_DIR, "vi_tau_sensitivity")
SIMULATION_SAVE_DIR = os.path.join(SAVE_DIR, "simulation_outputs")
SAVE_SIMULATION_PLOTS = True


def _make_args(cli_args, guide):
    return SimpleNamespace(
        iters=cli_args.iters,
        lr=cli_args.lr,
        particles=cli_args.particles,
        vi_samples=cli_args.vi_samples,
        flows=cli_args.flows,
        guide=guide,
        early_stop_patience=cli_args.early_stop_patience,
        early_stop_min_delta=cli_args.early_stop_min_delta,
        early_stop_min_iters=cli_args.early_stop_min_iters,
        early_stop_window=cli_args.early_stop_window,
        early_stop_check_every=cli_args.early_stop_check_every,
        early_stop_log_every=cli_args.early_stop_log_every,
        pred_max_samples=cli_args.pred_max_samples,
        compare_hmc=cli_args.compare_hmc,
    )


def _nonlinearity_label(tau_value):
    tau_key = tuple(float(value) for value in tau_value)
    try:
        return NONLINEARITY_LABELS[tau_key]
    except KeyError as exc:
        raise ValueError(
            f"No weak/median/strong label configured for tau_params={tau_key}"
        ) from exc


def _guide_label(guide):
    labels = dict(GUIDE_METHODS)
    return labels.get(guide, guide)


def _data_filename(run_kind):
    names = {
        "tanh_nnngp": "tanh_nnngp_data.npz",
        "mlp_nnngp": "mlp_nnngp_data.npz",
        "matern_gp_nnngp": "matern_gp_nnngp_data.npz",
    }
    return names.get(run_kind, f"{run_kind}_data.npz")


def _data_path(run_kind, tau_value):
    return os.path.join(
        SIMULATION_SAVE_DIR,
        run_kind,
        _nonlinearity_label(tau_value),
        _data_filename(run_kind),
    )


def _ensure_data_csv(data_path):
    csv_path = os.path.splitext(data_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        csv_path = _save_simulation_csv_from_npz(data_path)
    print(f"CSV 数据: {csv_path}")
    return csv_path


def _generate_tau_data(run_kind, tau_value, force=True):
    """Generate one data file for the requested run_kind and tau_params."""
    data_path = _data_path(run_kind, tau_value)
    plot_dir = os.path.join(os.path.dirname(data_path), "plots")
    if os.path.exists(data_path) and not force:
        print(f"使用已有 tau-sensitivity 数据: {data_path}")
        _ensure_data_csv(data_path)
        return data_path

    seed = int(random_seed)
    print("\n" + "-" * 70)
    print(f"生成 tau-sensitivity 数据: run_kind={run_kind}, tau_params={tau_value}, seed={seed}")
    print(f"数据输出: {data_path}")

    if run_kind == "tanh_nnngp":
        path = generate_tanh_nnngp_simulation(
            k=k,
            m=M,
            m_tilde=M_TILDE,
            grid_size=tanh_grid_size,
            domain_size=domain_size,
            matern_params=matern_params,
            tau_params=tau_value,
            g_params=g_params,
            parametric_g_params=parametric_g_params,
            beta=beta,
            sigma_epsilon=sigma_epsilon,
            random_seed=seed,
            save_path=data_path,
            plot_dir=plot_dir,
            save_plots=SAVE_SIMULATION_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
            standardize_g=True,
        )
        _ensure_data_csv(path)
        return path

    if run_kind == "mlp_nnngp":
        path = generate_mlp_nnngp_data(
            k=k,
            m=M,
            m_tilde=M_TILDE,
            grid_size=mlp_grid_size,
            domain_size=domain_size,
            matern_params=matern_params,
            tau_params=tau_value,
            g_params=g_params,
            beta=beta,
            sigma_epsilon=sigma_epsilon,
            mlp_hidden_dims=mlp_hidden_dims,
            mlp_weight_scale=mlp_weight_scale,
            mlp_bias_scale=mlp_bias_scale,
            random_seed=seed,
            save_path=data_path,
            plot_dir=plot_dir,
            save_plots=SAVE_SIMULATION_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
            standardize_g=True,
        )
        _ensure_data_csv(path)
        return path

    if run_kind == "matern_gp_nnngp":
        path = generate_matern_gp_nnngp_data(
            k=k,
            m=M,
            m_tilde=M_TILDE,
            grid_size=matern_gp_grid_size,
            domain_size=domain_size,
            matern_params=matern_params,
            tau_params=tau_value,
            g_params=g_params,
            beta=beta,
            sigma_epsilon=sigma_epsilon,
            random_seed=seed,
            save_path=data_path,
            plot_dir=plot_dir,
            save_plots=SAVE_SIMULATION_PLOTS,
            repeats=density_repeats,
            point_indices=density_point_indices,
            include_residual=matern_gp_include_residual,
        )
        _ensure_data_csv(path)
        return path

    raise ValueError(f"Unknown RUN_KINDS entry: {run_kind}")


def _run_hmc_tau(data_path, save_dir, args):
    """Run conditional NUTS with every model parameter fixed to its true value."""
    print("\nHMC parameter mode (conditional on true hyperparameters):")
    for parameter_name in (
        "sigma_f",
        "length_scale",
        "theta_tau1",
        "theta_tau2",
        "theta_g1",
        "theta_g2",
        "beta",
        "sigma_epsilon",
    ):
        print(f"  {parameter_name:15s}: fixed")

    data = load_npz_data(data_path)
    S = np.asarray(data["S"], dtype=np.float64)
    y_S = np.asarray(data["y_S"], dtype=np.float64).flatten()
    X_S = np.asarray(data["X_S"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    y_U = np.asarray(data["y_U"], dtype=np.float64).flatten()
    X_U = np.asarray(data["X_U"], dtype=np.float64)
    true_w_S = np.asarray(data["w_S"], dtype=np.float64).flatten() if "w_S" in data else None
    true_w_U = np.asarray(data["w_U"], dtype=np.float64).flatten() if "w_U" in data else None
    true_params, Z = prepare_fit_data(
        data,
        "matern_gp_nnngp",
        fit_overrides={"m": int(M), "m_tilde": int(M_TILDE)},
        force_rebuild_z=False,
        z_seed=int(random_seed),
    )

    m = int(true_params["m"])
    m_tilde = int(true_params["m_tilde"])
    neighbors = build_neighbor_indices(S, m)
    matern_kernel = make_parent_matern_kernel(true_params)
    L_true = compute_sparse_reverse_cholesky(S, matern_kernel, m)
    R_Z, R_Z_inv = precompute_RZ(Z, true_params["g_params"][2])

    os.makedirs(save_dir, exist_ok=True)
    hmc_w_S, hmc_beta, hmc_sigma = run_hmc_inference(
        X_S=X_S,
        y_S=y_S,
        S=S,
        true_params=true_params,
        m=m,
        m_tilde=m_tilde,
        L=L_true,
        neighbors=neighbors,
        Z=Z,
        R_Z=R_Z,
        R_Z_inv=R_Z_inv,
        num_warmup=args.hmc_warmup,
        num_samples=args.hmc_samples,
        thinning=args.hmc_thinning,
        num_chains=args.hmc_chains,
        save_dir=save_dir,
    )

    pred_max = None if str(args.pred_max_samples).lower() == "none" else int(args.pred_max_samples)
    pred_mean, pred_std, pred_samples, w_pred_mean, w_pred_std, w_pred_samples = predict(
        hmc_w_S,
        S,
        U,
        X_U,
        hmc_beta,
        hmc_sigma,
        true_params,
        m=m,
        Z=Z,
        neighbors_S=neighbors,
        max_samples=pred_max,
        seed=101,
        return_w=True,
    )
    metric_summary = save_vi_metric_summary(
        save_dir,
        y_U=y_U,
        pred_mean=pred_mean,
        pred_std=pred_std,
        pred_samples=pred_samples,
        true_w_U=true_w_U,
        w_pred_mean=w_pred_mean,
        w_pred_samples=w_pred_samples,
        true_w_S=true_w_S,
        w_S_samples=hmc_w_S,
        summary_filename="hmc_metric_summary.csv",
    )
    np.savez_compressed(
        os.path.join(save_dir, "hmc_prediction_results.npz"),
        pred_mean=pred_mean,
        pred_std=pred_std,
        pred_samples=pred_samples,
        w_pred_mean=w_pred_mean,
        w_pred_std=w_pred_std,
        w_pred_samples=w_pred_samples,
        y_U=y_U,
        w_U=true_w_U,
    )
    return {
        "base_run_kind": "matern_gp_nnngp",
        "guide": "hmc",
        "guide_label": "hmc",
        "metric_summary": metric_summary,
        "vi_params": true_params,
        "true_params": true_params,
    }


def _write_summary(rows, save_dir):
    if not rows:
        return
    nnngp_rows = [row for row in rows if row["base_run_kind"] == "matern_gp_nnngp"]
    if not nnngp_rows:
        return

    summary_dir = os.path.join(save_dir, "matern_gp_nnngp")
    os.makedirs(summary_dir, exist_ok=True)

    def formatted_row(table_row):
        return {
            key: f"{value:.10g}" if isinstance(value, (float, np.floating)) else value
            for key, value in table_row.items()
        }

    model_summary_path = os.path.join(summary_dir, "summary.csv")
    model_fieldnames = [
        "nonlinearity",
        "method",
        "vi_method",
        "beta_0",
        "beta_1",
        "beta_2",
        "sigma_epsilon_squared",
        "RMSPE",
        "CRPS",
        "CI_coverage_percent",
        "CI_width",
    ]

    true_params = nnngp_rows[0]["true_params"]
    true_beta = np.asarray(true_params["beta"])
    model_rows = [
        {
            "nonlinearity": "Truth",
            "method": "",
            "vi_method": "",
            "beta_0": true_beta[0],
            "beta_1": true_beta[1],
            "beta_2": true_beta[2],
            "sigma_epsilon_squared": float(true_params["sigma_epsilon"]) ** 2,
            "RMSPE": "",
            "CRPS": "",
            "CI_coverage_percent": "",
            "CI_width": "",
        }
    ]

    order = {label: index for index, label in enumerate(("weak", "median", "strong"))}
    vi_nf_rows = [row for row in nnngp_rows if row["guide_label"] == "vi_nf"]
    for row in sorted(
        vi_nf_rows,
        key=lambda item: order[_nonlinearity_label(item["data_tau_params"])],
    ):
        metrics = row["metric_summary"]
        vi_beta = np.asarray(row["vi_params"]["beta"])
        vi_sigma_epsilon = float(row["vi_params"]["sigma_epsilon"])
        model_rows.append(
            {
                "nonlinearity": _nonlinearity_label(row["data_tau_params"]),
                "method": "NNnGP",
                "vi_method": VI_TABLE_LABELS[row["guide_label"]],
                "beta_0": vi_beta[0],
                "beta_1": vi_beta[1],
                "beta_2": vi_beta[2],
                "sigma_epsilon_squared": vi_sigma_epsilon ** 2,
                "RMSPE": metrics["y_U_rmse"],
                "CRPS": metrics["y_U_crps"],
                "CI_coverage_percent": 100.0 * metrics["y_U_coverage_quantile_95"],
                "CI_width": metrics["y_U_mean_interval_width_quantile_95"],
            }
        )

    with open(model_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=model_fieldnames)
        writer.writeheader()
        writer.writerows(formatted_row(row) for row in model_rows)
    print(f"\nNNnGP VI-NF summary saved to: {model_summary_path}")

    vi_summary_path = os.path.join(summary_dir, "vi_method_summary.csv")
    vi_fieldnames = [
        "nonlinearity",
        "method",
        "sigma_C",
        "theta_C",
        "theta_tau_1",
        "theta_tau_2",
        "theta_lambda_1",
        "theta_lambda_2",
        "RMSPE",
        "CRPS",
        "CI_coverage_percent",
        "CI_width",
    ]
    guide_order = {"hmc": 0, "mean_field": 1, "lowrank": 2, "vi_nf": 3}
    vi_rows = []

    for nonlinearity in ("weak", "median", "strong"):
        level_rows = [
            row
            for row in nnngp_rows
            if _nonlinearity_label(row["data_tau_params"]) == nonlinearity
        ]
        if not level_rows:
            continue

        level_true_params = level_rows[0]["true_params"]
        true_matern = np.asarray(level_true_params["matern_params"])
        true_tau = np.asarray(level_true_params["tau_params"])
        true_g = np.asarray(level_true_params["g_params"])
        vi_rows.append(
            {
                "nonlinearity": nonlinearity,
                "method": "Truth",
                "sigma_C": true_matern[0],
                "theta_C": true_matern[1],
                "theta_tau_1": true_tau[0],
                "theta_tau_2": true_tau[1],
                "theta_lambda_1": true_g[0],
                "theta_lambda_2": true_g[1],
                "RMSPE": "",
                "CRPS": "",
                "CI_coverage_percent": "",
                "CI_width": "",
            }
        )

        for row in sorted(level_rows, key=lambda item: guide_order[item["guide_label"]]):
            metrics = row["metric_summary"]
            vi_matern = np.asarray(row["vi_params"]["matern_params"])
            vi_tau = np.asarray(row["vi_params"]["tau_params"])
            vi_g = np.asarray(row["vi_params"]["g_params"])
            vi_rows.append(
                {
                    "nonlinearity": nonlinearity,
                    "method": VI_TABLE_LABELS[row["guide_label"]],
                    "sigma_C": vi_matern[0],
                    "theta_C": vi_matern[1],
                    "theta_tau_1": vi_tau[0],
                    "theta_tau_2": vi_tau[1],
                    "theta_lambda_1": vi_g[0],
                    "theta_lambda_2": vi_g[1],
                    "RMSPE": metrics["y_U_rmse"],
                    "CRPS": metrics["y_U_crps"],
                    "CI_coverage_percent": 100.0 * metrics["y_U_coverage_quantile_95"],
                    "CI_width": metrics["y_U_mean_interval_width_quantile_95"],
                }
            )

    with open(vi_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=vi_fieldnames)
        writer.writeheader()
        writer.writerows(formatted_row(row) for row in vi_rows)
    print(f"VI-method comparison summary saved to: {vi_summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--preset", default="all_eb", choices=["all_fixed", "all_eb", "fixed_base", "fixed_base_fixed_noise"])
    parser.add_argument("--fixed", default="", help="Comma-separated parameters to force fixed")
    parser.add_argument("--eb", default="", help="Comma-separated parameters to force EB/MLE")
    parser.add_argument(
        "--guides",
        default=os.environ.get("VI_GUIDES", "flow,diagonal,lowrank"),
        help="Comma-separated guides to compare: flow, diagonal, lowrank.",
    )
    parser.add_argument("--iters", type=int, default=int(os.environ.get("VI_ITERS", "15000")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("VI_LR", "0.0002")))
    parser.add_argument("--particles", type=int, default=int(os.environ.get("VI_PARTICLES", "1")))
    parser.add_argument("--vi-samples", type=int, default=int(os.environ.get("VI_SAMPLES", "1000")))
    parser.add_argument("--flows", type=int, default=int(os.environ.get("VI_FLOWS", "3")))
    parser.add_argument("--hidden-dims", default=os.environ.get("VI_HIDDEN_DIMS", "256,256"))
    parser.add_argument("--early-stop-patience", type=int, default=int(os.environ.get("VI_EARLY_STOP_PATIENCE", "0")))
    parser.add_argument("--early-stop-min-delta", type=float, default=float(os.environ.get("VI_EARLY_STOP_MIN_DELTA", "0.001")))
    parser.add_argument("--early-stop-min-iters", type=int, default=int(os.environ.get("VI_EARLY_STOP_MIN_ITERS", "5000")))
    parser.add_argument("--early-stop-window", type=int, default=int(os.environ.get("VI_EARLY_STOP_WINDOW", "100")))
    parser.add_argument("--early-stop-check-every", type=int, default=int(os.environ.get("VI_EARLY_STOP_CHECK_EVERY", "500")))
    parser.add_argument("--early-stop-log-every", type=int, default=int(os.environ.get("VI_EARLY_STOP_LOG_EVERY", "100")))
    parser.add_argument("--pred-max-samples", default=os.environ.get("PRED_MAX_SAMPLES", "200"))
    parser.add_argument("--hmc-warmup", type=int, default=int(os.environ.get("HMC_WARMUP", "5000")))
    parser.add_argument(
        "--hmc-samples",
        type=int,
        default=int(os.environ.get("HMC_SAMPLES", "5000")),
        help="Number of post-warmup HMC iterations before thinning.",
    )
    parser.add_argument("--hmc-thinning", type=int, default=int(os.environ.get("HMC_THIN", "5")))
    parser.add_argument("--hmc-chains", type=int, default=int(os.environ.get("HMC_CHAINS", "1")))
    parser.add_argument("--compare-hmc", default="")
    parser.add_argument("--reuse-data", action="store_true", help="Reuse existing tau simulation files instead of regenerating them.")
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in parse_csv_list(args.hidden_dims))
    if not hidden_dims:
        hidden_dims = (128, 128)

    guide_values = tuple(parse_csv_list(args.guides))
    if not guide_values:
        guide_values = tuple(guide for guide, _ in GUIDE_METHODS)
    valid_guides = {guide for guide, _ in GUIDE_METHODS}
    unknown_guides = [guide for guide in guide_values if guide not in valid_guides]
    if unknown_guides:
        raise ValueError(f"Unknown guide(s): {unknown_guides}. Valid guides are: {sorted(valid_guides)}")

    param_mode = build_param_mode(args.preset, fixed_text=args.fixed, eb_text=args.eb)

    rows = []
    print("=" * 70)
    print("NNnGP VI tau-sensitivity experiment")
    print("=" * 70)
    print(f"RUN_KINDS: {RUN_KINDS}")
    print(f"TAU_VALUES: {TAU_VALUES}")
    print(f"GUIDES: {guide_values}")
    print(f"Fixed M={M}, M_TILDE={M_TILDE}")
    print(f"Output root: {args.save_dir}")
    print(f"Simulation root: {SIMULATION_SAVE_DIR}")
    print(f"preset: {args.preset}, fixed override: {args.fixed or '(none)'}, eb override: {args.eb or '(none)'}")

    for run_kind in RUN_KINDS:
        if run_kind not in {"tanh_nnngp", "mlp_nnngp", "matern_gp_nnngp"}:
            raise ValueError(f"Unknown RUN_KINDS entry: {run_kind}")

    print("\n" + "=" * 70)
    print("Stage 1/2: generate and save all nonlinearity datasets")
    print("=" * 70)
    data_paths = {}
    for run_kind in RUN_KINDS:
        for tau_value in TAU_VALUES:
            tau_value = tuple(float(x) for x in tau_value)
            nonlinearity_label = _nonlinearity_label(tau_value)
            data_path = _generate_tau_data(run_kind, tau_value, force=not args.reuse_data)
            data_paths[(run_kind, nonlinearity_label)] = data_path

    print("\n" + "=" * 70)
    print("Stage 2/2: run HMC and VI inference")
    print("=" * 70)
    for run_kind in RUN_KINDS:
        for tau_value in TAU_VALUES:
            tau_value = tuple(float(x) for x in tau_value)
            nonlinearity_label = _nonlinearity_label(tau_value)
            data_path = data_paths[(run_kind, nonlinearity_label)]

            if run_kind == "matern_gp_nnngp":
                hmc_save_dir = os.path.join(
                    args.save_dir,
                    run_kind,
                    "hmc",
                    nonlinearity_label,
                )
                hmc_result = _run_hmc_tau(data_path, hmc_save_dir, args)
                hmc_result["data_tau_params"] = tau_value
                rows.append(hmc_result)
                _write_summary(rows, args.save_dir)

            for guide in guide_values:
                guide_label = _guide_label(guide)
                vi_args = _make_args(args, guide)
                save_dir = os.path.join(args.save_dir, run_kind, guide_label, nonlinearity_label)
                result = run_one_simulation(
                    run_kind=f"{run_kind}_{guide_label}_{nonlinearity_label}",
                    data_path=data_path,
                    save_dir=save_dir,
                    args=vi_args,
                    param_mode=param_mode,
                    hidden_dims=hidden_dims,
                    fit_overrides={"m": int(M), "m_tilde": int(M_TILDE)},
                    force_rebuild_z=False,
                    z_seed=int(random_seed),
                )
                result["base_run_kind"] = run_kind
                result["guide"] = guide
                result["guide_label"] = guide_label
                result["data_tau_params"] = tau_value
                rows.append(result)
                _write_summary(rows, args.save_dir)

    print("\n全部 tau-sensitivity 实验完成。输出根目录:", args.save_dir)


if __name__ == "__main__":
    main()
