"""Run FlowJAX/NumPyro VI for NNnGP with configurable fixed/EB parameters.

This file intentionally imports FlowJAX, while hmc_inference.py does not.

Examples
--------
1) VI only infers w_S, all hyperparameters fixed to true values:
   JAX_PLATFORM=gpu python vi_inference.py --preset all_fixed

2) Non-centered VI with base kernel fixed, other parameters EB/MLE:
   JAX_PLATFORM=gpu python vi_inference.py --preset fixed_base

3) Custom: fix selected parameters, estimate the rest by EB:
   JAX_PLATFORM=gpu python vi_inference.py --preset all_eb --fixed sigma_f,length_scale,sigma_epsilon

4) Larger MAF:
   JAX_PLATFORM=gpu VI_FLOWS=5 VI_LR=0.00005 python vi_inference.py --preset fixed_base --hidden-dims 256,256
"""

import os
import argparse
import warnings
from collections import namedtuple
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SVIResult = namedtuple("SVIResult", ["params", "state", "losses"])

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", os.environ.get("XLA_CLIENT_MEM_FRACTION", "0.6"))

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.optim import Adam
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", os.environ.get("JAX_PLATFORM", "gpu"))
jax.config.update("jax_debug_nans", False)

try:
    from flowjax.flows import masked_autoregressive_flow
    from flowjax.distributions import StandardNormal
    from flowjax.experimental.numpyro import distribution_to_numpyro, register_params
    HAS_FLOWJAX = True
except Exception as exc:  # pragma: no cover
    print(f"⚠️ FlowJAX 不可用，将只能使用 lowrank/diagonal guide。原因: {exc}")
    HAS_FLOWJAX = False

from model import (
    PARAMETER_NAMES,
    nnngp_model_vi_configurable,
    extract_configurable_vi_params,
    use_noncentered_from_param_mode,
    compute_sparse_reverse_cholesky,
    build_neighbor_indices,
    precompute_fic_reference_geometry,
    precompute_eb_reference_geometry,
    predict,
)
from inference_utils import (
    ensure_dir,
    load_synthetic_data,
    make_parent_matern_kernel,
    z_to_w_samples,
)
from visualization import (
    evaluate_predictions,
    print_parameter_comparison,
    plot_prediction_maps,
    plot_prediction_scatter,
    plot_ws_true_vs_inferred,
    plot_training_y_true_vs_fitted,
    plot_elbo_loss_curve,
)


class TypedKeyNumpyroDistribution(dist.Distribution):
    """Convert old-style uint32 PRNGKey to new typed key before FlowJAX sample()."""

    arg_constraints = {}
    pytree_data_fields = ("base_dist",)
    pytree_aux_fields = ()

    def __init__(self, base_dist):
        self.base_dist = base_dist
        super().__init__(
            batch_shape=base_dist.batch_shape,
            event_shape=base_dist.event_shape,
            validate_args=getattr(base_dist, "_validate_args", None),
        )

    def sample(self, key, sample_shape=()):
        if getattr(key, "dtype", None) == jnp.uint32:
            key = jax.random.wrap_key_data(key)
        return self.base_dist.sample(key, sample_shape=sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    @property
    def support(self):
        return self.base_dist.support


ALL_PARAMS = list(PARAMETER_NAMES)


def parse_csv_list(text):
    if text is None or text == "":
        return []
    return [x.strip() for x in text.split(",") if x.strip()]



def print_param_mode(param_mode):
    print("\nVI parameter mode:")
    for name in ALL_PARAMS:
        print(f"  {name:15s}: {param_mode[name]}")
    print(f"  latent parameterization: {'non-centered w_S_std' if use_noncentered_from_param_mode(param_mode) else 'direct w_S'}")


def run_vi_inference(
    X_S,
    y_S,
    S,
    init_params,
    param_mode,
    m,
    m_tilde,
    neighbors,
    Z,
    L=None,
    fic_reference_geometry=None,
    eb_reference_geometry=None,
    num_iterations=15000,
    learning_rate=2e-4,
    num_particles=1,
    num_vi_samples=1000,
    num_flows=3,
    hidden_dims=(128, 128),
    guide_type="flow",
    early_stop_patience=0,
    early_stop_min_delta=1e-3,
    early_stop_min_iters=2000,
    early_stop_window=50,
    early_stop_check_every=100,
    early_stop_log_every=100,
    save_dir="results/vi_result",
):
    """Run configurable VI and return posterior samples and parameter estimates."""
    ensure_dir(save_dir)
    use_noncentered = use_noncentered_from_param_mode(param_mode)
    latent_name = "w_S_std" if use_noncentered else "w_S"

    print("\n正在运行 configurable VI 推断...")
    print(f"  随机变量: {latent_name}" + ("，随后通过 T^{-1} 转换为 w_S" if use_noncentered else ""))
    print_param_mode(param_mode)

    def vi_model(
        X_S,
        y_S,
        S,
        m=20,
        m_tilde=10,
        neighbors=None,
        Z=None,
        L=None,
        fic_ref_idx=None,
        fic_ref_N=None,
        fic_ref_B=None,
        fic_ref_F=None,
        fic_ref_dists=None,
        eb_base0_matern_dists=None,
        eb_ref_idx=None,
        eb_ref_N=None,
        eb_ref_matern_NN_dists=None,
        eb_ref_matern_iN_dists=None,
        eb_ref_euclid_iN_dists=None,
    ):
        return nnngp_model_vi_configurable(
            X_S=X_S,
            y_S=y_S,
            S=S,
            m=m,
            m_tilde=m_tilde,
            init_params=init_params,
            param_mode=param_mode,
            neighbors=neighbors,
            Z=Z,
            L=L,
            fic_ref_idx=fic_ref_idx,
            fic_ref_N=fic_ref_N,
            fic_ref_B=fic_ref_B,
            fic_ref_F=fic_ref_F,
            fic_ref_dists=fic_ref_dists,
            eb_base0_matern_dists=eb_base0_matern_dists,
            eb_ref_idx=eb_ref_idx,
            eb_ref_N=eb_ref_N,
            eb_ref_matern_NN_dists=eb_ref_matern_NN_dists,
            eb_ref_matern_iN_dists=eb_ref_matern_iN_dists,
            eb_ref_euclid_iN_dists=eb_ref_euclid_iN_dists,
        )

    model_kwargs = dict(
        X_S=jnp.asarray(X_S, dtype=jnp.float64),
        y_S=jnp.asarray(y_S, dtype=jnp.float64),
        S=jnp.asarray(S, dtype=jnp.float64),
        m=m,
        m_tilde=m_tilde,
        neighbors=jnp.asarray(neighbors, dtype=jnp.int32),
        Z=jnp.asarray(Z, dtype=jnp.float64),
        L=None if L is None else jnp.asarray(L, dtype=jnp.float64),
    )
    if fic_reference_geometry is not None:
        ref_idx, ref_N, ref_B, ref_F, ref_dists = fic_reference_geometry
        model_kwargs.update(
            fic_ref_idx=jnp.asarray(ref_idx, dtype=jnp.int32),
            fic_ref_N=jnp.asarray(ref_N, dtype=jnp.int32),
            fic_ref_B=jnp.asarray(ref_B, dtype=jnp.float64),
            fic_ref_F=jnp.asarray(ref_F, dtype=jnp.float64),
            fic_ref_dists=jnp.asarray(ref_dists, dtype=jnp.float64),
        )
    if eb_reference_geometry is not None:
        base0_dists, ref_idx, ref_N, ref_NN_dists, ref_iN_dists, ref_euclid_dists = eb_reference_geometry
        model_kwargs.update(
            eb_base0_matern_dists=jnp.asarray(base0_dists, dtype=jnp.float64),
            eb_ref_idx=jnp.asarray(ref_idx, dtype=jnp.int32),
            eb_ref_N=jnp.asarray(ref_N, dtype=jnp.int32),
            eb_ref_matern_NN_dists=jnp.asarray(ref_NN_dists, dtype=jnp.float64),
            eb_ref_matern_iN_dists=jnp.asarray(ref_iN_dists, dtype=jnp.float64),
            eb_ref_euclid_iN_dists=jnp.asarray(ref_euclid_dists, dtype=jnp.float64),
        )

    n = len(y_S)
    optimizer = Adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)

    if guide_type == "flow":
        if not HAS_FLOWJAX:
            raise RuntimeError("FlowJAX is not available. Use --guide lowrank or --guide diagonal.")
        print(f"使用 FlowJAX MAF guide for q({latent_name})")
        flow_key = jax.random.key(123)
        base_dist = StandardNormal(shape=(n,))
        flow = masked_autoregressive_flow(
            key=flow_key,
            base_dist=base_dist,
            flow_layers=num_flows,
            nn_width=hidden_dims[0],
            nn_depth=len(hidden_dims),
            nn_activation=jax.nn.relu,
        )
        def guide(X_S, y_S, S, m=20, m_tilde=10, neighbors=None, Z=None, L=None, **kwargs):
            # IMPORTANT:
            # register_params returns the flow object whose trainable leaves are
            # connected to numpyro.param. We must build the NumPyro distribution
            # from this registered flow, not from the original initial flow.
            flow_registered = register_params("flow_params", flow)

            numpyro_flow = TypedKeyNumpyroDistribution(
                distribution_to_numpyro(flow_registered)
            )

            numpyro.sample(latent_name, numpyro_flow)

        elbo_loss = Trace_ELBO(num_particles=1, vectorize_particles=False)
    elif guide_type == "lowrank":
        print(f"使用 AutoLowRankMultivariateNormal guide for q({latent_name})")
        guide = AutoLowRankMultivariateNormal(vi_model, rank=min(50, max(1, n // 10)))
        elbo_loss = Trace_ELBO(num_particles=num_particles, vectorize_particles=(num_particles > 1))
    elif guide_type == "diagonal":
        print(f"使用 AutoDiagonalNormal guide for q({latent_name})")
        guide = AutoDiagonalNormal(vi_model)
        elbo_loss = Trace_ELBO(num_particles=num_particles, vectorize_particles=(num_particles > 1))
    else:
        raise ValueError("guide_type must be flow, lowrank, or diagonal")

    svi = SVI(vi_model, guide, optimizer, loss=elbo_loss)
    if early_stop_patience and early_stop_patience > 0:
        print(
            "\n启用 VI early stopping: "
            f"patience={early_stop_patience}, min_delta={early_stop_min_delta}, "
            f"min_iters={early_stop_min_iters}, window={early_stop_window}, "
            f"check_every={early_stop_check_every}"
        )
        svi_state = svi.init(jax.random.key(44), **model_kwargs)
        losses_list = []
        best_loss = np.inf
        best_state = svi_state
        best_iter = -1
        stale_iters = 0
        completed_iters = 0
        check_every = max(1, int(early_stop_check_every))
        progress = (
            tqdm(total=num_iterations, desc="SVI", dynamic_ncols=True)
            if tqdm is not None
            else None
        )

        while completed_iters < num_iterations:
            chunk_size = min(check_every, num_iterations - completed_iters)
            chunk_result = svi.run(
                jax.random.key(44 + completed_iters),
                chunk_size,
                init_state=svi_state,
                progress_bar=False,
                **model_kwargs,
            )
            svi_state = chunk_result.state
            chunk_losses = np.asarray(jax.device_get(chunk_result.losses), dtype=np.float64)
            losses_list.extend(chunk_losses.tolist())
            completed_iters += chunk_size

            loss_value = float(chunk_losses[-1])
            window = max(1, int(early_stop_window))
            monitor_loss = float(np.mean(losses_list[-window:]))

            improved = monitor_loss < best_loss - early_stop_min_delta
            if improved:
                best_loss = monitor_loss
                best_state = svi_state
                best_iter = completed_iters - 1
                stale_iters = 0
            elif completed_iters >= early_stop_min_iters:
                stale_iters += chunk_size

            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{loss_value:.4g}",
                    monitor=f"{monitor_loss:.4g}",
                    best=f"{best_loss:.4g}",
                    stale=stale_iters,
                )
                progress.update(chunk_size)

            if early_stop_log_every > 0 and (
                completed_iters % early_stop_log_every == 0 or completed_iters == chunk_size
            ):
                msg = (
                    f"  iter {completed_iters}/{num_iterations}, loss={loss_value:.6g}, "
                    f"monitor={monitor_loss:.6g}, best={best_loss:.6g} @ iter {best_iter + 1}"
                )
                if tqdm is not None:
                    progress.write(msg)
                else:
                    print(msg)

            if completed_iters >= early_stop_min_iters and stale_iters >= early_stop_patience:
                msg = (
                    f"Early stopping at iter {completed_iters}; "
                    f"restoring best params from iter {best_iter + 1}."
                )
                if tqdm is not None:
                    progress.write(msg)
                else:
                    print(msg)
                break
        if tqdm is not None:
            progress.close()

        svi_result = SVIResult(
            params=svi.get_params(best_state),
            state=best_state,
            losses=jnp.asarray(losses_list),
        )
    else:
        svi_result = svi.run(jax.random.key(44), num_iterations, progress_bar=True, **model_kwargs)

    print("\nSVI parameter keys:")
    for k in sorted(svi_result.params.keys()):
        print(" ", k)

    predictive = Predictive(guide, params=svi_result.params, num_samples=num_vi_samples)
    guide_samples = predictive(jax.random.key(45), **model_kwargs)

    if use_noncentered:
        w_S_std_samples = np.asarray(guide_samples["w_S_std"])
        w_S_samples = z_to_w_samples(w_S_std_samples, L)
    else:
        w_S_std_samples = None
        w_S_samples = np.asarray(guide_samples["w_S"])

    vi_params = extract_configurable_vi_params(svi_result.params, init_params, param_mode)
    beta_samples = np.tile(vi_params["beta"], (len(w_S_samples), 1))
    sigma_eps_samples = np.tile(vi_params["sigma_epsilon"], (len(w_S_samples),))
    losses = np.asarray(svi_result.losses)

    plot_elbo_loss_curve(
        losses,
        title="VI ELBO Loss Curve",
        save_path=os.path.join(save_dir, "vi_elbo_curve.png"),
    )
    plot_elbo_loss_curve(
        losses,
        title="VI ELBO Loss Curve from Iteration 2000",
        save_path=os.path.join(save_dir, "vi_elbo_curve_from_2000.png"),
        start_iter=2000,
    )

    out_path = os.path.join(save_dir, "vi_results.npz")
    save_dict = dict(
        w_S_samples=w_S_samples,
        beta_samples=beta_samples,
        sigma_epsilon_samples=sigma_eps_samples,
        losses=losses,
        param_mode=np.array(str(param_mode)),
        true_params=np.array(init_params, dtype=object),
        vi_matern_params=np.asarray(vi_params["matern_params"]),
        vi_tau_params=np.asarray(vi_params["tau_params"]),
        vi_g_params=np.asarray(vi_params["g_params"]),
        vi_beta=np.asarray(vi_params["beta"]),
        vi_sigma_epsilon=np.asarray(vi_params["sigma_epsilon"]),
    )
    if w_S_std_samples is not None:
        save_dict["w_S_std_samples"] = w_S_std_samples
    np.savez_compressed(out_path, **save_dict)
    print(f"✅ VI 完成，结果保存到: {out_path}")
    return w_S_samples, beta_samples, sigma_eps_samples, vi_params, losses


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to test_synthetic_data.npz")
    parser.add_argument("--save-dir", default=os.path.join(RESULTS_DIR, "vi_result"))
    parser.add_argument("--preset", default="fixed_base", choices=["all_fixed", "all_eb", "fixed_base", "fixed_base_fixed_noise"])
    parser.add_argument("--fixed", default="", help="Comma-separated parameters to force fixed")
    parser.add_argument("--eb", default="", help="Comma-separated parameters to force EB/MLE")
    parser.add_argument("--guide", default=os.environ.get("VI_GUIDE", "flow"), choices=["flow", "lowrank", "diagonal"])
    parser.add_argument("--iters", type=int, default=int(os.environ.get("VI_ITERS", "15000")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("VI_LR", "0.0002")))
    parser.add_argument("--particles", type=int, default=int(os.environ.get("VI_PARTICLES", "1")))
    parser.add_argument("--vi-samples", type=int, default=int(os.environ.get("VI_SAMPLES", "1000")))
    parser.add_argument("--flows", type=int, default=int(os.environ.get("VI_FLOWS", "3")))
    parser.add_argument("--hidden-dims", default=os.environ.get("VI_HIDDEN_DIMS", "128,128"))
    parser.add_argument("--early-stop-patience", type=int, default=int(os.environ.get("VI_EARLY_STOP_PATIENCE", "0")))
    parser.add_argument("--early-stop-min-delta", type=float, default=float(os.environ.get("VI_EARLY_STOP_MIN_DELTA", "0.001")))
    parser.add_argument("--early-stop-min-iters", type=int, default=int(os.environ.get("VI_EARLY_STOP_MIN_ITERS", "5000")))
    parser.add_argument("--early-stop-window", type=int, default=int(os.environ.get("VI_EARLY_STOP_WINDOW", "100")))
    parser.add_argument("--early-stop-check-every", type=int, default=int(os.environ.get("VI_EARLY_STOP_CHECK_EVERY", "500")))
    parser.add_argument("--early-stop-log-every", type=int, default=int(os.environ.get("VI_EARLY_STOP_LOG_EVERY", "100")))
    parser.add_argument("--pred-max-samples", default=os.environ.get("PRED_MAX_SAMPLES", "200"))
    parser.add_argument(
        "--compare-hmc",
        default=os.path.join(RESULTS_DIR, "hmc_result", "hmc_results.npz"),
    )
    args = parser.parse_args()

    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(BASE_DIR, args.save_dir)
    if args.compare_hmc and not os.path.isabs(args.compare_hmc):
        args.compare_hmc = os.path.join(BASE_DIR, args.compare_hmc)

    hidden_dims = tuple(int(x) for x in parse_csv_list(args.hidden_dims))
    if not hidden_dims:
        hidden_dims = (128, 128)

    param_mode = {
        "sigma_f": "eb",
        "length_scale": "eb",
        "theta_tau1": "eb",
        "theta_tau2": "eb",
        "theta_g1": "eb",
        "theta_g2": "eb",
        "beta": "eb",
        "sigma_epsilon": "eb",
    }

    print("=" * 70)
    print("NNnGP configurable VI inference")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    data, data_path = load_synthetic_data(args.data)
    print(f"加载数据: {data_path}")
    S = np.asarray(data["S"], dtype=np.float64)
    y_S = np.asarray(data["y_S"], dtype=np.float64).flatten()
    X_S = np.asarray(data["X_S"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    y_U = np.asarray(data["y_U"], dtype=np.float64).flatten()
    true_w_U = np.asarray(data["w_U"], dtype=np.float64).flatten() if "w_U" in data else None
    X_U = np.asarray(data["X_U"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    true_w_S = np.asarray(data["w_S"], dtype=np.float64).flatten() if "w_S" in data else None
    m = int(true_params["m"])
    m_tilde = int(true_params["m_tilde"])
    print(f"参考点数量: {len(S)}, 预测点数量: {len(U)}, m={m}, m_tilde={m_tilde}")
    print(f"数据真值 beta: {np.asarray(true_params['beta'])}")

    print("\n预计算 neighbors 和 L_true（当 base kernel fixed 时使用非中心化）...")
    neighbors = build_neighbor_indices(S, m)
    matern_kernel = make_parent_matern_kernel(true_params)
    L_true = compute_sparse_reverse_cholesky(S, matern_kernel, m)
    print("预计算 all-EB 距离几何项（base logprob + Appendix C.2 共用）...")
    eb_reference_geometry = precompute_eb_reference_geometry(S, neighbors, m)
    fic_reference_geometry = None
    if use_noncentered_from_param_mode(param_mode):
        sigma_f0, length_scale0 = true_params["matern_params"]
        print("预计算 Appendix C.2 fixed-base reference 几何项...")
        fic_reference_geometry = precompute_fic_reference_geometry(
            S,
            neighbors,
            m,
            sigma_f0,
            length_scale0,
        )

    vi_w_S, vi_beta, vi_sigma, vi_params, losses = run_vi_inference(
        X_S=X_S,
        y_S=y_S,
        S=S,
        init_params=true_params,
        param_mode=param_mode,
        m=m,
        m_tilde=m_tilde,
        neighbors=neighbors,
        Z=Z,
        L=L_true,
        fic_reference_geometry=fic_reference_geometry,
        eb_reference_geometry=eb_reference_geometry,
        num_iterations=args.iters,
        learning_rate=args.lr,
        num_particles=args.particles,
        num_vi_samples=args.vi_samples,
        num_flows=args.flows,
        hidden_dims=hidden_dims,
        guide_type=args.guide,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_iters=args.early_stop_min_iters,
        early_stop_window=args.early_stop_window,
        early_stop_check_every=args.early_stop_check_every,
        early_stop_log_every=args.early_stop_log_every,
        save_dir=args.save_dir,
    )

    print_parameter_comparison(true_params, vi_params, title="VI fixed/EB 参数结果")

    # Optional comparison against HMC if available.
    if args.compare_hmc and os.path.exists(args.compare_hmc):
        hmc = np.load(args.compare_hmc, allow_pickle=True)
        hmc_w = np.asarray(hmc["w_S_samples"])
        hmc_mean = np.mean(hmc_w, axis=0)
        vi_mean = np.mean(vi_w_S, axis=0)
        min_dim = min(len(hmc_mean), len(vi_mean))
        rmse = np.sqrt(np.mean((hmc_mean[:min_dim] - vi_mean[:min_dim]) ** 2))
        corr = np.corrcoef(hmc_mean[:min_dim], vi_mean[:min_dim])[0, 1]
        print(f"\nHMC vs VI w_S posterior mean diagnostic: RMSE={rmse:.6f}, Corr={corr:.6f}")
    else:
        print("\n未找到 HMC 结果，跳过 HMC-vs-VI posterior 对比。")

    pred_max = None if str(args.pred_max_samples).lower() == "none" else int(args.pred_max_samples)
    print("\n运行 VI posterior prediction...")
    pred_mean, pred_std, pred_samples, w_pred_mean, w_pred_std, w_pred_samples = predict(
        vi_w_S,
        S,
        U,
        X_U,
        vi_beta,
        vi_sigma,
        vi_params,
        m=m,
        Z=Z,
        neighbors_S=neighbors,
        max_samples=pred_max,
        seed=202,
        return_w=True,
    )
    metrics = evaluate_predictions(y_U, pred_mean, pred_std, label="VI")

    np.savez_compressed(
        os.path.join(args.save_dir, "vi_prediction_results.npz"),
        pred_mean=pred_mean,
        pred_std=pred_std,
        pred_samples=pred_samples,
        w_pred_mean=w_pred_mean,
        w_pred_std=w_pred_std,
        w_pred_samples=w_pred_samples,
        y_U=y_U,
        w_U=true_w_U,
        metrics=metrics,
    )

    plot_prediction_maps(
        S, y_S, U, y_U, pred_mean, pred_std,
        title="VI Prediction Results",
        save_path=os.path.join(args.save_dir, "vi_prediction_map.png"),
        include_S=True,
    )
    if true_w_S is not None and true_w_U is not None:
        plot_prediction_maps(
            S, true_w_S, U, true_w_U, w_pred_mean, w_pred_std,
            title="VI Latent w Prediction Results",
            save_path=os.path.join(args.save_dir, "vi_w_prediction_map.png"),
            include_S=True,
        )
    plot_prediction_scatter(
        y_U, pred_mean, method_name="VI",
        save_path=os.path.join(args.save_dir, "vi_true_vs_predicted_yU.png"),
    )
    plot_training_y_true_vs_fitted(
        y_S, X_S, vi_w_S, vi_beta, method_name="VI",
        save_path=os.path.join(args.save_dir, "vi_true_vs_fitted_yS.png"),
    )
    if true_w_S is not None:
        plot_ws_true_vs_inferred(
            true_w_S, vi_w_S, method_name="VI",
            save_path=os.path.join(args.save_dir, "vi_true_vs_inferred_wS.png"),
        )

    print("\n完成。结果目录:", args.save_dir)


if __name__ == "__main__":
    main()
