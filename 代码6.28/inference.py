"""
inference.py -- NNnGP inference driver with Empirical-Bayes VI.

Usage:
    1. Save model_eb.py as model.py, and this file as inference.py.
    2. Generate data if needed:
           python data_utils.py
    3. Run:
           JAX_PLATFORM=gpu PRED_MAX_SAMPLES=200 python inference.py

Design:
    - HMC/NUTS is kept as a conditional gold standard with true hyperparameters,
      sampling only w_S through the fixed-parameter model nnngp_model.
    - VI uses empirical Bayes: q_phi(w_S) is learned by VI, while
      sigma_f, length_scale, theta_tau1, theta_tau2, theta_g1, theta_g2,
      beta, and sigma_epsilon are optimized by numpyro.param. The g-kernel
      range rho is fixed from true_params.
    - The optimized EB parameters are explicitly extracted from svi_result.params
      and passed to predict(), so prediction uses VI-trained parameters rather
      than true_params.
"""

import os
import warnings
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", os.environ.get("XLA_CLIENT_MEM_FRACTION", "0.6"))
# _xla_flags = os.environ.get("XLA_FLAGS", "")
# if "--xla_gpu_enable_command_buffer=" not in _xla_flags:
#     os.environ["XLA_FLAGS"] = (f"{_xla_flags} --xla_gpu_enable_command_buffer=false").strip()

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoLowRankMultivariateNormal
from numpyro.optim import Adam

jax.config.update("jax_enable_x64", False)
_platform = os.environ.get("JAX_PLATFORM", "gpu")
jax.config.update("jax_platform_name", _platform)
jax.config.update("jax_debug_nans", False)

try:
    import equinox as eqx  # noqa: F401
    import flowjax  # noqa: F401
    from flowjax.flows import masked_autoregressive_flow
    from flowjax.distributions import StandardNormal
    from flowjax.experimental.numpyro import distribution_to_numpyro, register_params
    HAS_FLOWJAX = True
except Exception as exc:
    print(f"⚠️ FlowJAX不可用，将使用AutoLowRankMultivariateNormal。原因: {exc}")
    HAS_FLOWJAX = False

import numpyro.distributions as dist


class TypedKeyNumpyroDistribution(dist.Distribution):
    """
    Wrap a NumPyro distribution so that old-style PRNGKey(uint32[2])
    is converted to new-style typed key before calling FlowJAX sample().
    This fixes FlowJAX + NumPyro Predictive key mismatch.
    """

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
        # NumPyro Predictive may pass old-style uint32[2] keys.
        # FlowJAX requires new-style typed keys.
        if getattr(key, "dtype", None) == jnp.uint32:
            key = jax.random.wrap_key_data(key)
        return self.base_dist.sample(key, sample_shape=sample_shape)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    @property
    def support(self):
        return self.base_dist.support

from model import (
    nnngp_model,
    nnngp_model_empirical_bayes,
    extract_empirical_bayes_params,
    compute_sparse_reverse_cholesky,
    build_neighbor_indices,
    g_matern32_kernel_jax,
    predict,
)
from visualization import (
    compare_posteriors,
    evaluate_predictions,
    plot_elbo_loss_curve,
    plot_prediction_maps,
    plot_prediction_performance_comparison,
    plot_training_y_true_vs_fitted,
    plot_ws_true_vs_inferred,
    print_parameter_comparison,
)

np.set_printoptions(precision=6, suppress=True)


# ==============================================================================
# Helpers
# ==============================================================================
def sanitize_true_params(true_params):
    """Convert numpy object-array params into Python/numpy values for safe logging and init."""
    return {
        "matern_params": tuple(float(x) for x in true_params["matern_params"]),
        "tau_params": tuple(float(x) for x in true_params["tau_params"]),
        "g_params": tuple(float(x) for x in true_params["g_params"]),
        "beta": np.asarray(true_params["beta"], dtype=np.float64),
        "sigma_epsilon": float(true_params["sigma_epsilon"]),
        "m": int(true_params["m"]),
        "m_tilde": int(true_params["m_tilde"]),
    }


def make_parent_matern_kernel(params):
    from sklearn.gaussian_process.kernels import Matern
    sigma_f, length_scale = params["matern_params"]
    return sigma_f * Matern(length_scale=length_scale, nu=1.5)


def precompute_RZ(Z, rho, jitter=1e-6):
    Z_jax = jnp.asarray(Z, dtype=jnp.float64)
    R_Z = g_matern32_kernel_jax(Z_jax, Z_jax, rho) + jitter * jnp.eye(Z_jax.shape[0], dtype=jnp.float64)
    R_Z_inv = jnp.linalg.inv(R_Z)
    return np.asarray(R_Z), np.asarray(R_Z_inv)


def z_to_w_samples(z_samples, L):
    """Fixed-parameter HMC: convert z = T w into w = T^{-1} z."""
    L = jnp.asarray(L, dtype=jnp.float64)
    z_samples = jnp.asarray(z_samples, dtype=jnp.float64)
    w_samples = jax.vmap(lambda z: jax.scipy.linalg.solve_triangular(L, z, lower=True, trans="N"))(z_samples)
    return np.asarray(w_samples)


# ==============================================================================
# HMC / NUTS: conditional gold standard with true parameters
# ==============================================================================
def run_hmc_inference(
    model,
    X_S,
    y_S,
    S,
    true_params,
    m=20,
    m_tilde=10,
    num_warmup=300,
    num_samples=500,
    num_chains=1,
    L=None,
    neighbors=None,
    Z=None,
    R_Z=None,
    R_Z_inv=None,
    save_dir=".",
):
    print("\n正在运行HMC/NUTS推断（固定真实参数，仅推断 w_S，用作条件金标准）...")
    kernel = NUTS(
        model,
        target_accept_prob=0.85,
        max_tree_depth=8,
        step_size=0.01,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        forward_mode_differentiation=False,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        jit_model_args=False,
        chain_method="sequential",
    )
    args = dict(
        X_S=jnp.asarray(X_S, dtype=jnp.float64),
        y_S=jnp.asarray(y_S, dtype=jnp.float64),
        S=jnp.asarray(S, dtype=jnp.float64),
        m=m,
        m_tilde=m_tilde,
        true_params=true_params,
        L=jnp.asarray(L, dtype=jnp.float64),
        neighbors=jnp.asarray(neighbors, dtype=jnp.int32),
        Z=jnp.asarray(Z, dtype=jnp.float64),
        R_Z=jnp.asarray(R_Z, dtype=jnp.float64),
        R_Z_inv=jnp.asarray(R_Z_inv, dtype=jnp.float64),
    )
    mcmc.run(jax.random.PRNGKey(42), **args)
    mcmc.print_summary(exclude_deterministic=False)
    samples = mcmc.get_samples()
    w_S_samples = z_to_w_samples(samples["w_S_std"], L)
    beta_samples = np.tile(true_params["beta"], (len(w_S_samples), 1))
    sigma_eps_samples = np.tile(true_params["sigma_epsilon"], (len(w_S_samples),))
    os.makedirs(save_dir, exist_ok=True)
    hmc_path = os.path.join(save_dir, "hmc_conditional_true_params_results.npz")
    np.savez_compressed(
        hmc_path,
        w_S_samples=w_S_samples,
        w_S_std_samples=np.asarray(samples["w_S_std"]),
        beta_samples=beta_samples,
        sigma_epsilon_samples=sigma_eps_samples,
    )
    print(f"✅ HMC推断完成！结果已保存至: {hmc_path}")
    return w_S_samples, beta_samples, sigma_eps_samples


# ==============================================================================
# Empirical-Bayes VI
# ==============================================================================
def run_empirical_bayes_vi_inference(
    X_S,
    y_S,
    S,
    init_params,
    m=20,
    m_tilde=10,
    num_iterations=15000,
    learning_rate=5e-5,
    num_particles=2,
    num_vi_samples=1000,
    num_flows=3,
    hidden_dims=(256,256),
    neighbors=None,
    Z=None,
    L=None,
    save_dir=".",
    guide_type="flow",  # "flow", "lowrank", or "diagonal"
):
    print("\n正在运行 Empirical-Bayes VI 推断...")
    print("  随机变量: w_S")
    print("  ELBO优化参数: sigma_f, length_scale, theta_tau1/2, theta_g1/2, beta, sigma_epsilon; rho固定")

    X_S_jax = jnp.asarray(X_S, dtype=jnp.float64)
    y_S_jax = jnp.asarray(y_S, dtype=jnp.float64)
    S_jax = jnp.asarray(S, dtype=jnp.float64)
    neighbors_jax = jnp.asarray(neighbors, dtype=jnp.int32)
    Z_jax = jnp.asarray(Z, dtype=jnp.float64)

    # Important: init_params is captured in this closure, not passed as a traced SVI argument.
    def eb_model(X_S, y_S, S, m=20, m_tilde=10, neighbors=None, Z=None, L=None):
        return nnngp_model_empirical_bayes(
            X_S=X_S,
            y_S=y_S,
            S=S,
            m=m,
            m_tilde=m_tilde,
            init_params=init_params,
            neighbors=neighbors,
            Z=Z,
            L=L,
        )

    model_kwargs = dict(
        X_S=X_S_jax,
        y_S=y_S_jax,
        S=S_jax,
        m=m,
        m_tilde=m_tilde,
        neighbors=neighbors_jax,
        Z=Z_jax,
        L=jnp.asarray(L, dtype=jnp.float64),
    )

    n = len(y_S)
    optimizer = Adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8)

    if guide_type == "flow" and HAS_FLOWJAX:
        print("使用 FlowJAX MAF guide for q(w_S)")
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
        numpyro_flow = TypedKeyNumpyroDistribution(distribution_to_numpyro(flow))
        def guide(X_S, y_S, S, m=20, m_tilde=10, neighbors=None, Z=None, L=None):
            register_params("flow_params", flow)
            numpyro.sample("w_S_std", numpyro_flow)

    elif guide_type == "lowrank":
        print("使用 NumPyro AutoLowRankMultivariateNormal guide for q(w_S)")
        guide = AutoLowRankMultivariateNormal(eb_model, rank=min(50, max(1, n // 10)))
    else:
        print("使用 NumPyro AutoDiagonalNormal guide for q(w_S)")
        guide = AutoDiagonalNormal(eb_model)

    if guide_type == "flow":
        elbo_loss = Trace_ELBO(num_particles=1, vectorize_particles=False)
    else:
        elbo_loss = Trace_ELBO(num_particles=num_particles, vectorize_particles=(num_particles > 1))

    svi = SVI(
        eb_model,
        guide,
        optimizer,
        loss=elbo_loss,
    )
    svi_result = svi.run(jax.random.key(44), num_iterations, progress_bar=True, **model_kwargs)
    
    # Correct sampling path: use Predictive(guide, params=svi_result.params), not raw flow.sample().
    predictive = Predictive(guide, params=svi_result.params, num_samples=num_vi_samples)
    guide_samples = predictive(jax.random.key(45), **model_kwargs)
    w_S_std_samples = np.asarray(guide_samples["w_S_std"])
    w_S_samples = z_to_w_samples(w_S_std_samples, L)

    eb_params = extract_empirical_bayes_params(svi_result.params, fallback_params=init_params)
    beta_samples = np.tile(eb_params["beta"], (len(w_S_samples), 1))
    sigma_eps_samples = np.tile(eb_params["sigma_epsilon"], (len(w_S_samples),))

    losses = np.asarray(svi_result.losses)
    os.makedirs(save_dir, exist_ok=True)
    elbo_path = os.path.join(save_dir, "eb_vi_elbo_curve.png")
    plot_elbo_loss_curve(
        losses,
        title="Empirical-Bayes VI ELBO Loss Curve",
        save_path=elbo_path,
        window=max(100, num_iterations // 100),
    )

    vi_path = os.path.join(save_dir, "eb_vi_results.npz")
    np.savez_compressed(
        vi_path,
        w_S_samples=w_S_samples,
        beta_samples=beta_samples,
        sigma_epsilon_samples=sigma_eps_samples,
        losses=losses,
        eb_matern_params=np.asarray(eb_params["matern_params"]),
        eb_tau_params=np.asarray(eb_params["tau_params"]),
        eb_g_params=np.asarray(eb_params["g_params"]),
        eb_beta=np.asarray(eb_params["beta"]),
        eb_sigma_epsilon=np.asarray(eb_params["sigma_epsilon"]),
    )
    print(f"✅ Empirical-Bayes VI完成！结果已保存至: {vi_path}")
    return w_S_samples, beta_samples, sigma_eps_samples, eb_params, losses


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    print("=" * 70)
    print("NNnGP推断测试程序：HMC conditional gold standard + Empirical-Bayes VI")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    print("注意：请先运行 python data_utils.py 生成测试数据")
    print("-" * 70)

    results_root = os.path.join(os.path.dirname(__file__), "results")
    run_dir = os.path.join(results_root, "inference_eb_result")
    os.makedirs(run_dir, exist_ok=True)

    print("\n[1/6] 正在加载测试数据...")
    try:
        from data_utils import load_data, RESULTS_DIR
    except ImportError:
        def load_data(path):
            data = np.load(path, allow_pickle=True)
            out = {key: data[key] for key in data.files}
            if isinstance(out.get("true_params"), np.ndarray):
                out["true_params"] = out["true_params"].item()
            return out
        RESULTS_DIR = "./results/data_utils_outputs"

    data_path = os.path.join(RESULTS_DIR, "test_synthetic_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到 {data_path}。请先运行 python data_utils.py")
    data = load_data(data_path)
    S = np.asarray(data["S"], dtype=np.float64)
    y_S = np.asarray(data["y_S"], dtype=np.float64).flatten()
    true_w_S = np.asarray(data["w_S"], dtype=np.float64).flatten()
    X_S = np.asarray(data["X_S"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    y_U = np.asarray(data["y_U"], dtype=np.float64).flatten()
    X_U = np.asarray(data["X_U"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = sanitize_true_params(data["true_params"])
    m = int(true_params["m"])
    m_tilde = int(true_params["m_tilde"])

    print("✅ 数据加载成功！")
    print(f"  参考点数量: {len(S)}")
    print(f"  预测点数量: {len(U)}")
    print(f"  最近邻数量 m: {m}")
    print(f"  FIC诱导点数量 m_tilde: {m_tilde}")

    print("\n[2/6] 正在预计算 HMC 所需的 true-parameter R_Z, R_Z_inv, reverse-Cholesky 和邻居索引...")
    rho_true = true_params["g_params"][2]
    R_Z_true, R_Z_inv_true = precompute_RZ(Z, rho_true)
    matern_kernel = make_parent_matern_kernel(true_params)
    L_true = compute_sparse_reverse_cholesky(S, matern_kernel, m)
    neighbors = build_neighbor_indices(S, m)
    print("✅ 预计算完成")

    print("\n[3/6] 正在运行 HMC/NUTS 条件金标准...")
    hmc_w_S, hmc_beta, hmc_sigma_epsilon = run_hmc_inference(
        nnngp_model,
        X_S=X_S,
        y_S=y_S,
        S=S,
        true_params=true_params,
        m=m,
        m_tilde=m_tilde,
        num_warmup=int(os.environ.get("HMC_WARMUP", "1000")),
        num_samples=int(os.environ.get("HMC_SAMPLES", "5000")),
        num_chains=1,
        L=L_true,
        neighbors=neighbors,
        Z=Z,
        R_Z=R_Z_true,
        R_Z_inv=R_Z_inv_true,
        save_dir=run_dir,
    )

    print("\n[4/6] 正在运行 Empirical-Bayes VI...")
    guide_type = os.environ.get("VI_GUIDE", "flow")
    eb_w_S, eb_beta, eb_sigma_epsilon, eb_params, eb_losses = run_empirical_bayes_vi_inference(
        X_S=X_S,
        y_S=y_S,
        S=S,
        init_params=true_params,
        m=m,
        m_tilde=m_tilde,
        num_iterations=int(os.environ.get("VI_ITERS", "15000")),
        learning_rate=float(os.environ.get("VI_LR", "0.00005")),
        num_particles=int(os.environ.get("VI_PARTICLES", "2")),
        num_vi_samples=int(os.environ.get("VI_SAMPLES", "1000")),
        num_flows=int(os.environ.get("VI_FLOWS", "5")),
        hidden_dims=(256, 256),
        neighbors=neighbors,
        Z=Z,
        L=L_true,
        save_dir=run_dir,
        guide_type=guide_type,
    )
    print_parameter_comparison(true_params, eb_params, est_label="EB")

    print("\n[5/6] 正在生成 HMC 与 EB-VI 的 w_S 后验对比图...")
    compare_posteriors(
        hmc_w_S,
        eb_w_S,
        title="HMC conditional true-parameter posterior vs Empirical-Bayes VI posterior (w_S)",
        save_path=os.path.join(run_dir, "hmc_vs_eb_vi_posterior_comparison.png"),
        vi_label="EB-VI",
    )
    hmc_mean = np.mean(hmc_w_S, axis=0)
    eb_mean = np.mean(eb_w_S, axis=0)
    min_dim = min(len(hmc_mean), len(eb_mean))
    posterior_rmse = np.sqrt(np.mean((hmc_mean[:min_dim] - eb_mean[:min_dim]) ** 2))
    posterior_corr = np.corrcoef(hmc_mean[:min_dim], eb_mean[:min_dim])[0, 1]
    print("注意：这里HMC和EB-VI的target不同，此对比是诊断，不是严格同一posterior误差。")
    print(f"w_S posterior mean diagnostic RMSE: {posterior_rmse:.6f}")
    print(f"w_S posterior mean diagnostic correlation: {posterior_corr:.6f}")

    print("\n正在生成 reference points 上的 y_S 真值 vs fitted y_S 散点图...")
    hmc_train_rmse, hmc_train_mae, hmc_train_corr = plot_training_y_true_vs_fitted(
        y_true=y_S,
        X_S=X_S,
        w_S_samples=hmc_w_S,
        beta_samples=hmc_beta,
        method_name="HMC",
        save_path=os.path.join(run_dir, "true_vs_fitted_yS_hmc.png"),
    )

    eb_train_rmse, eb_train_mae, eb_train_corr = plot_training_y_true_vs_fitted(
        y_true=y_S,
        X_S=X_S,
        w_S_samples=eb_w_S,
        beta_samples=eb_beta,
        method_name="EB-VI",
        save_path=os.path.join(run_dir, "true_vs_fitted_yS_eb_vi.png"),
    )

    print(f"HMC fitted y_S: RMSE={hmc_train_rmse:.6f}, MAE={hmc_train_mae:.6f}, Corr={hmc_train_corr:.6f}")
    print(f"EB-VI fitted y_S: RMSE={eb_train_rmse:.6f}, MAE={eb_train_mae:.6f}, Corr={eb_train_corr:.6f}")

    print("\n正在生成 true w_S vs inferred w_S 散点图...")
    hmc_ws_rmse, hmc_ws_corr = plot_ws_true_vs_inferred(
        true_w_S,
        hmc_w_S,
        method_name="HMC",
        save_path=os.path.join(run_dir, "true_vs_hmc_ws.png"),
    )

    eb_ws_rmse, eb_ws_corr = plot_ws_true_vs_inferred(
        true_w_S,
        eb_w_S,
        method_name="EB-VI",
        save_path=os.path.join(run_dir, "true_vs_eb_vi_ws.png"),
    )

    print(f"HMC true w_S vs posterior mean: RMSE={hmc_ws_rmse:.6f}, Corr={hmc_ws_corr:.6f}")
    print(f"EB-VI true w_S vs posterior mean: RMSE={eb_ws_rmse:.6f}, Corr={eb_ws_corr:.6f}")
    print("\n[6/6] 正在进行预测与性能评估...")
    pred_max_samples_env = os.environ.get("PRED_MAX_SAMPLES", "200")
    pred_max_samples = None if pred_max_samples_env.lower() == "none" else int(pred_max_samples_env)

    print("  运行 HMC 条件金标准预测（使用 true_params）...")
    hmc_pred_mean, hmc_pred_std, hmc_pred_samples = predict(
        hmc_w_S,
        S,
        U,
        X_U,
        hmc_beta,
        hmc_sigma_epsilon,
        true_params,
        m=m,
        Z=Z,
        neighbors_S=neighbors,
        max_samples=pred_max_samples,
        seed=101,
    )

    print("  运行 EB-VI 预测（使用 VI 训练后的 EB 参数，不使用 true_params）...")
    eb_pred_mean, eb_pred_std, eb_pred_samples = predict(
        eb_w_S,
        S,
        U,
        X_U,
        eb_beta,
        eb_sigma_epsilon,
        eb_params,
        m=m,
        Z=Z,
        neighbors_S=neighbors,
        max_samples=pred_max_samples,
        seed=202,
    )

    print("\n" + "=" * 50)
    print("HMC条件金标准预测性能:")
    hmc_metrics = evaluate_predictions(y_U, hmc_pred_mean, hmc_pred_std)
    print("\n" + "=" * 50)
    print("Empirical-Bayes VI预测性能:")
    eb_metrics = evaluate_predictions(y_U, eb_pred_mean, eb_pred_std)

    print("\n正在生成预测结果对比图...")
    plot_prediction_maps(
        S,
        y_S,
        U,
        y_U,
        hmc_pred_mean,
        hmc_pred_std,
        title="HMC Conditional Prediction Results",
        save_path=os.path.join(run_dir, "hmc_conditional_prediction.png"),
    )
    plot_prediction_maps(
        S,
        y_S,
        U,
        y_U,
        eb_pred_mean,
        eb_pred_std,
        title="Empirical-Bayes VI Prediction Results",
        save_path=os.path.join(run_dir, "eb_vi_prediction.png"),
    )
    plot_prediction_performance_comparison(
        y_U,
        hmc_pred_mean,
        eb_pred_mean,
        hmc_metrics,
        eb_metrics,
        save_path=os.path.join(run_dir, "prediction_comparison.png"),
    )

    np.savez_compressed(
        os.path.join(run_dir, "prediction_results.npz"),
        hmc_pred_mean=hmc_pred_mean,
        hmc_pred_std=hmc_pred_std,
        eb_pred_mean=eb_pred_mean,
        eb_pred_std=eb_pred_std,
        y_U=y_U,
        hmc_metrics=hmc_metrics,
        eb_metrics=eb_metrics,
        posterior_rmse=posterior_rmse,
        posterior_corr=posterior_corr,
        eb_matern_params=np.asarray(eb_params["matern_params"]),
        eb_tau_params=np.asarray(eb_params["tau_params"]),
        eb_g_params=np.asarray(eb_params["g_params"]),
        eb_beta=np.asarray(eb_params["beta"]),
        eb_sigma_epsilon=np.asarray(eb_params["sigma_epsilon"]),
    )

    print("\n" + "=" * 70)
    print("🎉 推断测试完成！所有结果已保存至:")
    print(f"  {run_dir}")
    print("\n📊 性能汇总:")
    print(f"  HMC RMSE: {hmc_metrics['rmse']:.4f}")
    print(f"  EB-VI RMSE: {eb_metrics['rmse']:.4f}")
    print(f"  HMC 95% CI覆盖率: {hmc_metrics['coverage']:.4f}")
    print(f"  EB-VI 95% CI覆盖率: {eb_metrics['coverage']:.4f}")
    print(f"  w_S诊断RMSE: {posterior_rmse:.6f}")
    print(f"  w_S诊断相关系数: {posterior_corr:.6f}")
    print("=" * 70)
