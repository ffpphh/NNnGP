"""Run HMC/NUTS inference for NNnGP only.

This file intentionally does NOT import FlowJAX. It is separated from
vi_inference.py to avoid package/import conflicts.

Example:
    JAX_PLATFORM=gpu HMC_WARMUP=5000 HMC_SAMPLES=5000 HMC_THIN=5 python hmc_inference.py
"""

import os
import argparse
import warnings
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PACKAGE_ROOT, "outputs", "core")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_CLIENT_MEM_FRACTION", os.environ.get("XLA_CLIENT_MEM_FRACTION", "0.6"))

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import NUTS, MCMC

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_platform_name", os.environ.get("JAX_PLATFORM", "gpu"))
jax.config.update("jax_debug_nans", False)

from model import (
    nnngp_model,
    compute_sparse_reverse_cholesky,
    build_neighbor_indices,
    predict,
)
from inference_utils import (
    ensure_dir,
    load_synthetic_data,
    make_parent_matern_kernel,
    precompute_RZ,
    z_to_w_samples,
)
from visualization import (
    evaluate_predictions,
    plot_prediction_maps,
    plot_prediction_scatter,
    plot_ws_true_vs_inferred,
    plot_training_y_true_vs_fitted,
)


def run_hmc_inference(
    X_S,
    y_S,
    S,
    true_params,
    m,
    m_tilde,
    L,
    neighbors,
    Z,
    R_Z,
    R_Z_inv,
    num_warmup=5000,
    num_samples=5000,
    thinning=5,
    num_chains=1,
    save_dir="results/hmc_result",
):
    print("\n正在运行 HMC/NUTS 推断（固定真实参数，仅推断 w_S_std / w_S）...")
    if num_warmup < 0 or num_samples <= 0 or thinning <= 0:
        raise ValueError("HMC requires warmup >= 0, samples > 0, and thinning > 0.")
    total_steps = num_warmup + num_samples
    retained_samples = (num_samples + thinning - 1) // thinning
    print(
        f"HMC steps: total={total_steps}, warmup={num_warmup}, "
        f"post-warmup={num_samples}, thin={thinning}, "
        f"retained={retained_samples}"
    )

    kernel = NUTS(
        nnngp_model,
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
        thinning=thinning,
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
    samples = mcmc.get_samples()

    w_S_std_samples = np.asarray(samples["w_S_std"])
    w_S_samples = z_to_w_samples(w_S_std_samples, L)
    beta_samples = np.tile(true_params["beta"], (len(w_S_samples), 1))
    sigma_eps_samples = np.tile(true_params["sigma_epsilon"], (len(w_S_samples),))

    ensure_dir(save_dir)
    out_path = os.path.join(save_dir, "hmc_results.npz")
    np.savez_compressed(
        out_path,
        w_S_samples=w_S_samples,
        w_S_std_samples=w_S_std_samples,
        beta_samples=beta_samples,
        sigma_epsilon_samples=sigma_eps_samples,
    )
    print(f"✅ HMC 推断完成，结果保存到: {out_path}")
    return w_S_samples, beta_samples, sigma_eps_samples


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to test_synthetic_data.npz")
    parser.add_argument("--save-dir", default=os.path.join(RESULTS_DIR, "hmc_result"))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("HMC_WARMUP", "5000")))
    parser.add_argument("--samples", type=int, default=int(os.environ.get("HMC_SAMPLES", "5000")))
    parser.add_argument("--thin", type=int, default=int(os.environ.get("HMC_THIN", "5")))
    parser.add_argument("--pred-max-samples", default=os.environ.get("PRED_MAX_SAMPLES", "200"))
    args = parser.parse_args()

    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(BASE_DIR, args.save_dir)

    print("=" * 70)
    print("NNnGP HMC/NUTS inference")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    data, data_path = load_synthetic_data(args.data)
    print(f"加载数据: {data_path}")

    S = np.asarray(data["S"], dtype=np.float64)
    y_S = np.asarray(data["y_S"], dtype=np.float64).flatten()
    X_S = np.asarray(data["X_S"], dtype=np.float64)
    U = np.asarray(data["U"], dtype=np.float64)
    y_U = np.asarray(data["y_U"], dtype=np.float64).flatten()
    X_U = np.asarray(data["X_U"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    true_w_S = np.asarray(data["w_S"], dtype=np.float64).flatten() if "w_S" in data else None
    m = int(true_params["m"])
    m_tilde = int(true_params["m_tilde"])

    print(f"参考点数量: {len(S)}, 预测点数量: {len(U)}, m={m}, m_tilde={m_tilde}")

    print("\n预计算 R_Z, R_Z_inv, reverse-Cholesky L, neighbors...")
    R_Z_true, R_Z_inv_true = precompute_RZ(Z, true_params["g_params"][2])
    matern_kernel = make_parent_matern_kernel(true_params)
    L_true = compute_sparse_reverse_cholesky(S, matern_kernel, m)
    neighbors = build_neighbor_indices(S, m)

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
        R_Z=R_Z_true,
        R_Z_inv=R_Z_inv_true,
        num_warmup=args.warmup,
        num_samples=args.samples,
        thinning=args.thin,
        save_dir=args.save_dir,
    )

    pred_max = None if str(args.pred_max_samples).lower() == "none" else int(args.pred_max_samples)
    print("\n运行 HMC posterior prediction...")
    pred_mean, pred_std, pred_samples = predict(
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
    )
    metrics = evaluate_predictions(y_U, pred_mean, pred_std, label="HMC")

    np.savez_compressed(
        os.path.join(args.save_dir, "hmc_prediction_results.npz"),
        pred_mean=pred_mean,
        pred_std=pred_std,
        pred_samples=pred_samples,
        y_U=y_U,
        metrics=metrics,
    )

    plot_prediction_maps(
        S, y_S, U, y_U, pred_mean, pred_std,
        title="HMC Conditional Prediction Results",
        save_path=os.path.join(args.save_dir, "hmc_prediction_map.png"),
        include_S=True,
    )
    plot_prediction_scatter(
        y_U, pred_mean, method_name="HMC",
        save_path=os.path.join(args.save_dir, "hmc_true_vs_predicted_yU.png"),
    )
    plot_training_y_true_vs_fitted(
        y_S, X_S, hmc_w_S, hmc_beta, method_name="HMC",
        save_path=os.path.join(args.save_dir, "hmc_true_vs_fitted_yS.png"),
    )
    if true_w_S is not None:
        plot_ws_true_vs_inferred(
            true_w_S, hmc_w_S, method_name="HMC",
            save_path=os.path.join(args.save_dir, "hmc_true_vs_inferred_wS.png"),
        )

    print("\n完成。结果目录:", args.save_dir)


if __name__ == "__main__":
    main()
