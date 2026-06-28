"""
model.py -- fixed NNnGP implementation

Main fixes compared with the original version:
1. Appendix C FIC correction uses R_Z, not identity, in C1.
2. The low-rank correction is computed only on S' = {s_m, ..., s_{k-1}} in 0-based indexing.
3. Non-centered NNGP base prior uses w = T^{-1} z, where T is the sparse reverse-Cholesky
   factor satisfying precision = T.T @ T.
4. Prediction no longer uses the true synthetic g_Z. It uses the marginal predictive
   distribution p(w_U | w_S) based on Proposition 3 / Appendix C in a location-wise form.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import Matern
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", False)

JITTER = 1e-6


def _chol_solve(L, rhs):
    """Solve A x = rhs from a lower Cholesky factor A = L @ L.T."""
    y = jax.scipy.linalg.solve_triangular(L, rhs, lower=True, trans="N")
    return jax.scipy.linalg.solve_triangular(L, y, lower=True, trans="T")


def _chol_logdet(L):
    """Log determinant from a lower Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))


def _matern_jitter_jax(sigma_f):
    """Relative jitter for Matérn covariance systems C = sigma_f^2 R."""
    return (sigma_f**2) * jnp.asarray(JITTER, dtype=jnp.float64)


def _matern_jitter_np(sigma_f):
    """Relative jitter for NumPy Matérn covariance systems C = sigma_f^2 R."""
    return float(sigma_f) ** 2 * JITTER


# ==============================================================================
# Kernels
# ==============================================================================
def g_matern32_kernel_jax(X, Y=None, rho=1.0):
    """JAX Matern 3/2 correlation kernel for the latent nonlinear GP g."""
    X = jnp.asarray(X, dtype=jnp.float64)
    Y = X if Y is None else jnp.asarray(Y, dtype=jnp.float64)
    dists = jnp.sqrt(jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1) + 1e-12)
    scaled = jnp.sqrt(3.0) * dists / (rho + 1e-12)
    return (1.0 + scaled) * jnp.exp(-scaled)


def g_matern32_kernel_np(X, Y=None, rho=1.0):
    """NumPy Matern 3/2 correlation kernel for prediction utilities."""
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    dists = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1) + 1e-12)
    scaled = np.sqrt(3.0) * dists / (rho + 1e-12)
    return (1.0 + scaled) * np.exp(-scaled)


def matern_kernel_jax(X, Y=None, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """JAX Matern 3/2 covariance kernel for the parent GP C."""
    X = jnp.asarray(X, dtype=jnp.float64)
    Y = X if Y is None else jnp.asarray(Y, dtype=jnp.float64)
    dists = jnp.sqrt(jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1) + 1e-12)
    if nu != 1.5:
        raise NotImplementedError("This implementation currently supports only Matern nu=1.5.")
    sqrt3 = jnp.sqrt(3.0)
    return sigma_f**2 * (1.0 + sqrt3 * dists / length_scale) * jnp.exp(-sqrt3 * dists / length_scale)


def matern32_np(X, Y=None, length_scale=1.0, sigma_f=1.0):
    """NumPy Matern 3/2 covariance kernel."""
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    dists = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1) + 1e-12)
    sqrt3 = np.sqrt(3.0)
    return sigma_f**2 * (1.0 + sqrt3 * dists / length_scale) * np.exp(-sqrt3 * dists / length_scale)


def _matern32_from_dists_jax(dists, length_scale=1.0, sigma_f=1.0):
    """JAX Matern 3/2 covariance from precomputed distances."""
    dists = jnp.asarray(dists, dtype=jnp.float64)
    sqrt3 = jnp.sqrt(3.0)
    scaled = sqrt3 * dists / length_scale
    return sigma_f**2 * (1.0 + scaled) * jnp.exp(-scaled)


# ==============================================================================
# NNGP utilities
# ==============================================================================
def maximin_ordering(points):
    """Maximin ordering for reference points."""
    points = np.asarray(points)
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)

    order = [np.random.randint(n)]
    remaining = set(range(n)) - set(order)
    for _ in range(n - 1):
        remaining_list = list(remaining)
        remaining_points = points[remaining_list]
        min_distances = np.min(cdist(remaining_points, points[order]), axis=1)
        next_idx = remaining_list[int(np.argmax(min_distances))]
        order.append(next_idx)
        remaining.remove(next_idx)
    return np.asarray(order, dtype=int)


def find_neighbors(points, m):
    """For each ordered point, find up to m nearest predecessors."""
    points = np.asarray(points)
    n = len(points)
    m = min(int(m), n - 1)
    neighbors = []
    for i in range(n):
        if i == 0:
            neighbors.append(np.array([], dtype=int))
        elif i <= m:
            neighbors.append(np.arange(i, dtype=int))
        else:
            distances = cdist(points[i : i + 1], points[:i])[0]
            neighbors.append(np.argsort(distances)[:m].astype(int))
    return neighbors


def build_neighbor_indices(points, m):
    """
    Fixed-shape neighbor index matrix of shape (k, m).

    Rows i < m are padded by repeats so that the array is JIT-compatible. These
    padded rows are ignored in the NNnGP nonlinear correction, while the exact
    NNGP base density for the first m points is handled by the reverse Cholesky.
    """
    points = np.asarray(points)
    n = len(points)
    m = min(int(m), n - 1)
    neighbors = np.full((n, m), 0, dtype=int)

    for i in range(n):
        if i == 0:
            neighbors[i, :] = 0
        elif i <= m:
            neighbors[i, :i] = np.arange(i)
            neighbors[i, i:] = i - 1
        else:
            distances = cdist(points[i : i + 1], points[:i])[0]
            neighbors[i, :] = np.argsort(distances)[:m]
    return neighbors


def compute_sparse_reverse_cholesky(points, kernel, m, jitter=JITTER):
    """
    Compute the sparse reverse-Cholesky factor T of the NNGP precision.

    The factor T is lower triangular and satisfies precision = T.T @ T. Its row i
    represents the standardized conditional residual
        (w_i - B_i w_N(i)) / sqrt(F_i).
    Therefore, for non-centered sampling z ~ N(0, I), the latent field is
        w = T^{-1} z,
    not T^{-T} z.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    m = min(int(m), n - 1)
    neighbor_list = find_neighbors(points, m)
    T = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        N_i = neighbor_list[i]
        if len(N_i) == 0:
            C_ii = kernel(points[i : i + 1])[0, 0]
            F_i = max(C_ii, jitter)
            T[i, i] = 1.0 / np.sqrt(F_i)
            continue

        C_NN = kernel(points[N_i]) + jitter * np.eye(len(N_i))
        C_iN = kernel(points[i : i + 1], points[N_i])[0]
        C_ii = kernel(points[i : i + 1])[0, 0]

        B_i = np.linalg.solve(C_NN, C_iN).T
        F_i = C_ii - C_iN @ np.linalg.solve(C_NN, C_iN.T)
        F_i = max(float(F_i), jitter)

        T[i, i] = 1.0 / np.sqrt(F_i)
        T[i, N_i] = -B_i / np.sqrt(F_i)

    return jnp.asarray(T, dtype=jnp.float64)


# ==============================================================================
# Appendix C fixed FIC correction
# ==============================================================================
def _reference_terms_jax(w_S, S, neighbors, m, theta_tau1, theta_tau2, theta_g1, theta_g2, sigma_f, length_scale):
    """
    Compute W=tilde{w}_{S'}, tau, F, residual for S' only.

    0-based Python convention: S' corresponds to indices i = m, ..., k-1.
    """
    w_S = jnp.asarray(w_S, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    k = w_S.shape[0]
    m_static = neighbors.shape[1]
    idx = jnp.arange(m_static, k)

    def one_point(i):
        N_i = neighbors[i]
        w_N = w_S[N_i]
        S_N = S[N_i]
        dists = jnp.linalg.norm(S[i] - S_N, axis=1)
        min_dist = jnp.min(dists)

        tau_i = jnp.sqrt(jnp.exp(theta_tau1) * jnp.power(min_dist + 1e-12, theta_tau2))
        lambda_diag = jnp.exp(theta_g1 + theta_g2 * dists)
        w_tilde_i = jnp.sqrt(lambda_diag) * w_N

        matern_jitter = _matern_jitter_jax(sigma_f)
        C_NN = matern_kernel_jax(S_N, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * jnp.eye(S_N.shape[0])
        C_iN = matern_kernel_jax(S[i][None, :], S_N, length_scale=length_scale, sigma_f=sigma_f)[0]
        C_NN_chol = jnp.linalg.cholesky(C_NN)
        alpha_c = _chol_solve(C_NN_chol, C_iN)
        h_i = jnp.dot(alpha_c, w_N)
        F_i = matern_kernel_jax(S[i][None, :], length_scale=length_scale, sigma_f=sigma_f)[0, 0] - jnp.dot(C_iN, alpha_c)
        F_i = jnp.maximum(F_i, matern_jitter)
        residual_i = w_S[i] - h_i
        return w_tilde_i, tau_i, F_i, residual_i

    return jax.vmap(one_point)(idx)


def precompute_fic_reference_geometry(S, neighbors, m, sigma_f, length_scale):
    """
    Precompute reference-neighbor quantities that are fixed when the base kernel is fixed.

    These are the expensive small Matérn solves in Appendix C.2. In fixed-base VI
    they do not change across ELBO steps.
    """
    S = np.asarray(S, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=int)
    m = min(int(m), len(S) - 1)
    idx = np.arange(m, len(S), dtype=int)
    N_ref = neighbors[idx]
    nW = len(idx)

    B_ref = np.zeros((nW, m), dtype=np.float64)
    F_ref = np.zeros(nW, dtype=np.float64)
    dist_ref = np.zeros((nW, m), dtype=np.float64)

    for row, i in enumerate(idx):
        N_i = N_ref[row]
        S_N = S[N_i]
        dist_ref[row] = np.linalg.norm(S[i] - S_N, axis=1)

        matern_jitter = _matern_jitter_np(sigma_f)
        C_NN = matern32_np(S_N, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * np.eye(m)
        C_iN = matern32_np(S[i : i + 1], S_N, length_scale=length_scale, sigma_f=sigma_f)[0]
        C_NN_chol = np.linalg.cholesky(C_NN)
        y = np.linalg.solve(C_NN_chol, C_iN)
        B_i = np.linalg.solve(C_NN_chol.T, y)
        F_i = matern32_np(S[i : i + 1], length_scale=length_scale, sigma_f=sigma_f)[0, 0] - C_iN @ B_i

        B_ref[row] = B_i
        F_ref[row] = max(float(F_i), matern_jitter)

    return idx, N_ref, B_ref, F_ref, dist_ref


def precompute_eb_reference_geometry(S, neighbors, m):
    """
    Precompute geometry for all-EB VI.

    Hyperparameters still change, so we cannot precompute B_i/F_i. We can still
    precompute all pairwise distances used by the repeated Matérn neighbor
    systems and by tau/lambda in Appendix C.2.
    """
    S = np.asarray(S, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=int)
    m = min(int(m), len(S) - 1)

    S0 = S[:m]
    base0_matern_dists = np.sqrt(np.sum((S0[:, None, :] - S0[None, :, :]) ** 2, axis=-1) + 1e-12)

    idx = np.arange(m, len(S), dtype=int)
    N_ref = neighbors[idx]
    nW = len(idx)

    matern_NN_dists = np.zeros((nW, m, m), dtype=np.float64)
    matern_iN_dists = np.zeros((nW, m), dtype=np.float64)
    euclid_iN_dists = np.zeros((nW, m), dtype=np.float64)

    for row, i in enumerate(idx):
        N_i = N_ref[row]
        S_N = S[N_i]
        matern_NN_dists[row] = np.sqrt(np.sum((S_N[:, None, :] - S_N[None, :, :]) ** 2, axis=-1) + 1e-12)
        matern_iN_dists[row] = np.sqrt(np.sum((S[i][None, :] - S_N) ** 2, axis=-1) + 1e-12)
        euclid_iN_dists[row] = np.linalg.norm(S[i] - S_N, axis=1)

    return base0_matern_dists, idx, N_ref, matern_NN_dists, matern_iN_dists, euclid_iN_dists


def _reference_terms_precomputed_jax(
    w_S,
    ref_idx,
    ref_N,
    ref_B,
    ref_F,
    ref_dists,
    theta_tau1,
    theta_tau2,
    theta_g1,
    theta_g2,
):
    """Compute W, tau, F and residual using precomputed fixed-base geometry."""
    w_N = w_S[ref_N]
    min_dists = jnp.min(ref_dists, axis=1)
    tau = jnp.sqrt(jnp.exp(theta_tau1) * jnp.power(min_dists + 1e-12, theta_tau2))
    lambda_diag = jnp.exp(theta_g1 + theta_g2 * ref_dists)
    W = jnp.sqrt(lambda_diag) * w_N
    h = jnp.sum(ref_B * w_N, axis=1)
    residual = w_S[ref_idx] - h
    return W, tau, ref_F, residual


def _base_and_reference_terms_eb_precomputed_jax(
    w_S,
    base0_matern_dists,
    ref_idx,
    ref_N,
    ref_matern_NN_dists,
    ref_matern_iN_dists,
    ref_euclid_iN_dists,
    sigma_f,
    length_scale,
    theta_tau1,
    theta_tau2,
    theta_g1,
    theta_g2,
):
    """
    Compute NNGP base log probability and Appendix C reference terms together.

    This removes the duplicate per-location Matérn neighbor solves that would
    otherwise happen in _nnngp_base_logprob_jax() and _reference_terms_jax().
    """
    m_static = ref_N.shape[1]
    eye_m = jnp.eye(m_static, dtype=jnp.float64)
    matern_jitter = _matern_jitter_jax(sigma_f)

    Cmm = _matern32_from_dists_jax(base0_matern_dists, length_scale=length_scale, sigma_f=sigma_f)
    Cmm = Cmm + matern_jitter * jnp.eye(base0_matern_dists.shape[0], dtype=jnp.float64)
    logp0 = _mvn0_logpdf_cholesky(w_S[:base0_matern_dists.shape[0]], Cmm)
    C_ii = sigma_f**2

    def one_point(i, N_i, d_NN, d_iN_matern, d_iN_euclid):
        w_N = w_S[N_i]
        C_NN = _matern32_from_dists_jax(d_NN, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * eye_m
        C_iN = _matern32_from_dists_jax(d_iN_matern, length_scale=length_scale, sigma_f=sigma_f)

        chol_NN = jnp.linalg.cholesky(C_NN)
        rhs = jnp.stack([w_N, C_iN], axis=1)
        solved = _chol_solve(chol_NN, rhs)
        alpha_w = solved[:, 0]
        alpha_c = solved[:, 1]

        h_i = jnp.dot(C_iN, alpha_w)
        F_i = C_ii - jnp.dot(C_iN, alpha_c)
        F_i = jnp.maximum(F_i, matern_jitter)
        residual_i = w_S[i] - h_i
        logp_i = -0.5 * (jnp.log(2.0 * jnp.pi * F_i) + residual_i * residual_i / F_i)

        tau_i = jnp.sqrt(jnp.exp(theta_tau1) * jnp.power(jnp.min(d_iN_euclid) + 1e-12, theta_tau2))
        lambda_diag = jnp.exp(theta_g1 + theta_g2 * d_iN_euclid)
        w_tilde_i = jnp.sqrt(lambda_diag) * w_N
        return logp_i, w_tilde_i, tau_i, F_i, residual_i

    logp_rest, W, tau, F, residual = jax.vmap(one_point)(
        ref_idx,
        ref_N,
        ref_matern_NN_dists,
        ref_matern_iN_dists,
        ref_euclid_iN_dists,
    )
    return logp0 + jnp.sum(logp_rest), W, tau, F, residual


def _fic_correction_from_reference_terms(
    W,
    tau,
    F,
    residual,
    Z,
    rho,
    R_Z,
    R_Z_chol,
    correction_jitter=JITTER,
):
    """Appendix C.2 low-rank FIC correction once W/tau/F/residual are available."""
    nW = W.shape[0]
    m_tilde = Z.shape[0]
    correction_jitter = jnp.asarray(correction_jitter, dtype=jnp.float64)
    if nW == 0:
        return jnp.array(0.0, dtype=jnp.float64)

    R_WZ = g_matern32_kernel_jax(W, Z, rho)  # (k-m, m_tilde)
    R_ZW = R_WZ.T

    # D1 = I - diag(Q_W), Q_W = R_WZ R_Z^{-1} R_ZW
    RZ_solve_ZW = _chol_solve(R_Z_chol, R_ZW)  # R_Z^{-1} R_ZW
    diag_Q = jnp.sum(RZ_solve_ZW.T * R_WZ, axis=1)
    D1_diag = jnp.maximum(1.0 - diag_Q, correction_jitter)
    D1_inv_diag = 1.0 / D1_diag

    # D2_inv = D_tau D_F^{-1} D_tau = tau^2 / F
    D2_inv_diag = tau**2 / F
    D3_inv_diag = D1_inv_diag + D2_inv_diag
    D3_diag = 1.0 / jnp.maximum(D3_inv_diag, correction_jitter)

    # C1 = R_Z + R_ZW D1^{-1} R_WZ
    C1 = R_Z + (R_ZW * D1_inv_diag[None, :]) @ R_WZ
    C1 = C1 + correction_jitter * jnp.eye(m_tilde, dtype=jnp.float64)

    # C2 = C1 - R_ZW D1^{-1} D3 D1^{-1} R_WZ
    diag_D = D1_inv_diag * D3_diag * D1_inv_diag
    C2 = C1 - (R_ZW * diag_D[None, :]) @ R_WZ
    C2 = C2 + correction_jitter * jnp.eye(m_tilde, dtype=jnp.float64)

    C2_chol = jnp.linalg.cholesky(C2)
    logdet_RZ = _chol_logdet(R_Z_chol)
    logdet_C2 = _chol_logdet(C2_chol)
    logdet_D1 = jnp.sum(jnp.log(D1_diag))
    logdet_D3 = jnp.sum(jnp.log(D3_diag))

    # a = D_tau D_F^{-1} (w_S' - h_S')
    a = tau / F * residual

    # a.T G a with G = D3 + (D3 D1^{-1} R_WZ) C2^{-1} (R_ZW D1^{-1} D3)
    a_D3_a = jnp.sum(a * a * D3_diag)
    t = R_ZW @ (D1_inv_diag * D3_diag * a)
    x = _chol_solve(C2_chol, t)
    quadratic = a_D3_a + jnp.dot(t, x)

    log_det_G_minus_log_det_Rtilde = logdet_D3 - logdet_C2 - logdet_D1 + logdet_RZ
    correction = 0.5 * (log_det_G_minus_log_det_Rtilde + quadratic)
    return correction


def compute_fic_correction(
    w_S,
    S,
    neighbors,
    m,
    Z,
    R_Z_inv,
    theta_tau1,
    theta_tau2,
    theta_g1,
    theta_g2,
    rho,
    sigma_f,
    length_scale,
    R_Z=None,
    R_Z_chol=None,
    correction_jitter=JITTER,
):
    """
    Appendix C.2 low-rank FIC correction for the NNnGP marginal density.

    Returns
        0.5 * (log|G| - log|R_tilde_W| + a.T @ G @ a)

    This implementation follows the non-whitened formula:
        D1 = I - diag(R_WZ R_Z^{-1} R_ZW)
        C1 = R_Z + R_ZW D1^{-1} R_WZ
        D3^{-1} = D1^{-1} + D_tau D_F^{-1} D_tau
        C2 = C1 - R_ZW D1^{-1} D3 D1^{-1} R_WZ
        log|G| - log|R_tilde_W| = log|D3| - log|C2| - log|D1| + log|R_Z|
    """
    w_S = jnp.asarray(w_S, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    correction_jitter = jnp.asarray(correction_jitter, dtype=jnp.float64)

    if R_Z is None:
        R_Z = g_matern32_kernel_jax(Z, Z, rho) + correction_jitter * jnp.eye(Z.shape[0], dtype=jnp.float64)
    else:
        R_Z = jnp.asarray(R_Z, dtype=jnp.float64)
    if R_Z_chol is None:
        R_Z_chol = jnp.linalg.cholesky(R_Z)
    else:
        R_Z_chol = jnp.asarray(R_Z_chol, dtype=jnp.float64)

    W, tau, F, residual = _reference_terms_jax(
        w_S, S, neighbors, m, theta_tau1, theta_tau2, theta_g1, theta_g2, sigma_f, length_scale
    )
    return _fic_correction_from_reference_terms(
        W,
        tau,
        F,
        residual,
        Z,
        rho,
        R_Z,
        R_Z_chol,
        correction_jitter=correction_jitter,
    )


def compute_fic_correction_precomputed(
    w_S,
    Z,
    theta_tau1,
    theta_tau2,
    theta_g1,
    theta_g2,
    rho,
    ref_idx,
    ref_N,
    ref_B,
    ref_F,
    ref_dists,
    R_Z=None,
    R_Z_chol=None,
    correction_jitter=JITTER,
):
    """Appendix C.2 correction using fixed-base precomputed reference geometry."""
    w_S = jnp.asarray(w_S, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    correction_jitter = jnp.asarray(correction_jitter, dtype=jnp.float64)
    ref_idx = jnp.asarray(ref_idx, dtype=jnp.int32)
    ref_N = jnp.asarray(ref_N, dtype=jnp.int32)
    ref_B = jnp.asarray(ref_B, dtype=jnp.float64)
    ref_F = jnp.asarray(ref_F, dtype=jnp.float64)
    ref_dists = jnp.asarray(ref_dists, dtype=jnp.float64)

    if R_Z is None:
        R_Z = g_matern32_kernel_jax(Z, Z, rho) + correction_jitter * jnp.eye(Z.shape[0], dtype=jnp.float64)
    else:
        R_Z = jnp.asarray(R_Z, dtype=jnp.float64)
    if R_Z_chol is None:
        R_Z_chol = jnp.linalg.cholesky(R_Z)
    else:
        R_Z_chol = jnp.asarray(R_Z_chol, dtype=jnp.float64)

    W, tau, F, residual = _reference_terms_precomputed_jax(
        w_S,
        ref_idx,
        ref_N,
        ref_B,
        ref_F,
        ref_dists,
        theta_tau1,
        theta_tau2,
        theta_g1,
        theta_g2,
    )
    return _fic_correction_from_reference_terms(
        W,
        tau,
        F,
        residual,
        Z,
        rho,
        R_Z,
        R_Z_chol,
        correction_jitter=correction_jitter,
    )


def compute_eb_base_and_fic_correction_precomputed(
    w_S,
    Z,
    sigma_f,
    length_scale,
    theta_tau1,
    theta_tau2,
    theta_g1,
    theta_g2,
    rho,
    base0_matern_dists,
    ref_idx,
    ref_N,
    ref_matern_NN_dists,
    ref_matern_iN_dists,
    ref_euclid_iN_dists,
    R_Z=None,
    R_Z_chol=None,
    correction_jitter=JITTER,
):
    """Combined all-EB NNGP base logprob and Appendix C.2 FIC correction."""
    w_S = jnp.asarray(w_S, dtype=jnp.float64)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    correction_jitter = jnp.asarray(correction_jitter, dtype=jnp.float64)
    base0_matern_dists = jnp.asarray(base0_matern_dists, dtype=jnp.float64)
    ref_idx = jnp.asarray(ref_idx, dtype=jnp.int32)
    ref_N = jnp.asarray(ref_N, dtype=jnp.int32)
    ref_matern_NN_dists = jnp.asarray(ref_matern_NN_dists, dtype=jnp.float64)
    ref_matern_iN_dists = jnp.asarray(ref_matern_iN_dists, dtype=jnp.float64)
    ref_euclid_iN_dists = jnp.asarray(ref_euclid_iN_dists, dtype=jnp.float64)

    if R_Z is None:
        R_Z = g_matern32_kernel_jax(Z, Z, rho) + correction_jitter * jnp.eye(Z.shape[0], dtype=jnp.float64)
    else:
        R_Z = jnp.asarray(R_Z, dtype=jnp.float64)
    if R_Z_chol is None:
        R_Z_chol = jnp.linalg.cholesky(R_Z)
    else:
        R_Z_chol = jnp.asarray(R_Z_chol, dtype=jnp.float64)

    logp_base, W, tau, F, residual = _base_and_reference_terms_eb_precomputed_jax(
        w_S,
        base0_matern_dists,
        ref_idx,
        ref_N,
        ref_matern_NN_dists,
        ref_matern_iN_dists,
        ref_euclid_iN_dists,
        sigma_f,
        length_scale,
        theta_tau1,
        theta_tau2,
        theta_g1,
        theta_g2,
    )
    correction = _fic_correction_from_reference_terms(
        W,
        tau,
        F,
        residual,
        Z,
        rho,
        R_Z,
        R_Z_chol,
        correction_jitter=correction_jitter,
    )
    return logp_base, correction


# ==============================================================================
# NumPy state for prediction based on Proposition 3 / Appendix C
# ==============================================================================
def _reference_state_np(w_S, S, neighbors, m, Z, true_params):
    """Build the low-rank state needed for location-wise p(w_U | w_S)."""
    w_S = np.asarray(w_S, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=int)
    Z = np.asarray(Z, dtype=np.float64)

    sigma_f, length_scale = true_params["matern_params"]
    theta_tau1, theta_tau2 = true_params["tau_params"]
    theta_g1, theta_g2, rho = true_params["g_params"]

    idx = np.arange(m, len(S))
    nW = len(idx)
    if nW == 0:
        raise ValueError("Need len(S) > m for NNnGP prediction.")

    W = np.zeros((nW, m), dtype=np.float64)
    tau = np.zeros(nW, dtype=np.float64)
    F = np.zeros(nW, dtype=np.float64)
    residual = np.zeros(nW, dtype=np.float64)

    for row, i in enumerate(idx):
        N_i = neighbors[i]
        S_N = S[N_i]
        w_N = w_S[N_i]
        dists = np.linalg.norm(S[i] - S_N, axis=1)
        tau[row] = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        lambda_diag = np.exp(theta_g1 + theta_g2 * dists)
        W[row] = np.sqrt(lambda_diag) * w_N

        matern_jitter = _matern_jitter_np(sigma_f)
        C_NN = matern32_np(S_N, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * np.eye(m)
        C_iN = matern32_np(S[i : i + 1], S_N, length_scale=length_scale, sigma_f=sigma_f)[0]
        B_i = np.linalg.solve(C_NN, C_iN).T
        h_i = B_i @ w_N
        F_i = matern32_np(S[i : i + 1], length_scale=length_scale, sigma_f=sigma_f)[0, 0] - C_iN @ np.linalg.solve(C_NN, C_iN.T)
        F[row] = max(float(F_i), matern_jitter)
        residual[row] = w_S[i] - h_i

    R_Z = g_matern32_kernel_np(Z, Z, rho) + JITTER * np.eye(len(Z))
    R_Z_inv = np.linalg.inv(R_Z)
    R_WZ = g_matern32_kernel_np(W, Z, rho)
    R_ZW = R_WZ.T

    diag_Q = np.sum((R_WZ @ R_Z_inv) * R_WZ, axis=1)
    D1 = np.maximum(1.0 - diag_Q, JITTER)
    D1_inv = 1.0 / D1
    D2_inv = tau**2 / F
    D3 = 1.0 / np.maximum(D1_inv + D2_inv, JITTER)

    C1 = R_Z + (R_ZW * D1_inv[None, :]) @ R_WZ
    C1 = C1 + JITTER * np.eye(len(Z))
    C1_inv = np.linalg.inv(C1)

    C2 = C1 - (R_ZW * (D1_inv * D3 * D1_inv)[None, :]) @ R_WZ
    C2 = C2 + JITTER * np.eye(len(Z))
    C2_inv = np.linalg.inv(C2)

    a = tau / F * residual
    t = R_ZW @ (D1_inv * D3 * a)
    Ga = D3 * a + (D3 * D1_inv) * (R_WZ @ (C2_inv @ t))

    return {
        "W": W,
        "Z": Z,
        "rho": rho,
        "R_Z_inv": R_Z_inv,
        "R_WZ": R_WZ,
        "R_ZW": R_ZW,
        "D1_inv": D1_inv,
        "D3": D3,
        "C1_inv": C1_inv,
        "C2_inv": C2_inv,
        "Ga": Ga,
        "sigma_f": sigma_f,
        "length_scale": length_scale,
        "theta_tau1": theta_tau1,
        "theta_tau2": theta_tau2,
        "theta_g1": theta_g1,
        "theta_g2": theta_g2,
    }


def _precompute_prediction_geometry(S, U, m, true_params):
    """Precompute prediction neighbors and location-only terms."""
    S = np.asarray(S, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    sigma_f, length_scale = true_params["matern_params"]
    theta_tau1, theta_tau2 = true_params["tau_params"]
    theta_g1, theta_g2, _ = true_params["g_params"]

    r = len(U)
    neighbor_idx = np.zeros((r, m), dtype=int)
    B = np.zeros((r, m), dtype=np.float64)
    F = np.zeros(r, dtype=np.float64)
    tau = np.zeros(r, dtype=np.float64)
    sqrt_lambda = np.zeros((r, m), dtype=np.float64)

    for j in range(r):
        d_all = cdist(U[j : j + 1], S)[0]
        N_j = np.argsort(d_all)[:m]
        neighbor_idx[j] = N_j
        S_N = S[N_j]
        dists = d_all[N_j]

        matern_jitter = _matern_jitter_np(sigma_f)
        C_NN = matern32_np(S_N, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * np.eye(m)
        C_uN = matern32_np(U[j : j + 1], S_N, length_scale=length_scale, sigma_f=sigma_f)[0]
        B[j] = np.linalg.solve(C_NN, C_uN).T
        F_j = matern32_np(U[j : j + 1], length_scale=length_scale, sigma_f=sigma_f)[0, 0] - C_uN @ np.linalg.solve(C_NN, C_uN.T)
        F[j] = max(float(F_j), matern_jitter)
        tau[j] = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda[j] = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))

    return neighbor_idx, B, F, tau, sqrt_lambda


def _predict_one_location_np(w_S, geom_row, state, rng):
    """Sample one latent w_u from the location-wise marginal p(w_u | w_S)."""
    N_j, B_j, F_j, tau_j, sqrt_lambda_j = geom_row
    w_N = w_S[N_j]
    h_j = B_j @ w_N
    v_j = sqrt_lambda_j * w_N

    R_vZ = g_matern32_kernel_np(v_j[None, :], state["Z"], state["rho"])[0]
    # Cross covariance under FIC for v not in W: q_vW = R_vZ R_Z^{-1} R_ZW
    q_vW = R_vZ @ state["R_Z_inv"] @ state["R_ZW"]

    # b = q_vW R_tilde_W^{-1}
    y = state["D1_inv"] * q_vW
    tmp = state["C1_inv"] @ (state["R_ZW"] @ y)
    b = y - state["D1_inv"] * (state["R_WZ"] @ tmp)

    nonlinear_mean = tau_j * (b @ state["Ga"])

    # R_{v|W}: use FIC diagonal R_tilde(v,v)=R(v,v)=1.
    R_v_given_W = 1.0 - q_vW @ b
    R_v_given_W = max(float(R_v_given_W), JITTER)

    # b G b.T using low-rank form of G.
    b_D3_b = np.sum(b * b * state["D3"])
    tb = state["R_ZW"] @ (state["D1_inv"] * state["D3"] * b)
    bGb = b_D3_b + tb @ (state["C2_inv"] @ tb)
    bGb = max(float(bGb), 0.0)

    var_w = F_j + tau_j**2 * (R_v_given_W + bGb)
    var_w = max(float(var_w), JITTER)
    mean_w = h_j + nonlinear_mean
    return rng.normal(mean_w, np.sqrt(var_w))


# ==============================================================================
# NumPyro model
# ==============================================================================
def nnngp_model(
    X_S,
    y_S,
    S,
    m=20,
    m_tilde=10,
    use_true_params=False,
    true_params=None,
    L=None,
    neighbors=None,
    Z=None,
    R_Z_inv=None,
    R_Z=None,
):
    """NNnGP model with Appendix C FIC correction."""
    X_S = jnp.asarray(X_S, dtype=jnp.float64)
    y_S = jnp.asarray(y_S, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    n = y_S.shape[0]

    if true_params is not None:
        sigma_f = float(true_params["matern_params"][0])
        length_scale = float(true_params["matern_params"][1])
        theta_tau1 = float(true_params["tau_params"][0])
        theta_tau2 = float(true_params["tau_params"][1])
        theta_g1 = float(true_params["g_params"][0])
        theta_g2 = float(true_params["g_params"][1])
        rho = float(true_params["g_params"][2])
        beta = jnp.asarray(true_params["beta"], dtype=jnp.float64)
        sigma_epsilon = float(true_params["sigma_epsilon"])
    else:
        sigma_f = numpyro.sample("sigma_f", dist.HalfNormal(1.0))
        length_scale = numpyro.sample("length_scale", dist.HalfNormal(0.5))
        theta_tau1 = numpyro.sample("theta_tau1", dist.Normal(0.0, 1.0))
        theta_tau2 = numpyro.sample("theta_tau2", dist.HalfNormal(1.0))
        theta_g1 = numpyro.sample("theta_g1", dist.Normal(0.0, 1.0))
        # Screening requires theta_g2 < 0; use a negative half-normal.
        theta_g2 = -numpyro.sample("neg_theta_g2", dist.HalfNormal(1.0))
        rho = jnp.asarray(1.0, dtype=jnp.float64)
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([X_S.shape[1]]).to_event(1))
        sigma_epsilon = numpyro.sample("sigma_epsilon", dist.HalfNormal(0.5))

    if L is None:
        raise ValueError("L / reverse Cholesky factor must be precomputed and passed in.")
    if neighbors is None:
        raise ValueError("neighbors must be precomputed and passed in.")
    if Z is None:
        raise ValueError("FIC inducing points Z must be passed in.")

    L = jnp.asarray(L, dtype=jnp.float64)
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    Z = jnp.asarray(Z, dtype=jnp.float64)

    if R_Z is None:
        R_Z = g_matern32_kernel_jax(Z, Z, rho) + JITTER * jnp.eye(Z.shape[0], dtype=jnp.float64)
    else:
        R_Z = jnp.asarray(R_Z, dtype=jnp.float64)
    R_Z_chol = jnp.linalg.cholesky(R_Z)
    if R_Z_inv is not None:
        R_Z_inv = jnp.asarray(R_Z_inv, dtype=jnp.float64)

    # Non-centered base NNGP: z = T w, so w = T^{-1} z.
    w_S_std = numpyro.sample("w_S_std", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    w_S = jax.scipy.linalg.solve_triangular(L, w_S_std, lower=True, trans="N")

    correction = compute_fic_correction(
        w_S,
        S,
        neighbors,
        m,
        Z,
        None,
        theta_tau1,
        theta_tau2,
        theta_g1,
        theta_g2,
        rho,
        sigma_f,
        length_scale,
        R_Z=R_Z,
        R_Z_chol=R_Z_chol,
    )
    numpyro.factor("non_gaussian_fic_correction", correction)

    mu = X_S @ beta + w_S
    numpyro.sample("y_S", dist.Normal(mu, jnp.maximum(sigma_epsilon, JITTER)).to_event(1), obs=y_S)


# ==============================================================================
# Empirical-Bayes NNnGP model for VI
# ==============================================================================
def _mvn0_logpdf_cholesky(x, K):
    """Log density of N(0, K) using Cholesky."""
    Lc = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.solve_triangular(Lc, x, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(Lc)))
    n = x.shape[0]
    return -0.5 * (n * jnp.log(2.0 * jnp.pi) + logdet + jnp.dot(alpha, alpha))


def _nnngp_base_logprob_jax(w_S, S, neighbors, sigma_f, length_scale):
    """Evaluate p_NNGP(w_S | sigma_f, length_scale) via Vecchia factors."""
    w_S = jnp.asarray(w_S, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    k = w_S.shape[0]
    m_static = neighbors.shape[1]

    S0 = S[:m_static]
    w0 = w_S[:m_static]
    matern_jitter = _matern_jitter_jax(sigma_f)
    Cmm = matern_kernel_jax(S0, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * jnp.eye(m_static, dtype=jnp.float64)
    logp0 = _mvn0_logpdf_cholesky(w0, Cmm)

    def process_point(i):
        N_i = neighbors[i]
        w_N = w_S[N_i]
        S_N = S[N_i]
        C_NN = matern_kernel_jax(S_N, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * jnp.eye(m_static, dtype=jnp.float64)
        C_iN = matern_kernel_jax(S[i][None, :], S_N, length_scale=length_scale, sigma_f=sigma_f)[0]
        C_ii = matern_kernel_jax(S[i][None, :], length_scale=length_scale, sigma_f=sigma_f)[0, 0]
        C_NN_chol = jnp.linalg.cholesky(C_NN)
        rhs = jnp.stack([w_N, C_iN], axis=1)
        solved = _chol_solve(C_NN_chol, rhs)
        alpha_w = solved[:, 0]
        alpha_c = solved[:, 1]
        h_i = jnp.dot(C_iN, alpha_w)
        F_i = C_ii - jnp.dot(C_iN, alpha_c)
        F_i = jnp.maximum(F_i, matern_jitter)
        resid = w_S[i] - h_i
        return -0.5 * (jnp.log(2.0 * jnp.pi * F_i) + resid * resid / F_i)

    idx = jnp.arange(m_static, k)
    logp_rest = jnp.sum(jax.vmap(process_point)(idx))
    return logp0 + logp_rest

def _standard_normal_logprob_vec(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    n = x.shape[0]
    return -0.5 * (n * jnp.log(2.0 * jnp.pi) + jnp.sum(x * x))


def _init_from_dict(init_params, key, default):
    """Read a scalar init value from the user's true_params-style dict."""
    if init_params is None:
        return default
    if key == "sigma_f":
        return float(init_params["matern_params"][0])
    if key == "length_scale":
        return float(init_params["matern_params"][1])
    if key == "theta_tau1":
        return float(init_params["tau_params"][0])
    if key == "theta_tau2":
        return float(init_params["tau_params"][1])
    if key == "theta_g1":
        return float(init_params["g_params"][0])
    if key == "theta_g2":
        return float(init_params["g_params"][1])
    if key == "rho":
        return float(init_params["g_params"][2])
    if key == "sigma_epsilon":
        return float(init_params["sigma_epsilon"])
    raise KeyError(key)


def nnngp_model_empirical_bayes(
    X_S,
    y_S,
    S,
    m=20,
    m_tilde=10,
    init_params=None,
    neighbors=None,
    Z=None,
    L=None,
):
    """
    Empirical-Bayes VI model for NNnGP.

    Random variable:
        w_S, approximated by a VI guide.

    Optimized EB parameters via numpyro.param:
        sigma_f, length_scale, theta_tau1, theta_tau2, theta_g1, theta_g2,
        beta, sigma_epsilon. The g-kernel range rho is fixed.

    The auxiliary sample site w_S has a standard-normal base density. We add
    p_NNnGP(w_S; theta) - N(w_S;0,I) as a factor, so the effective prior is the
    NNnGP prior under the current EB parameters.
    """
    X_S = jnp.asarray(X_S, dtype=jnp.float64)
    y_S = jnp.asarray(y_S, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    if neighbors is None:
        raise ValueError("neighbors must be precomputed and passed in.")
    if Z is None:
        raise ValueError("FIC inducing points Z must be passed in.")
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    n = y_S.shape[0]
    p = X_S.shape[1]

    sigma_f = numpyro.param(
        "eb_sigma_f",
        jnp.asarray(_init_from_dict(init_params, "sigma_f", 1.0), dtype=jnp.float64),
        constraint=dist.constraints.positive,
    )
    length_scale = numpyro.param(
        "eb_length_scale",
        jnp.asarray(_init_from_dict(init_params, "length_scale", 0.2), dtype=jnp.float64),
        constraint=dist.constraints.positive,
    )
    theta_tau1 = numpyro.param("eb_theta_tau1", jnp.asarray(_init_from_dict(init_params, "theta_tau1", 0.0), dtype=jnp.float64))
    theta_tau2 = numpyro.param(
        "eb_theta_tau2",
        jnp.asarray(_init_from_dict(init_params, "theta_tau2", 1.0), dtype=jnp.float64),
        constraint=dist.constraints.positive,
    )
    theta_g1 = numpyro.param("eb_theta_g1", jnp.asarray(_init_from_dict(init_params, "theta_g1", 0.0), dtype=jnp.float64))
    init_theta_g2 = _init_from_dict(init_params, "theta_g2", -2.0)
    neg_theta_g2 = numpyro.param(
        "eb_neg_theta_g2",
        jnp.asarray(max(-init_theta_g2, 1e-3), dtype=jnp.float64),
        constraint=dist.constraints.positive,
    )
    theta_g2 = -neg_theta_g2
    rho = jnp.asarray(_init_from_dict(init_params, "rho", 1.0), dtype=jnp.float64)
    if init_params is not None:
        beta_init = jnp.asarray(init_params["beta"], dtype=jnp.float64)
        if beta_init.shape[0] != p:
            beta_init = jnp.zeros((p,), dtype=jnp.float64)
    else:
        beta_init = jnp.zeros((p,), dtype=jnp.float64)
    beta = numpyro.param("eb_beta", beta_init)
    sigma_epsilon = numpyro.param(
        "eb_sigma_epsilon",
        jnp.asarray(_init_from_dict(init_params, "sigma_epsilon", 0.1), dtype=jnp.float64),
        constraint=dist.constraints.positive,
    )
    #fixed sigma = 0.1
#     sigma_epsilon = jnp.asarray(
#     _init_from_dict(init_params, "sigma_epsilon", 0.1),
#     dtype=jnp.float32
# )

    # The VI guide approximates this site. Its base density is corrected below.
    # w_S = numpyro.sample("w_S", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    if L is None:
        raise ValueError("L / reverse Cholesky factor must be passed for non-centered EB-VI.")

    L = jnp.asarray(L, dtype=jnp.float64)

    # Non-centered VI variable: z = T w, so w = T^{-1} z
    w_S_std = numpyro.sample(
        "w_S_std",
        dist.Normal(0.0, 1.0).expand([n]).to_event(1)
    )

    w_S = jax.scipy.linalg.solve_triangular(
        L,
        w_S_std,
        lower=True,
        trans="N"
    )

    R_Z = g_matern32_kernel_jax(Z, Z, rho) + JITTER * jnp.eye(Z.shape[0], dtype=jnp.float64)
    R_Z_chol = jnp.linalg.cholesky(R_Z)
    correction = compute_fic_correction(
        w_S,
        S,
        neighbors,
        m,
        Z,
        None,
        theta_tau1,
        theta_tau2,
        theta_g1,
        theta_g2,
        rho,
        sigma_f,
        length_scale,
        R_Z=R_Z,
        R_Z_chol=R_Z_chol,
    )
    numpyro.factor("non_gaussian_fic_correction", correction)

    mu = X_S @ beta + w_S
    numpyro.sample("y_S", dist.Normal(mu, jnp.maximum(sigma_epsilon, JITTER)).to_event(1), obs=y_S)


def extract_empirical_bayes_params(svi_params, fallback_params=None):
    """Convert optimized EB params to the true_params-style dict used by predict()."""
    def _get(name):
        if name not in svi_params:
            raise KeyError(f"Missing optimized parameter {name!r} in svi_result.params")
        return np.asarray(jax.device_get(svi_params[name]))

    sigma_f = float(_get("eb_sigma_f"))
    length_scale = float(_get("eb_length_scale"))
    theta_tau1 = float(_get("eb_theta_tau1"))
    theta_tau2 = float(_get("eb_theta_tau2"))
    theta_g1 = float(_get("eb_theta_g1"))
    theta_g2 = -float(_get("eb_neg_theta_g2"))
    rho = float(_init_from_dict(fallback_params, "rho", 1.0)) if fallback_params is not None else 1.0
    beta = np.asarray(_get("eb_beta"), dtype=np.float64)

    # sigma_epsilon may be fixed and therefore absent from svi_params.
    if "eb_sigma_epsilon" in svi_params:
        sigma_epsilon = float(_get("eb_sigma_epsilon"))
    elif fallback_params is not None:
        sigma_epsilon = float(fallback_params["sigma_epsilon"])
    else:
        sigma_epsilon = 0.1

    out = {
        "matern_params": (sigma_f, length_scale),
        "tau_params": (theta_tau1, theta_tau2),
        "g_params": (theta_g1, theta_g2, rho),
        "beta": beta,
        "sigma_epsilon": sigma_epsilon,
    }
    if fallback_params is not None:
        out["m"] = int(fallback_params.get("m", 20))
        out["m_tilde"] = int(fallback_params.get("m_tilde", len(beta)))
    return out

# ==============================================================================
# GPU / JAX vectorized prediction
# ==============================================================================
def _g_matern32_kernel_jax_fast(X, Y=None, rho=1.0):
    """Matern 3/2 g-kernel that preserves the input dtype for faster GPU prediction."""
    X = jnp.asarray(X)
    Y = X if Y is None else jnp.asarray(Y, dtype=X.dtype)
    rho = jnp.asarray(rho, dtype=X.dtype)
    dists = jnp.sqrt(jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1) + 1e-12)
    scaled = jnp.sqrt(jnp.asarray(3.0, dtype=X.dtype)) * dists / (rho + 1e-12)
    return (1.0 + scaled) * jnp.exp(-scaled)


def _precompute_reference_geometry_fast_np(S, neighbors, m, true_params):
    """
    Precompute all location-only quantities for S'.

    This is the key speed-up for prediction: for each posterior sample we only
    update quantities that depend on w_S, while B_i, F_i, tau_i and Lambda_i are
    reused.
    """
    S = np.asarray(S, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=int)
    m = min(int(m), len(S) - 1)

    sigma_f, length_scale = true_params["matern_params"]
    theta_tau1, theta_tau2 = true_params["tau_params"]
    theta_g1, theta_g2, _ = true_params["g_params"]

    idx = np.arange(m, len(S), dtype=int)
    nW = len(idx)
    N_ref = neighbors[idx]
    B_ref = np.zeros((nW, m), dtype=np.float64)
    F_ref = np.zeros(nW, dtype=np.float64)
    tau_ref = np.zeros(nW, dtype=np.float64)
    sqrt_lambda_ref = np.zeros((nW, m), dtype=np.float64)

    for row, i in enumerate(idx):
        N_i = N_ref[row]
        S_N = S[N_i]
        dists = np.linalg.norm(S[i] - S_N, axis=1)

        matern_jitter = _matern_jitter_np(sigma_f)
        C_NN = matern32_np(S_N, length_scale=length_scale, sigma_f=sigma_f) + matern_jitter * np.eye(m)
        C_iN = matern32_np(S[i : i + 1], S_N, length_scale=length_scale, sigma_f=sigma_f)[0]
        B_i = np.linalg.solve(C_NN, C_iN).T
        F_i = matern32_np(S[i : i + 1], length_scale=length_scale, sigma_f=sigma_f)[0, 0] - C_iN @ np.linalg.solve(C_NN, C_iN.T)

        B_ref[row] = B_i
        F_ref[row] = max(float(F_i), matern_jitter)
        tau_ref[row] = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda_ref[row] = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))

    return idx, N_ref, B_ref, F_ref, tau_ref, sqrt_lambda_ref


@jax.jit
def _predict_one_sample_all_locations_jax(
    key,
    w_S,
    beta,
    sigma_epsilon,
    ref_idx,
    ref_N,
    ref_B,
    ref_F,
    ref_tau,
    ref_sqrt_lambda,
    pred_N,
    pred_B,
    pred_F,
    pred_tau,
    pred_sqrt_lambda,
    X_U,
    Z,
    R_Z,
    R_Z_inv,
    rho,
    correction_jitter,
):
    """
    One posterior predictive draw for all prediction locations using GPU/JAX.

    This implements the location-wise version of p(w_U | w_S): it computes the
    diagonal predictive variance and samples locations independently. It avoids
    Python loops over locations and avoids using the synthetic true g_Z.
    """
    dtype = w_S.dtype
    jitter = jnp.asarray(correction_jitter, dtype=dtype)

    # ----- Build reference state W, residual, G a -----
    w_ref_N = w_S[ref_N]                                      # (nW, m)
    W = ref_sqrt_lambda * w_ref_N                             # (nW, m)
    h_ref = jnp.sum(ref_B * w_ref_N, axis=1)                  # (nW,)
    residual = w_S[ref_idx] - h_ref                           # (nW,)

    R_WZ = _g_matern32_kernel_jax_fast(W, Z, rho)              # (nW, mt)
    R_ZW = R_WZ.T                                             # (mt, nW)

    R_WZ_RZinv = R_WZ @ R_Z_inv
    diag_Q = jnp.sum(R_WZ_RZinv * R_WZ, axis=1)
    D1 = jnp.maximum(1.0 - diag_Q, jitter)
    D1_inv = 1.0 / D1

    D2_inv = ref_tau**2 / ref_F
    D3 = 1.0 / jnp.maximum(D1_inv + D2_inv, jitter)

    C1 = R_Z + (R_ZW * D1_inv[None, :]) @ R_WZ
    C1 = C1 + jitter * jnp.eye(Z.shape[0], dtype=dtype)

    diag_D = D1_inv * D3 * D1_inv
    C2 = C1 - (R_ZW * diag_D[None, :]) @ R_WZ
    C2 = C2 + jitter * jnp.eye(Z.shape[0], dtype=dtype)

    a = ref_tau / ref_F * residual
    t = R_ZW @ (D1_inv * D3 * a)
    x = jnp.linalg.solve(C2, t)
    Ga = D3 * a + (D3 * D1_inv) * (R_WZ @ x)                  # (nW,)

    # ----- Prediction locations, vectorized over all U -----
    w_pred_N = w_S[pred_N]                                    # (r, m)
    h_U = jnp.sum(pred_B * w_pred_N, axis=1)                  # (r,)
    V = pred_sqrt_lambda * w_pred_N                           # (r, m)

    R_VZ = _g_matern32_kernel_jax_fast(V, Z, rho)              # (r, mt)
    q_VW = (R_VZ @ R_Z_inv) @ R_ZW                            # (r, nW)

    # b = q_VW R_tilde_W^{-1}, using SMW low-rank structure.
    Y = q_VW * D1_inv[None, :]                                # (r, nW)
    TMP = jnp.linalg.solve(C1, (Y @ R_WZ).T).T                # (r, mt)
    B_VW = Y - (TMP @ R_WZ.T) * D1_inv[None, :]               # (r, nW)

    nonlinear_mean = pred_tau * (B_VW @ Ga)

    # R_{v|W} diagonal, with R(v,v)=1 for the standardized Matern correlation.
    R_v_given_W = 1.0 - jnp.sum(q_VW * B_VW, axis=1)
    R_v_given_W = jnp.maximum(R_v_given_W, jitter)

    # diag(B G B^T), using low-rank representation of G.
    b_D3_b = jnp.sum(B_VW * B_VW * D3[None, :], axis=1)
    TB = (B_VW * (D1_inv * D3)[None, :]) @ R_WZ               # (r, mt)
    TB_C2 = jnp.linalg.solve(C2, TB.T).T                      # (r, mt)
    bGb = b_D3_b + jnp.sum(TB_C2 * TB, axis=1)
    bGb = jnp.maximum(bGb, 0.0)

    var_w = pred_F + pred_tau**2 * (R_v_given_W + bGb)
    var_w = jnp.maximum(var_w, jitter)
    mean_w = h_U + nonlinear_mean

    key_w, key_y = jax.random.split(key)
    w_U = mean_w + jnp.sqrt(var_w) * jax.random.normal(key_w, shape=mean_w.shape, dtype=dtype)
    mu_y = X_U @ beta + w_U
    y_U = mu_y + jnp.maximum(sigma_epsilon, jitter) * jax.random.normal(key_y, shape=mu_y.shape, dtype=dtype)
    return y_U, w_U


def predict(
    w_S_samples,
    S,
    U,
    X_U,
    beta_samples,
    sigma_epsilon_samples,
    true_params,
    m=20,
    Z=None,
    neighbors_S=None,
    max_samples=None,
    seed=123,
    use_float32=True,
    return_w=False,
    verbose=True,
    z_jitter=JITTER,
):
    """
    GPU/JAX-vectorized location-wise posterior prediction.

    Compared with the previous NumPy implementation, this version removes the
    Python loop over prediction locations. It still loops over posterior samples
    at Python level, but each sample predicts all U locations in one compiled JAX
    call, so it can run on GPU when JAX is using a GPU backend.

    Parameters
    ----------
    use_float32 : bool
        Recommended True on RTX 4090/consumer GPUs. The inference may use
        float64, but prediction is usually much faster and sufficiently stable
        with float32.
    """
    if Z is None:
        raise ValueError("Z must be passed to predict().")

    backend = jax.default_backend()
    if verbose:
        print(f"预测阶段 JAX backend: {backend}, devices: {jax.devices()}")

    w_S_samples = np.asarray(w_S_samples)
    S = np.asarray(S, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    X_U = np.asarray(X_U)
    beta_samples = np.asarray(beta_samples)
    sigma_epsilon_samples = np.asarray(sigma_epsilon_samples)
    Z = np.asarray(Z, dtype=np.float64)

    if max_samples is not None and len(w_S_samples) > max_samples:
        use_idx = np.linspace(0, len(w_S_samples) - 1, max_samples).astype(int)
        w_S_samples = w_S_samples[use_idx]
        beta_samples = beta_samples[use_idx]
        sigma_epsilon_samples = sigma_epsilon_samples[use_idx]

    m = min(int(m), len(S) - 1)
    if neighbors_S is None:
        neighbors_S = build_neighbor_indices(S, m)
    else:
        neighbors_S = np.asarray(neighbors_S, dtype=int)

    if verbose:
        print("正在预计算 reference/prediction 几何项（CPU一次性小矩阵计算）...")
    ref_idx, ref_N, ref_B, ref_F, ref_tau, ref_sqrt_lambda = _precompute_reference_geometry_fast_np(
        S, neighbors_S, m, true_params
    )
    pred_N, pred_B, pred_F, pred_tau, pred_sqrt_lambda = _precompute_prediction_geometry(S, U, m, true_params)

    _, _, rho = true_params["g_params"]
    R_Z_np = g_matern32_kernel_np(Z, Z, rho) + float(z_jitter) * np.eye(len(Z))
    R_Z_inv_np = np.linalg.inv(R_Z_np)

    dtype = jnp.float32 if use_float32 else jnp.float64
    ref_idx_j = jnp.asarray(ref_idx, dtype=jnp.int32)
    ref_N_j = jnp.asarray(ref_N, dtype=jnp.int32)
    ref_B_j = jnp.asarray(ref_B, dtype=dtype)
    ref_F_j = jnp.asarray(ref_F, dtype=dtype)
    ref_tau_j = jnp.asarray(ref_tau, dtype=dtype)
    ref_sqrt_lambda_j = jnp.asarray(ref_sqrt_lambda, dtype=dtype)

    pred_N_j = jnp.asarray(pred_N, dtype=jnp.int32)
    pred_B_j = jnp.asarray(pred_B, dtype=dtype)
    pred_F_j = jnp.asarray(pred_F, dtype=dtype)
    pred_tau_j = jnp.asarray(pred_tau, dtype=dtype)
    pred_sqrt_lambda_j = jnp.asarray(pred_sqrt_lambda, dtype=dtype)
    X_U_j = jnp.asarray(X_U, dtype=dtype)
    Z_j = jnp.asarray(Z, dtype=dtype)
    R_Z_j = jnp.asarray(R_Z_np, dtype=dtype)
    R_Z_inv_j = jnp.asarray(R_Z_inv_np, dtype=dtype)
    rho_j = jnp.asarray(rho, dtype=dtype)
    z_jitter_j = jnp.asarray(z_jitter, dtype=dtype)

    num_samples = len(w_S_samples)
    r = len(U)
    if verbose:
        print(f"正在用 JAX/GPU 生成预测样本：{num_samples} posterior samples, {r} locations")

    sample_dtype = np.float32 if use_float32 else np.float64
    y_pred_samples = np.zeros((num_samples, r), dtype=sample_dtype)
    w_pred_samples = np.zeros((num_samples, r), dtype=sample_dtype) if return_w else None
    base_key = jax.random.key(seed)
    keys = jax.random.split(base_key, num_samples)

    # First call compiles; subsequent calls reuse the compiled program.
    for s_idx in range(num_samples):
        if verbose and s_idx % 20 == 0:
            print(f"  处理 posterior sample {s_idx}/{num_samples}")

        y_draw, w_draw = _predict_one_sample_all_locations_jax(
            keys[s_idx],
            jnp.asarray(w_S_samples[s_idx], dtype=dtype),
            jnp.asarray(beta_samples[s_idx], dtype=dtype),
            jnp.asarray(sigma_epsilon_samples[s_idx], dtype=dtype),
            ref_idx_j,
            ref_N_j,
            ref_B_j,
            ref_F_j,
            ref_tau_j,
            ref_sqrt_lambda_j,
            pred_N_j,
            pred_B_j,
            pred_F_j,
            pred_tau_j,
            pred_sqrt_lambda_j,
            X_U_j,
            Z_j,
            R_Z_j,
            R_Z_inv_j,
            rho_j,
            z_jitter_j,
        )
        y_pred_samples[s_idx] = np.asarray(y_draw)
        if return_w:
            w_pred_samples[s_idx] = np.asarray(w_draw)

    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_std = np.std(y_pred_samples, axis=0)
    if return_w:
        w_pred_mean = np.mean(w_pred_samples, axis=0)
        w_pred_std = np.std(w_pred_samples, axis=0)
        return y_pred_mean, y_pred_std, y_pred_samples, w_pred_mean, w_pred_std, w_pred_samples
    return y_pred_mean, y_pred_std, y_pred_samples


# ==============================================================================
# Configurable VI / Empirical-Bayes model
# ==============================================================================
PARAMETER_NAMES = [
    "sigma_f", "length_scale", "theta_tau1", "theta_tau2",
    "theta_g1", "theta_g2", "beta", "sigma_epsilon",
]


def _mode_for(param_mode, name):
    """Return 'fixed' or 'eb' for a parameter."""
    if param_mode is None:
        return "eb"
    return param_mode.get(name, "eb")


def _fixed_or_eb_scalar(name, init_params, param_mode, default, constraint=None):
    """
    Return either a fixed true/init value or a numpyro.param EB estimate.

    mode='fixed': use value from init_params and do not optimize.
    mode='eb'   : create numpyro.param('eb_<name>') and optimize by ELBO.
    """
    value0 = _init_from_dict(init_params, name, default)
    if _mode_for(param_mode, name) == "fixed":
        return jnp.asarray(value0, dtype=jnp.float64)

    init_value = jnp.asarray(value0, dtype=jnp.float64)
    if constraint is None:
        return numpyro.param(f"eb_{name}", init_value)
    return numpyro.param(f"eb_{name}", init_value, constraint=constraint)


def _fixed_or_eb_theta_g2(init_params, param_mode):
    """theta_g2 must be negative for screening; optimize -theta_g2 if EB."""
    init_theta_g2 = _init_from_dict(init_params, "theta_g2", -2.0)
    if _mode_for(param_mode, "theta_g2") == "fixed":
        return jnp.asarray(init_theta_g2, dtype=jnp.float64)

    neg_theta_g2 = numpyro.param(
        "eb_neg_theta_g2",
        jnp.asarray(max(-init_theta_g2, 1e-3), dtype=jnp.float64),
        constraint=dist.constraints.positive,
    )
    return -neg_theta_g2


def _fixed_or_eb_beta(init_params, param_mode, p):
    if init_params is not None and "beta" in init_params:
        beta_init = jnp.asarray(init_params["beta"], dtype=jnp.float64)
        if beta_init.shape[0] != p:
            beta_init = jnp.zeros((p,), dtype=jnp.float64)
    else:
        beta_init = jnp.zeros((p,), dtype=jnp.float64)

    if _mode_for(param_mode, "beta") == "fixed":
        return beta_init
    return numpyro.param("eb_beta", beta_init)


def use_noncentered_from_param_mode(param_mode):
    """
    Non-centered z = T w is only consistent when the base Matérn parameters
    that define T are fixed. If sigma_f or length_scale is EB, use direct w_S.
    """
    return (
        _mode_for(param_mode, "sigma_f") == "fixed"
        and _mode_for(param_mode, "length_scale") == "fixed"
    )


def nnngp_model_vi_configurable(
    X_S,
    y_S,
    S,
    m=20,
    m_tilde=10,
    init_params=None,
    param_mode=None,
    neighbors=None,
    Z=None,
    optimize_Z=False,
    z_jitter=JITTER,
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
    """
    Configurable VI/EB model for NNnGP.

    param_mode controls each parameter:
        'fixed' : use init_params / true value, not optimized
        'eb'    : optimize by numpyro.param under the ELBO

    If sigma_f and length_scale are fixed, the latent variable is non-centered:
        w_S_std ~ N(0,I), w_S = T^{-1} w_S_std,
    and we add only the NNnGP non-Gaussian FIC correction. The NNGP base prior is
    already represented by w_S_std ~ N(0,I).

    If sigma_f or length_scale is EB, T would change during optimization, so the
    model uses direct parameterization:
        w_S ~ N(0,I)
    and replaces the auxiliary standard normal density with
        log p_NNGP(w_S; sigma_f,length_scale) + correction.
    """
    X_S = jnp.asarray(X_S, dtype=jnp.float64)
    y_S = jnp.asarray(y_S, dtype=jnp.float64)
    S = jnp.asarray(S, dtype=jnp.float64)
    if neighbors is None:
        raise ValueError("neighbors must be precomputed and passed in.")
    if Z is None:
        raise ValueError("FIC inducing points Z must be passed in.")
    neighbors = jnp.asarray(neighbors, dtype=jnp.int32)
    Z = jnp.asarray(Z, dtype=jnp.float64)
    if optimize_Z:
        Z = numpyro.param("eb_Z", Z)
    if param_mode is None:
        param_mode = {}

    n = y_S.shape[0]
    p = X_S.shape[1]

    sigma_f = _fixed_or_eb_scalar(
        "sigma_f", init_params, param_mode, 1.0,
        constraint=dist.constraints.positive,
    )
    length_scale = _fixed_or_eb_scalar(
        "length_scale", init_params, param_mode, 0.2,
        constraint=dist.constraints.positive,
    )
    theta_tau1 = _fixed_or_eb_scalar("theta_tau1", init_params, param_mode, 0.0)
    theta_tau2 = _fixed_or_eb_scalar(
        "theta_tau2", init_params, param_mode, 1.0,
        constraint=dist.constraints.positive,
    )
    theta_g1 = _fixed_or_eb_scalar("theta_g1", init_params, param_mode, 0.0)
    theta_g2 = _fixed_or_eb_theta_g2(init_params, param_mode)
    rho = jnp.asarray(1.0, dtype=jnp.float32)
    beta = _fixed_or_eb_beta(init_params, param_mode, p)
    sigma_epsilon = _fixed_or_eb_scalar(
        "sigma_epsilon", init_params, param_mode, 0.1,
        constraint=dist.constraints.positive,
    )

    use_noncentered = use_noncentered_from_param_mode(param_mode)

    z_jitter = jnp.asarray(z_jitter, dtype=jnp.float64)
    R_Z = g_matern32_kernel_jax(Z, Z, rho) + z_jitter * jnp.eye(Z.shape[0], dtype=jnp.float64)
    R_Z_chol = jnp.linalg.cholesky(R_Z)

    if use_noncentered:
        if L is None:
            raise ValueError("L must be provided when sigma_f and length_scale are fixed.")
        L = jnp.asarray(L, dtype=jnp.float64)
        w_S_std = numpyro.sample(
            "w_S_std",
            dist.Normal(0.0, 1.0).expand([n]).to_event(1),
        )
        w_S = jax.scipy.linalg.solve_triangular(L, w_S_std, lower=True, trans="N")

        if all(x is not None for x in (fic_ref_idx, fic_ref_N, fic_ref_B, fic_ref_F, fic_ref_dists)):
            correction = compute_fic_correction_precomputed(
                w_S,
                Z,
                theta_tau1,
                theta_tau2,
                theta_g1,
                theta_g2,
                rho,
                fic_ref_idx,
                fic_ref_N,
                fic_ref_B,
                fic_ref_F,
                fic_ref_dists,
                R_Z=R_Z,
                R_Z_chol=R_Z_chol,
                correction_jitter=z_jitter,
            )
        else:
            correction = compute_fic_correction(
                w_S,
                S,
                neighbors,
                m,
                Z,
                None,
                theta_tau1,
                theta_tau2,
                theta_g1,
                theta_g2,
                rho,
                sigma_f,
                length_scale,
                R_Z=R_Z,
                R_Z_chol=R_Z_chol,
                correction_jitter=z_jitter,
            )
        numpyro.factor("non_gaussian_fic_correction", correction)
    else:
        w_S = numpyro.sample(
            "w_S",
            dist.Normal(0.0, 1.0).expand([n]).to_event(1),
        )
        if all(
            x is not None
            for x in (
                eb_base0_matern_dists,
                eb_ref_idx,
                eb_ref_N,
                eb_ref_matern_NN_dists,
                eb_ref_matern_iN_dists,
                eb_ref_euclid_iN_dists,
            )
        ):
            logp_base, correction = compute_eb_base_and_fic_correction_precomputed(
                w_S,
                Z,
                sigma_f,
                length_scale,
                theta_tau1,
                theta_tau2,
                theta_g1,
                theta_g2,
                rho,
                eb_base0_matern_dists,
                eb_ref_idx,
                eb_ref_N,
                eb_ref_matern_NN_dists,
                eb_ref_matern_iN_dists,
                eb_ref_euclid_iN_dists,
                R_Z=R_Z,
                R_Z_chol=R_Z_chol,
                correction_jitter=z_jitter,
            )
        else:
            logp_base = _nnngp_base_logprob_jax(w_S, S, neighbors, sigma_f, length_scale)
            correction = compute_fic_correction(
                w_S,
                S,
                neighbors,
                m,
                Z,
                None,
                theta_tau1,
                theta_tau2,
                theta_g1,
                theta_g2,
                rho,
                sigma_f,
                length_scale,
                R_Z=R_Z,
                R_Z_chol=R_Z_chol,
                correction_jitter=z_jitter,
            )
        logp_aux = _standard_normal_logprob_vec(w_S)
        numpyro.factor("empirical_bayes_nnngp_prior", logp_base + correction - logp_aux)

    mu = X_S @ beta + w_S
    numpyro.sample(
        "y_S",
        dist.Normal(mu, jnp.maximum(sigma_epsilon, JITTER)).to_event(1),
        obs=y_S,
    )


def extract_configurable_vi_params(svi_params, true_params, param_mode):
    """Extract fixed/EB parameters into the true_params-style dict used by predict()."""
    if param_mode is None:
        param_mode = {}

    def _get_raw(name):
        return np.asarray(jax.device_get(svi_params[name]))

    def _scalar(name, true_value):
        if _mode_for(param_mode, name) == "fixed":
            return float(true_value)
        eb_name = f"eb_{name}"
        if eb_name not in svi_params:
            return float(true_value)
        return float(_get_raw(eb_name))

    sigma_f = _scalar("sigma_f", true_params["matern_params"][0])
    length_scale = _scalar("length_scale", true_params["matern_params"][1])
    theta_tau1 = _scalar("theta_tau1", true_params["tau_params"][0])
    theta_tau2 = _scalar("theta_tau2", true_params["tau_params"][1])
    theta_g1 = _scalar("theta_g1", true_params["g_params"][0])

    if _mode_for(param_mode, "theta_g2") == "fixed":
        theta_g2 = float(true_params["g_params"][1])
    elif "eb_neg_theta_g2" in svi_params:
        theta_g2 = -float(_get_raw("eb_neg_theta_g2"))
    else:
        theta_g2 = float(true_params["g_params"][1])

    rho = 1.0

    if _mode_for(param_mode, "beta") == "fixed":
        beta = np.asarray(true_params["beta"], dtype=np.float64)
    elif "eb_beta" in svi_params:
        beta = np.asarray(_get_raw("eb_beta"), dtype=np.float64)
    else:
        beta = np.asarray(true_params["beta"], dtype=np.float64)

    sigma_epsilon = _scalar("sigma_epsilon", true_params["sigma_epsilon"])

    return {
        "matern_params": (sigma_f, length_scale),
        "tau_params": (theta_tau1, theta_tau2),
        "g_params": (theta_g1, theta_g2, rho),
        "beta": beta,
        "sigma_epsilon": sigma_epsilon,
        "m": int(true_params.get("m", 20)),
        "m_tilde": int(true_params.get("m_tilde", len(beta))),
    }
