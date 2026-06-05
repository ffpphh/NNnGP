import numpy as np
import os
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde, norm
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from matplotlib import colors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "data_utils_outputs")

def _ensure_parent_dir(file_path):
    """确保输出文件的父目录存在。"""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

def maximin_ordering(points):
    """对参考点进行maximin排序（Vecchia近似标准做法）"""
    n = len(points)
    order = [np.random.randint(n)]
    remaining = set(range(n)) - set(order)
    
    for _ in range(n-1):
        remaining_points = points[list(remaining)]
        min_distances = np.min(cdist(remaining_points, points[order]), axis=1)
        next_idx = list(remaining)[np.argmax(min_distances)]
        order.append(next_idx)
        remaining.remove(next_idx)
    
    return np.array(order)

def generate_nngp_ws(S, kernel, m, random_seed=42):
    """
    生成参考点 S 上的纯净 NNGP 潜变量（无非线性修正）。
    严格按照 Datta et al. (2016) 的递归条件高斯定义。
    
    参数:
        S : (k, d) 参考点坐标
        kernel : 父 GP 的协方差函数（如 Matern）
        m : 最近邻数量
        random_seed : 随机种子
    
    返回:
        w : (k,) NNGP 潜变量
    """
    np.random.seed(random_seed)
    k = len(S)
    w = np.zeros(k)
    
    # 第一个点
    w[0] = np.random.normal(0, np.sqrt(kernel(S[0:1])[0,0]))
    
    for i in range(1, k):
        # 在前 i 个点中寻找最近的邻居（最多 m 个）
        dists = cdist(S[i:i+1], S[:i])[0]
        n_neighbors = min(i, m)
        neighbor_idx = np.argsort(dists)[:n_neighbors]
        
        S_nei = S[neighbor_idx]
        w_nei = w[neighbor_idx]
        
        C_nn = kernel(S_nei) + 1e-6 * np.eye(n_neighbors)
        C_in = kernel(S[i:i+1], S_nei)[0]
        inv_C_nn = np.linalg.inv(C_nn)
        B = C_in @ inv_C_nn
        mu = B @ w_nei
        F = kernel(S[i:i+1])[0,0] - C_in @ inv_C_nn @ C_in.T
        F = max(F, 1e-8)
        
        w[i] = np.random.normal(mu, np.sqrt(F))
    
    return w

def parametric_tanh_g(v, params=(1.0, 0.5, 0.0)):
    """Simple signed nonlinear function g(v) = amplitude * tanh(slope * mean(v) + bias)."""
    amplitude, slope, bias = params
    return float(amplitude * np.tanh(slope * np.mean(v) + bias))

def generate_nnngp_data(
    k=500,          # 参考点数量
    m=10,           # 最近邻数量
    m_tilde=50,     # FIC诱导点数量
    matern_params=(1.0, 0.2),  # 父GP核参数：(sigma_f, length_scale)，协方差幅度为 sigma_f^2
    tau_params=(0.0, 1.0),     # 非线性强度参数：(theta_tau1, theta_tau2)
    g_params=(0.0, -2.0, 1.0), # 距离权重参数：(theta_g1, theta_g2, unused_rho)
    parametric_g_params=(1.0, 0.5, 0.0), # tanh g参数：(amplitude, slope, bias)
    beta=(0.5, 1.0, -1.0),     # 回归系数：(截距, x1系数, x2系数)
    sigma_epsilon=0.1,         # 观测噪声标准差
    save_path=os.path.join(RESULTS_DIR, "synthetic_data.npz"),
    precompute_Z_with_NNGP=True,  # 是否使用 NNGP 预热来选取诱导点
    grid_size=100,  # 每个维度的网格点数
    domain_size=5.0, # 空间域 [0,domain_size]^2
    random_seed=42
):
    """
    严格按照论文层次模型生成合成数据（两阶段法）：
    阶段1（预热）：用纯净 NNGP 生成 w_S_NNGP，计算 v_i，并保留 Z 作为诊断/兼容输出。
    阶段2（正式）：使用参数化 tanh g(v) 递归生成非线性 NNnGP 数据。
    
    参数:
        precompute_Z_with_NNGP : bool, 是否使用 NNGP 预热。若为 False，则随机初始化 Z（不推荐）。
    """
    np.random.seed(random_seed)
    print("正在生成合成数据（两阶段法）...")
    
    # 1. 生成 [0,domain_size]^2 上的均匀网格
    x = np.linspace(0, domain_size, grid_size)
    y = np.linspace(0, domain_size, grid_size)
    xx, yy = np.meshgrid(x, y)
    all_points = np.column_stack((xx.ravel(), yy.ravel()))
    n_total = len(all_points)
    if k > n_total:
        raise ValueError(f"k={k} cannot exceed grid_size^2={n_total}.")
    
    # 2. 随机选择参考点并进行maximin排序
    s_indices = np.random.choice(n_total, k, replace=False)
    S = all_points[s_indices]
    s_order = maximin_ordering(S)
    S = S[s_order]
    s_indices = s_indices[s_order]
    
    # 3. 预测点（剩余所有点）
    u_indices = np.setdiff1d(np.arange(n_total), s_indices)
    U = all_points[u_indices]
    
    # 4. 初始化核函数
    matern_kernel = Matern(length_scale=matern_params[1], nu=1.5)
    matern_kernel = (matern_params[0] ** 2) * matern_kernel
    
    # ==============================================
    # 阶段1：使用纯净 NNGP 生成诊断/兼容用的 Z
    # ==============================================
    print("阶段1/2: 预热 - 生成纯净 NNGP 的 w_S 并选取诊断用 Z...")
    
    if precompute_Z_with_NNGP:
        # 生成纯净 NNGP 的 w_S（无非线性项）
        w_S_nngp = generate_nngp_ws(S, matern_kernel, m, random_seed=random_seed)
        
        # 计算所有 i >= m 的 v_i
        v_samples = []
        for i in range(m, k):
            # 找到前 i 个点中最近的 m 个邻居（此时 i>=m，邻居数至少为 m）
            distances = cdist(S[i:i+1], S[:i])[0]
            neighbor_indices = np.argsort(distances)[:m]
            w_Nsi = w_S_nngp[neighbor_indices]
            
            # 计算距离加权矩阵 Lambda
            dist_to_neighbors = cdist(S[i:i+1], S[neighbor_indices])[0]
            Lambda_i = np.diag(np.exp(g_params[0] + g_params[1] * dist_to_neighbors))
            v_i = np.sqrt(Lambda_i) @ w_Nsi
            v_samples.append(v_i)
        
        v_samples = np.array(v_samples)  # shape (k-m, m)
        
        # 从 v_samples 中随机选择 m_tilde 个作为诱导点 Z
        if m_tilde <= len(v_samples):
            Z = v_samples[np.random.choice(len(v_samples), m_tilde, replace=False)]
        else:
            # 如果 m_tilde 大于 v_samples 数量，补充标准正态随机点
            Z = np.vstack([
                v_samples,
                np.random.randn(m_tilde - len(v_samples), m) * np.std(v_samples, axis=0)
            ])
    else:
        # 不使用预热：随机初始化 Z（标准正态）
        Z = np.random.randn(m_tilde, m)
    
    print(f"诊断用 Z 已选取，形状: {Z.shape}")
    
    # ==============================================
    # 阶段2：正式生成非线性 NNnGP 数据（使用参数化 tanh g）
    # ==============================================
    print("阶段2/2: 正式生成非线性 NNnGP 潜场 w_S, w_U 及观测 y...")
    
    w_S = np.zeros(k)
    # 生成前 m 个点（直接使用父 GP 联合分布）
    if m > 0:
        cov = matern_kernel(S[:m]) + 1e-4 * np.eye(m)
        w_S[:m] = np.random.multivariate_normal(np.zeros(m), cov)
    
    # 递归生成剩余参考点
    for i in range(m, k):
        # 找到前 i 个点中最近的 m 个邻居
        distances = cdist(S[i:i+1], S[:i])[0]
        neighbor_indices = np.argsort(distances)[:m]
        N_si = S[neighbor_indices]
        w_Nsi = w_S[neighbor_indices]
        
        # 计算 NNGP 条件均值和方差（克里金）
        C_NN = matern_kernel(N_si) + 1e-4 * np.eye(m)
        C_sN = matern_kernel(S[i:i+1], N_si)[0]
        inv_C_NN = np.linalg.inv(C_NN)
        B_si = C_sN @ inv_C_NN
        h_i = B_si @ w_Nsi
        C_si_cond = matern_kernel(S[i:i+1])[0,0] - C_sN @ inv_C_NN @ C_sN.T
        C_si_cond = max(C_si_cond, 1e-6)
        
        # 计算 τ_i
        min_dist = np.min(distances[neighbor_indices])
        tau_i = np.sqrt(np.exp(tau_params[0]) * (min_dist + 1e-12) ** tau_params[1])
        
        # 计算 v_i（使用当前已生成的 w_Nsi）
        dist_to_neighbors = cdist(S[i:i+1], N_si)[0]
        Lambda_i = np.diag(np.exp(g_params[0] + g_params[1] * dist_to_neighbors))
        v_i = np.sqrt(Lambda_i) @ w_Nsi
        
        # 参数化非线性 g(v)，可正可负。
        g_i = parametric_tanh_g(v_i, parametric_g_params)
        
        # 采样 w_i
        mu_i = h_i + tau_i * g_i
        w_S[i] = np.random.normal(mu_i, np.sqrt(C_si_cond))
    
    # 生成预测点上的 w_U
    w_U = np.zeros(len(U))
    for i in range(len(U)):
        u = U[i:i+1]
        distances = cdist(u, S)[0]
        neighbor_indices = np.argsort(distances)[:m]
        N_ui = S[neighbor_indices]
        w_Nui = w_S[neighbor_indices]
        
        # 克里金
        C_NN = matern_kernel(N_ui) + 1e-4 * np.eye(m)
        C_uN = matern_kernel(u, N_ui)[0]
        inv_C_NN = np.linalg.inv(C_NN)
        B_ui = C_uN @ inv_C_NN
        h_u = B_ui @ w_Nui
        C_u_cond = matern_kernel(u)[0,0] - C_uN @ inv_C_NN @ C_uN.T
        C_u_cond = max(C_u_cond, 1e-6)
        
        min_dist = np.min(distances[neighbor_indices])
        tau_u = np.sqrt(np.exp(tau_params[0]) * (min_dist + 1e-12) ** tau_params[1])
        
        dist_to_neighbors = cdist(u, N_ui)[0]
        Lambda_u = np.diag(np.exp(g_params[0] + g_params[1] * dist_to_neighbors))
        v_u = np.sqrt(Lambda_u) @ w_Nui
        
        g_u = parametric_tanh_g(v_u, parametric_g_params)
        
        mu_u = h_u + tau_u * g_u
        w_U[i] = np.random.normal(mu_u, np.sqrt(C_u_cond))
    
    # 生成观测数据 y
    X_all = np.column_stack((np.ones(n_total), all_points[:,0], all_points[:,1]))
    w_all = np.zeros(n_total)
    w_all[s_indices] = w_S
    w_all[u_indices] = w_U
    
    y_all = X_all @ np.array(beta) + w_all + np.random.normal(0, sigma_epsilon, n_total)
    y_S = y_all[s_indices]
    y_U = y_all[u_indices]
    
    # 保存数据
    _ensure_parent_dir(save_path)
    np.savez_compressed(
        save_path,
        all_points=all_points,
        S=S, s_indices=s_indices, w_S=w_S, y_S=y_S,
        U=U, u_indices=u_indices, w_U=w_U, y_U=y_U,
        X_all=X_all, X_S=X_all[s_indices], X_U=X_all[u_indices],
        Z=Z,
        true_params={
            'matern_params': matern_params,
            'tau_params': tau_params,
            'g_params': g_params,
            'parametric_g_params': parametric_g_params,
            'g_type': 'parametric_tanh',
            'beta': beta,
            'sigma_epsilon': sigma_epsilon,
            'm': m, 'm_tilde': m_tilde,
            'domain_size': domain_size,
        }
    )
    
    print(f"数据生成完成！已保存至: {save_path}")
    print(f"参考点数量: {k}, 预测点数量: {len(U)}")
    return save_path

def load_data(data_path=os.path.join(RESULTS_DIR, "synthetic_data.npz")):
    """加载保存的合成数据"""
    data = np.load(data_path, allow_pickle=True)
    data_dict = {key: data[key] for key in data.files}
    data_dict['true_params'] = data_dict['true_params'].item()
    return data_dict

def _red_high_value_norm(values, vmin=None, vmax=None):
    """Use a broad low-purple/green/high-red scale with red for larger values."""
    values = np.asarray(values)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmin < 0.0 < vmax:
        return colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return colors.Normalize(vmin=vmin, vmax=vmax)

def plot_spatial_field(
    points,
    values,
    title="Spatial Field",
    save_path=None,
    marked_points=None,
    marked_labels=None,
):
    """绘制空间场热力图"""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        points[:,0],
        points[:,1],
        c=values,
        cmap="Spectral_r",
        norm=_red_high_value_norm(values),
        s=20,
        marker='s',
        linewidths=0,
        rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, label='Value')

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
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_results_comparison(true_values, pred_mean, pred_std, title="Prediction Results", save_path=None):
    """统一绘图风格：相同点大小、坐标比例、前两张图共用 color scale。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    x = true_values[:, 0]
    y = true_values[:, 1]
    y_true = true_values[:, 2]

    # True 和 Predicted Mean 使用同一套颜色范围，便于视觉比较
    vmin = min(np.nanmin(y_true), np.nanmin(pred_mean))
    vmax = max(np.nanmax(y_true), np.nanmax(pred_mean))
    mean_norm = _red_high_value_norm(np.concatenate([y_true, pred_mean]), vmin=vmin, vmax=vmax)

    common_scatter_kwargs = dict(
        s=18,
        marker="s",
        linewidths=0,
        rasterized=True
    )

    im1 = axes[0].scatter(
        x, y, c=y_true, cmap="Spectral_r",
        norm=mean_norm,
        **common_scatter_kwargs
    )
    axes[0].set_title("True Values")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].scatter(
        x, y, c=pred_mean, cmap="Spectral_r",
        norm=mean_norm,
        **common_scatter_kwargs
    )
    axes[1].set_title("Predicted Mean")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].scatter(
        x, y, c=pred_std, cmap="Spectral_r",
        norm=colors.Normalize(vmin=float(np.nanmin(pred_std)), vmax=float(np.nanmax(pred_std))),
        **common_scatter_kwargs
    )
    axes[2].set_title("Predicted Standard Deviation")
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[2])

    x_pad = 0.02 * max(float(np.max(x) - np.min(x)), 1.0)
    y_pad = 0.02 * max(float(np.max(y) - np.min(y)), 1.0)
    for ax in axes:
        ax.set_xlim(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
        ax.set_ylim(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(title, fontsize=14)

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _precompute_fixed_s_terms(S, m, true_params):
    """Precompute location-only terms for repeated fixed-S NNnGP simulation."""
    sigma_f, length_scale = true_params["matern_params"]
    theta_tau1, theta_tau2 = true_params["tau_params"]
    theta_g1, theta_g2, _ = true_params["g_params"]
    matern_kernel = (sigma_f ** 2) * Matern(length_scale=length_scale, nu=1.5)

    k = len(S)
    neighbor_idx = np.full((k, m), 0, dtype=int)
    B = np.zeros((k, m), dtype=np.float64)
    F = np.zeros(k, dtype=np.float64)
    tau = np.zeros(k, dtype=np.float64)
    sqrt_lambda = np.zeros((k, m), dtype=np.float64)

    cov0 = matern_kernel(S[:m]) + 1e-4 * np.eye(m)

    for i in range(m, k):
        distances = cdist(S[i:i + 1], S[:i])[0]
        N_i = np.argsort(distances)[:m]
        S_N = S[N_i]
        dists = distances[N_i]

        C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m)
        C_iN = matern_kernel(S[i:i + 1], S_N)[0]
        inv_C_NN = np.linalg.inv(C_NN)

        neighbor_idx[i] = N_i
        B[i] = C_iN @ inv_C_NN
        F_i = matern_kernel(S[i:i + 1])[0, 0] - C_iN @ inv_C_NN @ C_iN.T
        F[i] = max(float(F_i), 1e-6)
        tau[i] = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
        sqrt_lambda[i] = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))

    return {
        "matern_kernel": matern_kernel,
        "cov0": cov0,
        "neighbor_idx": neighbor_idx,
        "B": B,
        "F": F,
        "tau": tau,
        "sqrt_lambda": sqrt_lambda,
    }

def _simulate_fixed_s_y(S, X_S, Z, true_params, precomputed, rng):
    """Simulate one y_S draw on fixed reference locations S and fixed inducing locations Z."""
    w_S, _, _, _, _ = _simulate_fixed_s_state(S, Z, true_params, precomputed, rng)
    beta = np.asarray(true_params["beta"], dtype=np.float64)
    sigma_epsilon = float(true_params["sigma_epsilon"])
    return X_S @ beta + w_S + rng.normal(0.0, sigma_epsilon, len(S))

def _simulate_fixed_s_state(S, Z, true_params, precomputed, rng):
    """Simulate one latent w_S draw with parametric tanh g."""
    m = int(true_params["m"])
    k = len(S)
    parametric_g_params = true_params.get("parametric_g_params", (1.0, 0.5, 0.0))

    w_S = np.zeros(k, dtype=np.float64)
    if m > 0:
        w_S[:m] = rng.multivariate_normal(np.zeros(m), precomputed["cov0"])

    for i in range(m, k):
        N_i = precomputed["neighbor_idx"][i]
        w_N = w_S[N_i]
        h_i = precomputed["B"][i] @ w_N
        v_i = precomputed["sqrt_lambda"][i] * w_N

        g_i = parametric_tanh_g(v_i, parametric_g_params)

        mu_i = h_i + precomputed["tau"][i] * g_i
        w_S[i] = rng.normal(mu_i, np.sqrt(precomputed["F"][i]))

    return w_S, None, None, None, None

def _simulate_w_at_locations(locations, S, w_S, Z, true_params, precomputed, g_Z, R_Z_inv, g_kernel, rng):
    """Simulate latent w at arbitrary fixed locations given one simulated reference field."""
    locations = np.asarray(locations, dtype=np.float64)
    m = int(true_params["m"])
    theta_tau1, theta_tau2 = true_params["tau_params"]
    theta_g1, theta_g2, _ = true_params["g_params"]
    parametric_g_params = true_params.get("parametric_g_params", (1.0, 0.5, 0.0))
    matern_kernel = precomputed["matern_kernel"]

    s_lookup = {tuple(point): idx for idx, point in enumerate(np.asarray(S, dtype=np.float64))}
    w_values = np.zeros(len(locations), dtype=np.float64)

    for j, loc in enumerate(locations):
        s_idx = s_lookup.get(tuple(loc))
        if s_idx is not None:
            w_loc = w_S[s_idx]
        else:
            distances = cdist(loc.reshape(1, -1), S)[0]
            N_j = np.argsort(distances)[:m]
            S_N = S[N_j]
            w_N = w_S[N_j]
            dists = distances[N_j]

            C_NN = matern_kernel(S_N) + 1e-4 * np.eye(m)
            C_uN = matern_kernel(loc.reshape(1, -1), S_N)[0]
            inv_C_NN = np.linalg.inv(C_NN)
            B_j = C_uN @ inv_C_NN
            h_j = B_j @ w_N
            F_j = matern_kernel(loc.reshape(1, -1))[0, 0] - C_uN @ inv_C_NN @ C_uN.T
            F_j = max(float(F_j), 1e-6)

            tau_j = np.sqrt(np.exp(theta_tau1) * (np.min(dists) + 1e-12) ** theta_tau2)
            sqrt_lambda_j = np.sqrt(np.exp(theta_g1 + theta_g2 * dists))
            v_j = sqrt_lambda_j * w_N

            g_j = parametric_tanh_g(v_j, parametric_g_params)

            mu_j = h_j + tau_j * g_j
            w_loc = rng.normal(mu_j, np.sqrt(F_j))

        w_values[j] = w_loc

    return w_values

def _simulate_y_at_locations(locations, S, w_S, Z, true_params, precomputed, g_Z, R_Z_inv, g_kernel, rng):
    """Simulate y at arbitrary fixed locations given one simulated reference field."""
    locations = np.asarray(locations, dtype=np.float64)
    beta = np.asarray(true_params["beta"], dtype=np.float64)
    sigma_epsilon = float(true_params["sigma_epsilon"])
    x_rows = np.column_stack((np.ones(len(locations)), locations[:, 0], locations[:, 1]))
    w_values = _simulate_w_at_locations(
        locations,
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
    y_values = x_rows @ beta + w_values + rng.normal(0.0, sigma_epsilon, len(locations))
    return y_values

def plot_random_y_density_repeats(
    data_path=os.path.join(RESULTS_DIR, "test_synthetic_data.npz"),
    repeats=200,
    random_seed=None,
    save_path=os.path.join(RESULTS_DIR, "random_y_density_repeats.png"),
):
    """
    Randomly choose two coordinates from all_points each run, repeatedly simulate y there,
    and plot empirical densities against matched Gaussian curves.
    """
    data = load_data(data_path)
    all_points = np.asarray(data["all_points"], dtype=np.float64)
    S = np.asarray(data["S"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    m = int(true_params["m"])

    if len(S) <= m:
        raise ValueError("Need len(S) > m to simulate NNnGP reference locations.")

    rng = np.random.default_rng(random_seed)
    point_indices = np.sort(rng.choice(np.arange(len(all_points)), size=2, replace=False))
    point_locations = all_points[point_indices]

    precomputed = _precompute_fixed_s_terms(S, m, true_params)
    y_draws = np.zeros((repeats, 2), dtype=np.float64)
    for r in range(repeats):
        w_S, g_Z, R_Z_inv, _, g_kernel = _simulate_fixed_s_state(S, Z, true_params, precomputed, rng)
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

    _plot_two_point_density(
        y_draws,
        point_indices,
        point_locations,
        repeats,
        save_path,
        title_prefix="Random fixed-location",
    )
    joint_save_path = None
    if save_path:
        root, ext = os.path.splitext(save_path)
        joint_save_path = f"{root}_joint{ext or '.png'}"
        _plot_two_point_joint_density(
            y_draws,
            point_indices,
            point_locations,
            repeats,
            joint_save_path,
        )

    npz_path = None
    if save_path:
        npz_path = os.path.splitext(save_path)[0] + ".npz"
        np.savez_compressed(
            npz_path,
            y_draws=y_draws,
            point_indices=point_indices,
            point_locations=point_locations,
            repeats=repeats,
        )

    print("随机固定坐标 y 重复模拟完成")
    print(f"  point_indices_in_all_points: {point_indices.tolist()}")
    print(f"  point_locations: {point_locations.tolist()}")
    print(f"  density_plot: {save_path}")
    if joint_save_path:
        print(f"  joint_density_plot: {joint_save_path}")
    if npz_path:
        print(f"  samples_npz: {npz_path}")
    return y_draws, point_indices, point_locations

def plot_random_w_density_repeats(
    data_path=os.path.join(RESULTS_DIR, "test_synthetic_data.npz"),
    repeats=200,
    random_seed=None,
    save_path=os.path.join(RESULTS_DIR, "random_w_density_repeats.png"),
):
    """
    Randomly choose two coordinates from all_points each run, repeatedly simulate latent w
    there, and plot empirical densities against matched Gaussian curves.
    """
    data = load_data(data_path)
    all_points = np.asarray(data["all_points"], dtype=np.float64)
    S = np.asarray(data["S"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    m = int(true_params["m"])

    if len(S) <= m:
        raise ValueError("Need len(S) > m to simulate NNnGP reference locations.")

    rng = np.random.default_rng(random_seed)
    point_indices = np.sort(rng.choice(np.arange(len(all_points)), size=2, replace=False))
    point_locations = all_points[point_indices]

    precomputed = _precompute_fixed_s_terms(S, m, true_params)
    w_draws = np.zeros((repeats, 2), dtype=np.float64)
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

    _plot_two_point_density(
        w_draws,
        point_indices,
        point_locations,
        repeats,
        save_path,
        title_prefix="Random fixed-location",
        variable_label="w",
    )
    joint_save_path = None
    if save_path:
        root, ext = os.path.splitext(save_path)
        joint_save_path = f"{root}_joint{ext or '.png'}"
        _plot_two_point_joint_density(
            w_draws,
            point_indices,
            point_locations,
            repeats,
            joint_save_path,
            variable_label="w",
        )

    npz_path = None
    if save_path:
        npz_path = os.path.splitext(save_path)[0] + ".npz"
        np.savez_compressed(
            npz_path,
            w_draws=w_draws,
            point_indices=point_indices,
            point_locations=point_locations,
            repeats=repeats,
        )

    print("随机固定坐标 w 重复模拟完成")
    print(f"  point_indices_in_all_points: {point_indices.tolist()}")
    print(f"  point_locations: {point_locations.tolist()}")
    print(f"  density_plot: {save_path}")
    if joint_save_path:
        print(f"  joint_density_plot: {joint_save_path}")
    if npz_path:
        print(f"  samples_npz: {npz_path}")
    return w_draws, point_indices, point_locations

def _plot_two_point_density(draws, point_indices, point_locations, repeats, save_path, title_prefix="Fixed-location", variable_label="y"):
    """Plot two empirical densities with matched Gaussian references."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    palette = ["#1f77b4", "#d62728"]

    for j, ax in enumerate(axes):
        samples = draws[:, j]
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        std = max(std, 1e-8)
        xs = np.linspace(np.min(samples) - 0.5 * std, np.max(samples) + 0.5 * std, 300)
        kde = gaussian_kde(samples)

        ax.hist(samples, bins=24, density=True, alpha=0.30, color=palette[j], edgecolor="white")
        ax.plot(xs, kde(xs), color=palette[j], linewidth=2.2, label="Empirical KDE")
        ax.plot(xs, norm.pdf(xs, loc=mean, scale=std), color="black", linestyle="--", linewidth=1.8, label="Matched Gaussian")
        ax.axvline(mean, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title(
            f"Point {j + 1}: index={point_indices[j]}, "
            f"loc=({point_locations[j, 0]:.3f}, {point_locations[j, 1]:.3f})"
        )
        ax.set_xlabel(f"Repeated {variable_label} value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(f"{title_prefix} {variable_label} density over {repeats} repeated NNnGP simulations", fontsize=13)

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_two_point_joint_density(draws, point_indices, point_locations, repeats, save_path, variable_label="y"):
    """Plot the empirical joint density of the two fixed-location values."""
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

    ax.set_xlabel(f"Point 1 {variable_label}, index={point_indices[0]}, loc=({point_locations[0, 0]:.3f}, {point_locations[0, 1]:.3f})")
    ax.set_ylabel(f"Point 2 {variable_label}, index={point_indices[1]}, loc=({point_locations[1, 0]:.3f}, {point_locations[1, 1]:.3f})")
    ax.set_title(f"Joint density of two fixed-location {variable_label} values ({repeats} repeats), corr={corr:.3f}")
    fig.colorbar(contour, ax=ax, label="Joint density")

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_fixed_y_density_repeats(
    data_path=os.path.join(RESULTS_DIR, "test_synthetic_data.npz"),
    repeats=200,
    point_indices=None,
    random_seed=2026,
    save_path=os.path.join(RESULTS_DIR, "fixed_y_density_repeats.png"),
):
    """
    Fix two reference locations, repeatedly simulate y at those locations, and plot densities.

    The dashed curve is a Gaussian with the same sample mean/std, used as a visual check
    for skewness, heavy tails, or multimodality.
    """
    data = load_data(data_path)
    S = np.asarray(data["S"], dtype=np.float64)
    X_S = np.asarray(data["X_S"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    true_params = data["true_params"]
    m = int(true_params["m"])

    if len(S) <= m:
        raise ValueError("Need len(S) > m to simulate NNnGP reference locations.")

    rng = np.random.default_rng(random_seed)
    if point_indices is None:
        point_indices = np.sort(rng.choice(np.arange(m, len(S)), size=2, replace=False))
    else:
        point_indices = np.asarray(point_indices, dtype=int)
        if point_indices.shape != (2,):
            raise ValueError("point_indices must contain exactly two indices.")

    precomputed = _precompute_fixed_s_terms(S, m, true_params)
    y_draws = np.zeros((repeats, 2), dtype=np.float64)
    for r in range(repeats):
        y_S = _simulate_fixed_s_y(S, X_S, Z, true_params, precomputed, rng)
        y_draws[r] = y_S[point_indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    palette = ["#1f77b4", "#d62728"]

    for j, ax in enumerate(axes):
        samples = y_draws[:, j]
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=1))
        xs = np.linspace(np.min(samples) - 0.5 * std, np.max(samples) + 0.5 * std, 300)
        kde = gaussian_kde(samples)

        ax.hist(samples, bins=24, density=True, alpha=0.30, color=palette[j], edgecolor="white")
        ax.plot(xs, kde(xs), color=palette[j], linewidth=2.2, label="Empirical KDE")
        ax.plot(xs, norm.pdf(xs, loc=mean, scale=std), color="black", linestyle="--", linewidth=1.8, label="Matched Gaussian")
        ax.axvline(mean, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title(f"Point {j + 1}: index={point_indices[j]}, loc=({S[point_indices[j], 0]:.3f}, {S[point_indices[j], 1]:.3f})")
        ax.set_xlabel("Repeated y value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(f"Fixed-location y density over {repeats} repeated NNnGP simulations", fontsize=13)

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    npz_path = None
    if save_path:
        npz_path = os.path.splitext(save_path)[0] + ".npz"
        np.savez_compressed(
            npz_path,
            y_draws=y_draws,
            point_indices=point_indices,
            point_locations=S[point_indices],
            repeats=repeats,
        )

    print("固定点 y 重复模拟完成")
    print(f"  point_indices: {point_indices.tolist()}")
    print(f"  point_locations: {S[point_indices].tolist()}")
    print(f"  density_plot: {save_path}")
    if npz_path:
        print(f"  samples_npz: {npz_path}")
    return y_draws, point_indices
    
# ==============================================================================
# 直接运行此文件时执行的测试代码
# 用法: python data_utils.py
# ==============================================================================
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 设置全局随机种子，保证结果可复现
    np.random.seed(42)

    grid_size = 100
    domain_size = 5.0
    k = 500
    m = 10
    m_tilde = 50
    matern_params = (1.0, 0.2)
    tau_params = (1.0, 0.5)
    g_params = (0.0, -2.0, 1.0)
    parametric_g_params = (1.5, 1.5, 0.0)
    # g(v) = amplitude * tanh(slope * mean(v) + bias)
    # amplitude controls output range [-amplitude, amplitude].
    # slope controls how quickly tanh saturates; bias shifts g positive/negative.
    beta = (0.5, 0.5, -0.5)
    sigma_epsilon = 0.1
    
    print("=" * 70)
    print("NNnGP合成数据生成测试程序")
    print("=" * 70)
    print("\n正在生成合成数据...")
    print("参数设置:")
    print(f"  - 空间域: [0,{domain_size:g}]^2 {grid_size}×{grid_size}网格 (共{grid_size ** 2}个点)")
    print(f"  - 参考点数量: {k}个")
    print(f"  - 最近邻数量: {m}个")
    print(f"  - FIC诱导点数量: {m_tilde}个")
    print(f"  - g函数: parametric tanh, params={parametric_g_params}")
    print("  - 协变量: 常数项 + 经纬度")
    print("-" * 70)

    # 统一输出目录：results 下的新文件夹
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成数据（使用默认参数，可根据需要调整）
    data_path = generate_nnngp_data(
        k=k,
        m=m,
        m_tilde=m_tilde,
        matern_params=matern_params,
        tau_params=tau_params,
        g_params=g_params,
        parametric_g_params=parametric_g_params,
        beta=beta,
        sigma_epsilon=sigma_epsilon,
        grid_size=grid_size,
        domain_size=domain_size,
        save_path=os.path.join(output_dir, "test_synthetic_data.npz")
    )
    
    # 加载生成的数据进行验证
    print("\n正在验证生成的数据...")
    data = load_data(data_path)
    
    # 打印数据基本信息
    print("\n" + "=" * 70)
    print("数据验证结果")
    print("=" * 70)
    print(f"总点数: {len(data['all_points'])} ({grid_size}×{grid_size}网格)")
    print(f"参考点数量: {len(data['S'])}")
    print(f"预测点数量: {len(data['U'])}")
    print(f"参考点索引数量: {len(data['s_indices'])}")
    print(f"预测点索引数量: {len(data['u_indices'])}")
    print(f"协变量矩阵形状 (X_S): {data['X_S'].shape}")
    print(f"协变量矩阵形状 (X_U): {data['X_U'].shape}")
    print(f"潜场w_S形状: {data['w_S'].shape}")
    print(f"潜场w_U形状: {data['w_U'].shape}")
    print(f"观测y_S形状: {data['y_S'].shape}")
    print(f"观测y_U形状: {data['y_U'].shape}")
    
    print("\n真实参数:")
    for key, value in data['true_params'].items():
        print(f"  {key}: {value}")
    
    # 构建完整的空间场（用于可视化）
    w_all = np.zeros(len(data['all_points']))
    w_all[data['s_indices']] = data['w_S']
    w_all[data['u_indices']] = data['w_U']
    
    y_all = np.zeros(len(data['all_points']))
    y_all[data['s_indices']] = data['y_S']
    y_all[data['u_indices']] = data['y_U']
    
    # 数据统计信息
    print("\n数据统计信息:")
    print(f"  潜场w - 均值: {np.mean(w_all):.4f}, 标准差: {np.std(w_all):.4f}")
    print(f"  潜场w - 最小值: {np.min(w_all):.4f}, 最大值: {np.max(w_all):.4f}")
    print(f"  观测y - 均值: {np.mean(y_all):.4f}, 标准差: {np.std(y_all):.4f}")
    print(f"  y-w - 均值: {np.mean(y_all - w_all):.4f}, 标准差: {np.std(y_all - w_all):.4f}")
    print(f"  y-w - 最小值: {np.min(y_all - w_all):.4f}, 最大值: {np.max(y_all - w_all):.4f}")
    print(f"  corr(w, y): {np.corrcoef(w_all, y_all)[0, 1]:.4f}")
    print(f"  观测噪声标准差: {data['true_params']['sigma_epsilon']:.4f}")
    
    # 生成可视化图
    print("\n" + "=" * 70)
    print("正在生成可视化图...")

    _, random_point_indices, random_marked_points = plot_random_w_density_repeats(
        data_path=data_path,
        repeats=2000,
        random_seed=2026,
        save_path=os.path.join(output_dir, "fixed_w_density_repeats.png"),
    )
    
    # 1. 真实潜场热力图
    plot_spatial_field(
        data['all_points'],
        w_all,
        title="True Latent Field w (NNnGP)",
        save_path=os.path.join(output_dir, "test_true_latent_field.png")
    )
    
    # 2. 带噪声的观测数据热力图
    plot_spatial_field(
        data['all_points'],
        y_all,
        title="Observed Data y (with Gaussian Noise)",
        save_path=os.path.join(output_dir, "test_observed_data.png"),
        marked_points=random_marked_points,
        marked_labels=[str(i + 1) for i in range(len(random_point_indices))],
    )
    
    # 3. 参考点分布可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data['all_points'][:, 0], 
        data['all_points'][:, 1], 
        c='lightgray', 
        s=1, 
        alpha=0.5,
        label='All Grid Points'
    )
    plt.scatter(
        data['S'][:, 0], 
        data['S'][:, 1], 
        c='crimson', 
        s=25, 
        edgecolor='black',
        linewidth=0.5,
        label=f"Reference Points (k={len(data['S'])})"
    )
    plt.title('Reference Points Distribution (Maximin Ordered)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_reference_points_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 潜场与观测数据的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(w_all, y_all, alpha=0.5, s=5)
    plt.plot([np.min(w_all), np.max(w_all)], [np.min(w_all), np.max(w_all)], 'r--', label='y = w')
    plt.title('Latent Field w vs Observed Data y')
    plt.xlabel('True Latent Value w')
    plt.ylabel('Observed Value y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_w_vs_y.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print(f"测试完成！所有结果已保存至: {output_dir}")
    print(f"  1. {os.path.join(output_dir, 'test_synthetic_data.npz')} - 生成的合成数据文件")
    print(f"  2. {os.path.join(output_dir, 'test_true_latent_field.png')} - 真实潜场热力图")
    print(f"  3. {os.path.join(output_dir, 'test_observed_data.png')} - 带随机固定点标记的观测数据热力图")
    print(f"  4. {os.path.join(output_dir, 'fixed_w_density_repeats.png')} - 随机固定点 w 边际密度图")
    print(f"  5. {os.path.join(output_dir, 'fixed_w_density_repeats_joint.png')} - 随机固定点 w 联合密度图")
    print(f"  6. {os.path.join(output_dir, 'test_reference_points_distribution.png')} - 参考点分布图")
    print(f"  7. {os.path.join(output_dir, 'test_w_vs_y.png')} - 潜场与观测数据对比图")
    print("=" * 70)
    print("\n你可以通过以下代码在其他文件中加载此数据:")
    print("from data_utils import load_data")
    print(f"data = load_data('{os.path.join(output_dir, 'test_synthetic_data.npz')}')")
