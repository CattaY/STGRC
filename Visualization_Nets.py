import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import importlib
import json

import pandas as pd
importlib.invalidate_caches()
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 设置后端为 Agg
import os
import sys
# 定义模块绝对路径
MODULE_PATH = "/home/yangna/JetBrains/graphRC"
MODULE_FILE = "func_graphRC.py"
FULL_PATH = os.path.join(MODULE_PATH, MODULE_FILE)
# 验证文件是否存在
if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"找不到模块文件: {FULL_PATH}")
# 添加路径到Python搜索路径
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)  # 优先搜索
# 导入模块
try:
    import func_graphRC as fg
    print("✅ 模块导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"当前Python路径: {sys.path}")
import numpy as np
from scipy.stats import pearsonr

# -----------------------------
# Metrics for "critical fluctuation"
# -----------------------------
def node_fluctuation_metric(X_mnf, mode="trace_cov", eps=1e-12):
    """
    X_mnf: (m, n, f)
    
    Args:
        X_mnf: Input data with shape (m, n, f) where m=time, n=nodes, f=features
        mode: Calculation mode - 'trace_cov', 'norm_std', 'pc1_var', 'mean', or 'mean_std'
        eps: Small constant for numerical stability in log scale
        
    Returns:
        size_metric (n,): Metric per node (for node size)
        color_metric (n,): Metric per node (for node color)
        
    Modes:
        - 'trace_cov': Trace of covariance matrix (fluctuation measure)
        - 'norm_std': Standard deviation of L2 norm (fluctuation measure)
        - 'pc1_var': Variance of first principal component (collective fluctuation)
        - 'mean': Mean value across time (state level measure, no log scale)
        - 'mean_std': Mean value for size, std value for color (no log scale)
    """
    m, n, f = X_mnf.shape

    if mode == "trace_cov":
        # fluctuation = trace of covariance across time within window, per node
        # Cov over time for each node: fxf, trace = sum of variances across f dims
        mu = X_mnf.mean(axis=0, keepdims=True)           # (1,n,f)
        Xc = X_mnf - mu                                  # (m,n,f)
        var = (Xc ** 2).mean(axis=0)                      # (n,f) ~ variance per dim
        tr = var.sum(axis=1)                              # (n,)
        metric = tr

    elif mode == "norm_std":
        # fluctuation = std over time of L2 norm of state vector
        norm = np.linalg.norm(X_mnf, axis=-1)             # (m,n)
        metric = norm.std(axis=0)                         # (n,)

    elif mode == "pc1_var":
        # fluctuation = variance of projection onto PC1 (computed per window across nodes' mean vectors)
        # This emphasizes collective fluctuation modes; slightly more complex and less "local".
        mu = X_mnf.mean(axis=0)                           # (n,f)
        mu0 = mu - mu.mean(axis=0, keepdims=True)
        # SVD for PC1
        _, _, VT = np.linalg.svd(mu0, full_matrices=False)
        pc1 = VT[0]                                       # (f,)
        proj = (X_mnf @ pc1)                              # (m,n)
        metric = proj.var(axis=0)

    elif mode == "mean":
        # fluctuation = mean value of state vector across time, per node
        # This shows the average state level rather than fluctuation
        metric = X_mnf.mean(axis=0).mean(axis=1)          # (n,)
        # For mean mode, we don't use log scale
        color_metric = metric
        size_metric = metric
        return size_metric, color_metric

    elif mode == "mean_std":
        # size = mean value, color = std value (no log scale)
        # Mean value for node size
        size_metric = X_mnf.mean(axis=0).mean(axis=1)     # (n,)
        # Standard deviation for node color
        color_metric = X_mnf.std(axis=0).mean(axis=1)     # (n,)
        return size_metric, color_metric

    else:
        raise ValueError("mode must be one of {'trace_cov','norm_std','pc1_var','mean','mean_std'}")

    # color_metric: log-scale is often nicer for heavy-tailed fluctuation
    color_metric = np.log10(metric + eps)
    size_metric = metric
    return size_metric, color_metric


# -----------------------------
# Edge sparsification (directed, signed)
# -----------------------------
def sparsify_directed(A, topk_out=3, topk_in=0, keep_self=False):
    """
    Keep top-k outgoing edges per node by |A[i,j]|.
    Optionally also keep top-k incoming edges per node.
    Returns a sparse matrix A_sp with others zeroed.
    """
    n = A.shape[0]
    mask = np.zeros_like(A, dtype=bool)

    if topk_out > 0:
        for i in range(n):
            row = np.abs(A[i])
            if not keep_self:
                row = row.copy()
                row[i] = -np.inf
            idx = np.argsort(row)[::-1]
            idx = idx[:topk_out]
            mask[i, idx] = True

    if topk_in > 0:
        for j in range(n):
            col = np.abs(A[:, j])
            if not keep_self:
                col = col.copy()
                col[j] = -np.inf
            idx = np.argsort(col)[::-1]
            idx = idx[:topk_in]
            mask[idx, j] = True

    A_sp = np.where(mask, A, 0.0)
    if not keep_self:
        np.fill_diagonal(A_sp, 0.0)
    return A_sp


# -----------------------------
# Layout: fixed positions from reference adjacency
# -----------------------------
def build_fixed_layout(A_list, layout="spring", seed=0):
    """
    Build node positions from reference adjacency: mean(|A_k|).
    """
    A_ref = np.mean(np.abs(np.stack(A_list, axis=0)), axis=0)
    G_ref = nx.from_numpy_array(A_ref, create_using=nx.Graph)
    if layout == "spring":
        pos = nx.spring_layout(G_ref, seed=seed)
    elif layout == "spectral":
        pos = nx.spectral_layout(G_ref)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_ref)
    elif layout == "circular":
        pos = nx.circular_layout(G_ref)
    elif layout == "shell":
        # Use shell layout with multiple shells for better separation
        n = A_ref.shape[0]
        # Split nodes into multiple shells based on degree
        degrees = np.sum(np.abs(A_ref) > 0, axis=1)
        shell1 = [i for i in range(n) if degrees[i] <= np.percentile(degrees, 33)]
        shell2 = [i for i in range(n) if np.percentile(degrees, 33) < degrees[i] <= np.percentile(degrees, 66)]
        shell3 = [i for i in range(n) if degrees[i] > np.percentile(degrees, 66)]
        shells = [shell1, shell2, shell3]
        pos = nx.shell_layout(G_ref, nlist=shells)
    else:
        raise ValueError("layout must be one of: 'spring', 'spectral', 'kamada_kawai', 'circular', 'shell'")
    return pos


# -----------------------------
# Main plotting function (paper-grade)
# -----------------------------
def plot_directed_signed_snapshots(
    A_list,
    X_list,
    window_indices,
    titles=None,
    fluct_mode="trace_cov",
    topk_out=3,
    topk_in=0,
    layout="spring",
    seed=0,
    node_size_range=(200, 1200),    # in points^2
    edge_width_range=(0.5, 2.0),
    node_cmap="viridis",
    edge_cmap="coolwarm",
    edge_alpha=0.65,
    arrow_size=10,
    figsize=None,
):
    """
    Paper-grade node-link snapshots for directed, signed networks.

    A_list: list of (n,n) arrays, length K
    X_list: list of (m,n,f) arrays, length K
    window_indices: list of ints (which windows to plot)
    titles: list of titles for each window (optional)
    """

    assert len(A_list) == len(X_list), "A_list and X_list must have same length"
    K = len(A_list)
    sel = list(window_indices)
    assert all(0 <= k < K for k in sel), "window_indices out of range"
    num = len(sel)
    if titles is None:
        titles = [f"Window {k}" for k in sel]
    else:
        assert len(titles) == num

    # Fixed layout
    pos = build_fixed_layout(A_list, layout=layout, seed=seed)
    n = A_list[0].shape[0]

    # Compute global scales across selected windows (for comparability)
    all_node_size = []
    all_node_color = []
    all_edge_abs = []
    all_edge_signed = []

    A_sp_list = []
    node_metrics = []

    for k in sel:
        A = A_list[k]
        X = X_list[k]
        A_sp = sparsify_directed(A, topk_out=topk_out, topk_in=topk_in, keep_self=False)
        A_sp_list.append(A_sp)

        s_metric, c_metric = node_fluctuation_metric(X, mode=fluct_mode)
        node_metrics.append((s_metric, c_metric))

        all_node_size.append(s_metric)
        all_node_color.append(c_metric)

        # edge scales
        nz = np.nonzero(A_sp)
        if len(nz[0]) > 0:
            vals = A_sp[nz]
            all_edge_abs.append(np.abs(vals))
            all_edge_signed.append(vals)

    all_node_size = np.concatenate(all_node_size) if len(all_node_size) else np.array([0.0])
    all_node_color = np.concatenate(all_node_color) if len(all_node_color) else np.array([0.0])

    # Node size normalization: use robust percentiles to avoid extreme outliers
    s_lo, s_hi = np.percentile(all_node_size, [5, 95])
    if np.isclose(s_hi, s_lo):
        s_lo, s_hi = all_node_size.min(), all_node_size.max() + 1e-12

    # Node color normalization (log fluctuation)
    c_lo, c_hi = np.percentile(all_node_color, [5, 95])
    if np.isclose(c_hi, c_lo):
        c_lo, c_hi = all_node_color.min(), all_node_color.max() + 1e-12
    node_norm = Normalize(vmin=c_lo, vmax=c_hi)

    # Edge width normalization (abs weight)
    if len(all_edge_abs) > 0:
        edge_abs = np.concatenate(all_edge_abs)
        w_lo, w_hi = np.percentile(edge_abs, [5, 95])
    else:
        w_lo, w_hi = 0.0, 1.0
    if np.isclose(w_hi, w_lo):
        w_hi = w_lo + 1e-12
    edge_w_norm = Normalize(vmin=w_lo, vmax=w_hi)

    # Edge color normalization (signed, centered at 0)
    if len(all_edge_signed) > 0:
        edge_signed = np.concatenate(all_edge_signed)
        v = np.percentile(np.abs(edge_signed), 95)
        v = max(v, 1e-12)
    else:
        v = 1.0
    edge_c_norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    # Figure layout
    if figsize is None:
        figsize = (5.0 * num, 5.0)
    fig, axes = plt.subplots(1, num, figsize=figsize, constrained_layout=True)
    if num == 1:
        axes = [axes]

    # Helper: map node sizes
    def map_node_sizes(s_metric):
        s = np.clip((s_metric - s_lo) / (s_hi - s_lo + 1e-12), 0.0, 1.0)
        s_min, s_max = node_size_range
        return s_min + (s_max - s_min) * s

    # Helper: map edge widths
    def map_edge_widths(abs_w):
        z = edge_w_norm(abs_w)
        w_min, w_max = edge_width_range
        return w_min + (w_max - w_min) * z

    # Draw each snapshot
    for ax, k, title, (s_metric, c_metric), A_sp in zip(axes, sel, titles, node_metrics, A_sp_list):
        # Undirected graph
        G = nx.from_numpy_array(A_sp, create_using=nx.Graph)

        # Node visuals
        node_sizes = map_node_sizes(s_metric)
        node_colors = node_norm(c_metric)

        # Edge visuals
        edges = list(G.edges())
        if len(edges) > 0:
            e_vals = np.array([A_sp[i, j] for i, j in edges])
            e_abs = np.abs(e_vals)
            widths = map_edge_widths(e_abs)
            e_colors = plt.get_cmap(edge_cmap)(edge_c_norm(e_vals))
        else:
            widths = []
            e_colors = []

        # Draw nodes first (so edges appear on top)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=node_cmap,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            edgecolors="k",
        )

        # Draw edges (no arrows for undirected graph)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=widths,
            edge_color=e_colors,
            alpha=edge_alpha,
        )

        ax.set_title(title)
        ax.axis("off")

    # -----------------------------
    # Shared colorbars (node + edge)
    # -----------------------------
    # Node colorbar: std value (no log scale)
    sm_node = ScalarMappable(norm=node_norm, cmap=node_cmap)
    sm_node.set_array([])
    cbar_node = fig.colorbar(sm_node, ax=axes, fraction=0.025, pad=0.02)
    cbar_node.set_label(r"Node std (standard deviation)")

    # Edge colorbar: signed coupling
    sm_edge = ScalarMappable(norm=edge_c_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, ax=axes, fraction=0.025, pad=0.08)
    cbar_edge.set_label("Edge coupling (signed)")

    # -----------------------------
    # Node size legend (mean value magnitude)
    # -----------------------------
    # Pick representative metric values for legend (in original scale)
    s_vals = np.percentile(all_node_size, [20, 50, 80])
    s_leg_sizes = map_node_sizes(s_vals)
    handles = [
        Line2D([0], [0], marker='o', color='w', label=f"{v:.2g}",
               markerfacecolor='gray', markeredgecolor='k',
               markersize=np.sqrt(ms/np.pi))  # rough mapping points^2 -> points
        for v, ms in zip(s_vals, s_leg_sizes)
    ]
    # Put legend on last axis
    axes[-1].legend(
        handles=handles,
        title="Node mean\n(original scale)",
        loc="lower left",
        frameon=True,
        borderpad=0.6,
        handletextpad=0.8
    )

    return fig, axes

# 读取设置参数
def flatten_dict_generator(d, parent_key=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten_dict_generator(v, k)
        else:
            yield k, v

# -----------------------------

if __name__ == '__main__':
    path = "/home/yangna/JetBrains/graphRC/Figures/EWS/[2026-01-30  22:13:45] mutualistic_weight_reduction"
    # EWS_ats = 3 # number of windows at EWS point
    color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2] #['#0EF0B1', '#0E9CF0', '#F06A0E']
    with open(path+'/arguments.json', 'r') as f:
        arguments = json.load(f)
    arg = dict(flatten_dict_generator(arguments))
    label = arg['label']
    tar0, tar1 = arg['target']
    EWS_idx = arg['EWS_win']
    # #############读取数据#############
    Model_list = ["Henon", "ADVP", "Lorenz"]
    data_all = np.load(path + '/data_noise_%s_ns=%.1f.npz' % (arg["label"], arg["ns"]))
    data_noise = data_all['data_noise']
    ubase = data_all['A0']
    coupling = data_all['coupling']
    norm_term = np.std(data_noise[tar0, tar1, :arg['num_win']*arg['win_step']+arg['win_m']])
    print("✅ 读取结果完成！")

    [f, nodes, m] = data_noise.shape # features, nodes, total_length
    arg['datashape'] = (f, nodes, m)
    num_zones = arg["num_win"]
    myzones = []
    for i in range(num_zones):
        myzones.append(range(0 + i * arg['win_step'], 0 + i * arg['win_step'] + arg['win_m'], 1))
    batches = [data_noise[:, :, myzones[i]]
    for i in range(num_zones)
    ]
    A_list = [ubase * (1-coupling[j*arg['win_step']]) for j in range(num_zones)]
    # ######### 计算波动模式 ###############
    # 使用 circular 布局来减少边交叉
    fig, axes = plot_directed_signed_snapshots(
        A_list=A_list,
        X_list=batches,
        window_indices=[100, 200, 434, 899],
        titles=["Pre", "Rising", "Near-critical", "Post"],
        fluct_mode="mean", #'trace_cov', 'norm_std', 'pc1_var', 'mean', 'mean_std'
        topk_out=3,
        topk_in=1,
        layout="circular",  # 使用圆形布局减少边交叉
        seed=0
    )
    plt.savefig(path+"/snapshots.svg", dpi=300)
    plt.close()
    print("✅ 可视化网络变化完成！")
