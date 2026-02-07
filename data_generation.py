#!/usr/bin/env python3
"""
数据生成模块
支持gLV和CLorenz两种系统的数据生成
"""

import numpy as np
import os
import datetime
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_mixture(n=20, eta=0.45, seed=None):
    """生成对称和反对称混合的连接矩阵（用于gLV）"""
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)
    R = rng.uniform(-1, 1, size=(n, n))
    S = (R + R.T)/2                    # 对称部分
    K = (R - R.T)/2                    # 反对称部分
    U_base = (1-eta)*S + eta*K         # eta 反对称占比
    np.fill_diagonal(U_base, 0.0)
    # 行归一
    A0 = U_base / (np.sum(np.abs(U_base), axis=1, keepdims=True)+1e-12)
    return A0


def generate_ring(nodes, epsilon):
    """环状连接：每个节点只连接左右邻居（用于gLV）"""
    U = np.zeros((nodes, nodes))
    for i in range(nodes):
        U[i, (i-1) % nodes] = epsilon  # 左邻居
        U[i, (i+1) % nodes] = epsilon  # 右邻居
    return U


def ODE_gLV(t, x, alpha, D, N, U_intra, U_inter, nodes, rate, beta=0.05):
    """
    gLV模型微分方程
    U_intra: N*N，节点内物种相互作用矩阵
    U_inter: nodes*nodes，节点间相互作用矩阵
    """
    X = np.asarray(x, dtype=float).reshape(nodes, N)
    X = np.clip(X, 0, None)  # 防止负值
    # intras
    intras = np.zeros((nodes, N))
    for i in range(nodes):
        intras[i, :] = np.dot(X[i, :], U_intra.T)
    # coupling
    coupling = np.zeros((nodes, N))
    for k in range(N):
        W = alpha * U_inter
        coupling[:, k] = np.dot(W, X[:, k])
    # 增长项-节点内竞争-节点间耦合-种内竞争
    dX = X * (rate - intras - coupling - beta * X) + D
    return dX.reshape(-1)


def ODE_CLorenz(t, x, sigma, b, rho, C, nodes, N):
    """Vectorized ODE for coupled Lorenz systems"""
    couple = int(N/3)
    X = x.reshape(nodes, N)
    dxdt = np.zeros((nodes, N))
    
    # Add noise once for all nodes
    nt = 0.05 * np.random.uniform(-1, 1, size=(nodes, N))
    
    # Vectorized computation for all nodes
    # For each node, compute derivatives for all 3-variable groups
    for j in range(couple):
        idx = 3 * j
        if j == 0:
            # First group: no coupling from previous group
            dxdt[:, idx] = sigma * (X[:, idx+1] - X[:, idx]) + nt[:, idx]
        else:
            # Subsequent groups: coupled from previous group
            dxdt[:, idx] = sigma * (X[:, idx+1] - X[:, idx]) + C * X[:, idx-3]**2 + nt[:, idx]
        
        dxdt[:, idx+1] = rho * X[:, idx] - X[:, idx+1] - X[:, idx] * X[:, idx+2] + nt[:, idx+1]
        dxdt[:, idx+2] = -b * X[:, idx+2] + X[:, idx] * X[:, idx+1] + nt[:, idx+2]
    
    return dxdt.reshape(-1)


def betaspace(A, x):
    """计算有效相互作用强度"""
    sAout = np.sum(A)
    x_nss = A.dot(x)
    if np.sum(A) == 0:
        return 0.0, 0.0
    else:
        beta_eff = np.sum((A.dot(A))) / np.sum(A)
        x_eff = np.sum(x_nss) / np.sum(sAout)
        return x_eff, beta_eff


def check_oscillation_single(data, threshold=0.1):
    """
    检查数据是否有振荡现象
    通过计算差分的正负变化来判断
    threshold -- 阈值，通过调整该值可以调整判断灵敏度
    """
    if len(data) < 100:
        return False
    amplitude = np.max(data) - np.min(data)
    if amplitude <  threshold:
        return False
    t = np.arange(len(data))
    diff = np.diff(data)
    zero_crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(zero_crossings) < 4:
        return False
    periods=[]
    # 计算连续过零点的时间间隔
    zero_times = t[zero_crossings]
    period_intervals = np.diff(zero_times)
    # 取完整周期（两个连续过零点）
    for i in range(0, len(period_intervals)-1, 2):
        if i+1 < len(period_intervals):
            periods.append(period_intervals[i] + period_intervals[i+1])
    osc_start = zero_times[0]
    osc_end = zero_times[min(2 * len(periods), len(zero_times) - 1)]
    osc_duration = osc_end - osc_start
    if len(periods) < 2 or osc_duration > 0.8 * len(data): # 保证有一定的平稳期和振荡期
        return False
    
    return True


def check_oscillation(data, threshold=0.1):
    """
    检查数据是否有振荡现象
    通过计算差分的正负变化来判断
    threshold -- 阈值，通过调整该值可以调整判断灵敏度
    """
    if len(data) < 100:
        return False
    if data.ndim == 1:
        return check_oscillation_single(data, threshold)
     # 多维数据，检查每个维度
    if data.ndim == 2:
        for i in range(data.shape[1]):
            if check_oscillation_single(data[:, i], threshold):
                return True
    if data.ndim == 3:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if check_oscillation_single(data[:, i, j], threshold):
                    return True
    else:
        return False


def simulate_gLV_data(n_nodes=10, n_features=3, total_length=4000, seed=None):
    """模拟一条gLV数据"""
    rng = np.random.RandomState(seed)
    
    # 参数设置
    discard = 1000  # 丢弃前1000点，让系统稳定
    first_fixed = 0  # 固定参数阶段
    total_steps = total_length+discard
    tail = total_steps - first_fixed
    
    # fraction_reduce从0.98缓慢降到0.75，产生临界点
    frac_high = 0.98
    frac_low = 0.75
    fraction_reduce = np.concatenate((np.full(first_fixed, frac_high), 
                                     np.linspace(frac_high, frac_low, tail)))
    
    # 生成连接矩阵
    U_intra = generate_mixture(n=n_features, eta=0.6, seed=seed)
    A0 = generate_ring(nodes=n_nodes, epsilon=1)
    # 初始状态
    rate = rng.uniform(0.15, 0.3, size=(n_nodes, n_features))
    x = np.abs(rng.normal(0.2, 0.8, size=(n_nodes, n_features))).reshape(-1)
    
    # 存储数据
    X_trim = np.zeros((total_steps - discard, n_features * n_nodes))
    outputs = np.zeros((total_steps, 4))
    
    trim_idx = 0
    for t in range(total_steps):
        current_A = A0 * (1.0 - fraction_reduce[t])
        x_node = x.reshape(n_features, n_nodes)
        x_node = np.sum(x_node, axis=0)
        x_eff, beta_eff = betaspace(current_A, x_node)
        mean_x = np.mean(x)
        
        outputs[t, 0] = mean_x
        outputs[t, 1] = x_eff
        outputs[t, 2] = beta_eff
        outputs[t, 3] = fraction_reduce[t]
        
        # 参数D作为扰动源，产生振荡
        D = 0.1  #+ 0.05 * np.sin(t * 0.02)  # 添加周期性扰动
        
        system_fun = lambda tt, yy: ODE_gLV(tt, yy, alpha=fraction_reduce[t], 
                                            D=D, N=n_features, U_intra=U_intra, U_inter=current_A,
                                            nodes=n_nodes, rate=rate, beta=0.05)
        sol = solve_ivp(fun=system_fun, t_span=(0, 1), y0=x, method='RK45', 
                       atol=1e-6, rtol=1e-6)
        x = sol.y[:, -1]
        
        if t >= discard:
            X_trim[trim_idx, :] = x
            trim_idx += 1
    
    outputs = outputs[discard:, :]
    X_trim = X_trim.reshape(-1, n_nodes, n_features)
    
    return outputs, X_trim, A0


def simulate_CLorenz_data(n=8, N=6, total_length=4000, seed=None, 
                          frac_high=-4, frac_low=4):
    """模拟一条CLorenz数据"""
    total_steps = total_length + 1000
    discard = 1000
    first_fixed = 1000
    tail = total_steps - first_fixed
    fraction_reduce = np.concatenate((np.full(first_fixed, frac_high), 
                                     np.linspace(frac_high, frac_low, tail)))

    N = 3*int(N/3)
    # 创建初始状态：每个节点有N个变量（3个一组的Lorenz系统）
    x0 = np.tile([2.0, 2.0, -2.0], N // 3)  # 基础模式 [2,2,-2, 2,2,-2, ...]
    x = np.random.rand(n, N) * 0.1 + np.tile(x0, n).reshape(n, N)
    x = x.reshape(-1) # 展平为 (nodes*N,)
    
    if total_steps > discard:
        X_trim = np.zeros((total_steps - discard, N*n))
    else:
        X_trim = np.empty((0, N*n))

    trim_idx = 0
    outputs = np.zeros((total_steps, 4))
    
    rtol = 1e-3
    atol = 1e-5
    
    for t in range(total_steps):
        mean_x = np.mean(x)
        outputs[t, 0] = mean_x
        outputs[t, 3] = fraction_reduce[t]
        system_fun = lambda tt, yy: ODE_CLorenz(tt, yy,
                                                rho=fraction_reduce[t], nodes=n, N=N,
                                                sigma=4, b=8/3, C=0.1)
        sol = solve_ivp(fun=system_fun, t_span=(0, 1), y0=x, method='RK45', 
                       atol=atol, rtol=rtol, max_step=0.2)  # 增加 max_step
        x = sol.y[:, -1]

        if t >= discard:
            X_trim[trim_idx, :] = x
            trim_idx += 1
    
    outputs = outputs[discard:,:]
    X_trim = X_trim.reshape(-1, n, N)
    
    return outputs, X_trim


def generate_and_filter_data(system_type='gLV', num_datasets=100, 
                           n_nodes=10, n_features=3, seed=None,
                           total_length=4000, output_dir=None):
    """
    生成并筛选有振荡的数据
    
    Parameters:
    -----------
    system_type : str
        系统类型，'gLV' 或 'CLorenz'
    num_datasets : int
        需要生成的有效数据数量
    n_nodes : int
        节点数量
    n_features : int
        特征数量（对于CLorenz，实际使用3的倍数）
    seed : int
        随机种子
    total_length : int
        数据总长度
    output_dir : str
        输出目录
        
    Returns:
    --------
    valid_count : int
        生成的有效数据数量
    all_metadata : list
        元数据列表
    """
    if output_dir is None:
        output_dir = f"/home/yangna/JetBrains/graphRC/Data/{system_type}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    valid_count = 0
    all_metadata = []
    
    print(f"开始生成 {num_datasets} 条{system_type}数据...")
    i = 0
    
    while valid_count < num_datasets:
        i += 1
        current_seed = seed + i if seed is not None else None
        print(f"生成第 {i} 条数据 (有效数据 {valid_count+1}/{num_datasets}, seed={current_seed})...")
        try:
            if system_type == 'gLV':
                outputs, X_trim, A0 = simulate_gLV_data(
                    n_nodes=n_nodes, 
                    n_features=n_features, 
                    total_length=total_length,
                    seed=current_seed
                )
                # 检查是否有振荡
                if_valid = check_oscillation(X_trim, threshold=0.2)
            elif system_type == 'CLorenz':
                outputs, X_trim = simulate_CLorenz_data(
                    n=n_nodes, 
                    N=n_features, 
                    total_length=total_length,
                    seed=current_seed
                )
                A0 = None
                if_valid = True  # CLorenz无需筛选
            else:
                print(f"不支持的系统类型: {system_type}")
                return 0, []
            
            if if_valid:
                valid_count += 1
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
                filename = f"{system_type}_oscillation_{valid_count:03d}_{timestamp.replace(':', '-')}.npz"
                filepath = os.path.join(output_dir, filename)
                
                # 保存数据
                save_dict = {
                    'outputs': outputs, 
                    'X_trim': X_trim, 
                    'fraction_reduce': outputs[:, 3]
                }
                if A0 is not None:
                    save_dict['A0'] = A0
                
                np.savez(filepath, **save_dict)
                
                # 生成数据图
                fig_dir = f"/home/yangna/JetBrains/graphRC/Figures/{system_type}_generated"
                os.makedirs(fig_dir, exist_ok=True)
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # 1. 时间序列图（前6个节点）
                for node in range(min(6, n_nodes)):
                    for feature in range(3):
                        axes[0, 0].plot(X_trim[:, node, feature], label=f'Node {node+1} Feature {feature+1}', alpha=0.7)
                axes[0, 0].set_title('Time Series (Second Feature)')
                axes[0, 0].set_xlabel('Time')
                axes[0, 0].set_ylabel('Value')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. mean_x和fraction_reduce
                axes[0, 1].plot(outputs[:, 0], label='mean_x', color='blue')
                ax2 = axes[0, 1].twinx()
                ax2.plot(outputs[:, 3], label='fraction_reduce', color='red', alpha=0.7)
                axes[0, 1].set_title('System State')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('mean_x')
                ax2.set_ylabel('fraction_reduce')
                
                # 3. x_eff和beta_eff
                axes[1, 0].plot(outputs[:, 1], label='x_eff', color='green')
                axes[1, 0].plot(outputs[:, 2], label='beta_eff', color='orange', alpha=0.7)
                axes[1, 0].set_title('Effective Interactions')
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Value')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # 4. 相图 (x_eff vs beta_eff)
                axes[1, 1].scatter(outputs[:, 2], outputs[:, 1], c=outputs[:, 3], cmap='viridis', s=5)
                axes[1, 1].set_xlabel('beta_eff')
                axes[1, 1].set_ylabel('x_eff')
                axes[1, 1].set_title('Phase Space (color=fraction)')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                figname = f"{system_type}_{valid_count:03d}_{timestamp.replace(':', '-')}.svg"
                figpath = os.path.join(fig_dir, figname)
                plt.savefig(figpath, dpi=150, bbox_inches='tight')
                plt.close()
                
                metadata = {
                    'id': valid_count,
                    'filename': filename,
                    'seed': current_seed,
                    'mean_x_range': (np.min(outputs[:, 0]), np.max(outputs[:, 0])),
                    'x_eff_range': (np.min(outputs[:, 1]), np.max(outputs[:, 1])),
                    'beta_eff_range': (np.min(outputs[:, 2]), np.max(outputs[:, 2]))
                }
                all_metadata.append(metadata)
                
                print(f"  ✓ 有效数据 {valid_count}: {filename}")
            else:
                print(f"  ✗ gLV数据无振荡，跳过")
                
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            continue
    
    # 保存元数据
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write("ID,Filename,Seed,MeanX_Min,MeanX_Max,XEff_Min,XEff_Max,BetaEff_Min,BetaEff_Max\n")
        for meta in all_metadata:
            f.write(f"{meta['id']},{meta['filename']},{meta['seed']},"
                   f"{meta['mean_x_range'][0]:.4f},{meta['mean_x_range'][1]:.4f},"
                   f"{meta['x_eff_range'][0]:.4f},{meta['x_eff_range'][1]:.4f},"
                   f"{meta['beta_eff_range'][0]:.4f},{meta['beta_eff_range'][1]:.4f}\n")
    
    print(f"\n生成完成！共生成 {valid_count} 条有效振荡数据")
    print(f"数据保存在: {output_dir}")
    print(f"元数据: {metadata_file}")
    
    return valid_count, all_metadata


if __name__ == "__main__":
    # 示例：生成gLV数据
    generate_and_filter_data(
        system_type='gLV',
        num_datasets=20,
        n_nodes=6,
        n_features=3,
        total_length=2000,
        output_dir="/home/yangna/JetBrains/graphRC/Data/gLV"
    )
