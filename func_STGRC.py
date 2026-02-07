import os
import json
import argparse
import time
import numpy as np
import math
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm
from scipy.special import logsumexp
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument('-GPUid', default='0', type=str, help="GPU used")
    parser.add_argument('-win_step', default=1, type=int, help="Window step")
    parser.add_argument('-num_win', default=-1, type=int, help="Number of windows")
    parser.add_argument('-NNkey', default='N', type=str,
                        choices=['N', 'RC', 'graphRC'],
                        help="Type of neural network")
    grouped_args = {}
    #### data mark
    parser.add_argument('-dataset', default='UU_16nodes', type=str, help="Dataset")
    parser.add_argument('-label', default='N', type=str, help="Sample label")
    # parser.add_argument('-ubase', default=None, type=str, help="Network path or tag")
    # parser.add_argument('-coupling', default=None, type=str, help="Coupling path or tag")
    parser.add_argument('-datashape', default=(0, 0, 0), type=tuple, help="Shape of data, features * nodes * timelength")
    parser.add_argument('-ns', default=0, type=float, help="Noise strength")
    parser.add_argument('-bif_point_para', default=-1, type=int, help="Bifurcation point based on parameter")
    parser.add_argument('-bif_point_obsvt', default=-1, type=int, help="Bifurcation point based on observation")
    parser.add_argument('-EWS_win', default=-1, type=int, help="Window index of EWS")
    grouped_args['data_group'] = ['dataset', 'label', 'datashape', 'ns', 'bif_point_para', 'bif_point_obsvt', 'EWS_win']
    #### stPCA
    stPCA_group = parser.add_argument_group('parameters related to stPCA')
    stPCA_group.add_argument('-ran_idx', default=0, type=int, help="Select some dimension for DEV analysis")
    stPCA_group.add_argument('-win_m', default=10, type=int, help="length of each window")
    stPCA_group.add_argument('-L', default=3, type=int, help="Embedding dimension")
    stPCA_group.add_argument('-lam', default=0.2, type=float, help="1-lambda in Manuscript")
    stPCA_group.add_argument('-target', default=(0, 0), type=tuple, help="Target variaable if prediction")
    grouped_args['stPCA_group'] = ['win_m', 'L', 'lam', 'ran_idx', 'target']
    #### Reservoir Computing Hyperparameters
    RC_group = parser.add_argument_group('parameters related to RC')
    RC_group.add_argument('-fun_act', default='tanh', type=str, help="Activation function")
    RC_group.add_argument('-res_nodes', default=800, type=int, help="Number of reservoir nodes")
    RC_group.add_argument('-deg', default=0.1, type=float, help="Average degree of reservoir")
    RC_group.add_argument('-aa', default=5, type=float, help="Scaler of reservoir")
    RC_group.add_argument('-alpha', default=1, type=float, help="Leakage factor of reservoir")
    RC_group.add_argument('-rho', default=1, type=float, help="Spectral radius of reservoir")
    RC_group.add_argument('-warmup_steps', default=0, type=int, help="Warmup steps of reservoir")
    grouped_args['RC_group'] = ['fun_act', 'res_nodes', 'deg', 'aa', 'alpha', 'rho', 'warmup_steps']

    args, unknown = parser.parse_known_args()
    return vars(args), grouped_args

"""根据分组信息组织参数"""
def group_args(args, groups):
    grouped_dict = {}
    # 未分组参数
    grouped_args_set = {arg for arg_names in groups.values() for arg in arg_names}
    ungrouped_args = {arg: value for arg, value in args.items() if arg not in grouped_args_set}
    grouped_dict.update(ungrouped_args) # 将未分组参数添加到根级别
    # 分组参数
    for group_name, arg_names in groups.items():
        grouped_dict[group_name] = {arg: args[arg] for arg in arg_names}
    return grouped_dict


def stPCA(traindata: torch.Tensor,  **kwargs):
    device = torch.device("cuda:" + kwargs['GPUid'] if torch.cuda.is_available() else "cpu")
    traindata = traindata.to(dtype=torch.float32)
    lam = kwargs.get("lam", 0.2) ### 1-lambda in Manuscript
    L = kwargs.get("L", 3)
    [n, m] = traindata.shape
    X = traindata    ## n*m, input_dimensions*trainlength
    P=X[:, 1:]     ## n*(m-1)
    Q=X[:,:-1]     ## n*(m-1)
    b=1-lam          ## lambda in Manuscript
    ##### Z
    # Use CPU for large matrix operations to avoid GPU memory issues
    if n * L > 5000:  # Threshold for using CPU
        device_cpu = torch.device("cpu")
        H = torch.zeros((n*L, n*L)).to(device_cpu)
        XX = torch.matmul(X.cpu(), X.cpu().T)
        PP = torch.matmul(P.cpu(), P.cpu().T)
        PQ = torch.matmul(P.cpu(), Q.cpu().T)
        QP = torch.matmul(Q.cpu(), P.cpu().T)
        QQ = torch.matmul(Q.cpu(), Q.cpu().T)
    else:
        H = torch.zeros((n*L, n*L)).to(device)
        XX = torch.matmul(X, X.T)
        PP = torch.matmul(P, P.T)
        PQ = torch.matmul(P, Q.T)
        QP = torch.matmul(Q, P.T)
        QQ = torch.matmul(Q, Q.T)
    
    H[:n, :n] = lam*XX-b*PP
    H[:n, n:2*n] = b*PQ
    for j in range(2, L, 1):
        H[n*(j-1):n*j, n*(j-1)-n:n*j-n] = b*QP
        H[n*(j-1):n*j, n*(j-1):n*j] = lam*XX-b*PP-b*QQ
        H[n*(j-1):n*j, n*(j-1)+n:n*j+n] = b*PQ
    H[n*(L-1):n*L, n*(L-2):n*(L-1)] = b*QP
    H[n*(L-1):n*L, n*(L-1):n*L] = lam*XX-b*QQ

    # Compute eigenvalues on CPU if matrix is large
    if n * L > 5000:
        D, V = torch.linalg.eig(H)
        D = D.to(device)
        V = V.to(device)
    else:
        D, V = torch.linalg.eig(H)
    ao = torch.real(D)
    aa, eigvIdx = torch.sort(ao, descending=True)  ## 特征值由大到小排序
    V = V[:, eigvIdx]
    if aa[0] > 0: # and abs(aa[0])>=abs(aa[-1]):
        cW = V[:, 0]
    else:
        cW = V[:, -1]
    W = cW.view(L, n)    ## reshape的顺序是否重要?
    W = torch.real(W).float()
    Z = (max(abs(aa))*torch.matmul(W, X)).T   # Z.T in Manuscript ## X: n*m, W: n*L, Z: m*L
    ## Flat Z
    Z = torch.real(Z)
    [z_rows, z_cols] = Z.shape  ## m, L
    flat_z = []  # Z矩阵到z(1,...,m)的转换
    for zi in range(z_rows):
        tmp = []
        for zj in range(z_cols):
            if zi - zj < 0:
                break
            tmp.append(Z[zi - zj, zj])
        tmp = torch.tensor(tmp).flatten()
        flat_z.append(torch.mean(tmp))
    flat_z = torch.tensor(flat_z).flatten()

    flat_z_pred = []
    for zj in range(1, z_cols, 1):
        num = z_cols - zj + 1
        tmp = []
        for ni in range(num):
            zi = z_rows - ni - 1
            tmp.append(Z[zi, zj + ni - 1])
        tmp = torch.tensor(tmp).flatten()
        flat_z_pred.append(torch.mean(tmp))
    flat_z_pred = torch.tensor(flat_z_pred).flatten()
    # flat_z_total = torch.cat((flat_z, flat_z_pred))
    var_y = torch.norm(Z, p='fro')
    return {'Z': Z, 'eigvalue': max(abs(aa)),
            'var_y': var_y, 'flat_z': flat_z}

##########defining defining W_in, W_r, W_b of RC##########
def RC_ge(**arg):
    a = arg['aa']
    n = int(arg['res_nodes'])
    # (features, nodes, timelength) = udata.shape
    #######defining W_in and W_b
    dim = int(arg['datashape'][0])
    W_in = np.zeros((n, dim))
    n_win = n - n % dim
    index = np.random.permutation(range(n))
    index = index[:n_win]
    index = np.reshape(index, [int(n_win / dim), dim])
    for d in range(dim):
        W_in[index[:, d], d] = a * (2 * np.random.rand(int(n_win / dim)) - 1)
    W_b = a * (2 * np.random.rand(n) - 1)
    sample = random.randint(1, 999)
    W_r = np.loadtxt("/home/yangna/JetBrains/Data/Wr_groups/k=%.2f/%in%i/Wr_%in%i_a5_%i.txt"
                     % (arg['deg'], int(arg['deg'] * n), n,
                        int(arg['deg'] * n), n, sample),
                     delimiter=",")
    W_r = arg['rho'] * W_r
    return {'W_in': W_in, 'W_b': W_b, 'W_r': W_r}

def afunc(x, **kwargs):
    if kwargs.get("fun_act") == 'ReLu':
        return torch.where(x < 0, 0, x)
    elif kwargs.get("fun_act") == 'softplus':
        return torch.log(1 + torch.exp(x))
    elif kwargs.get("fun_act") == 'ELU':
        return torch.where(x > 0, x, (torch.exp(x) - 1))
    elif kwargs.get("fun_act") == 'tanh':
        # return torch.tanh(x)
        return torch.tanh(x)
    else:
        return x

# 计算储备池状态（输入含 warm up data，输出时舍去）, 储备池输入：udata
def Res_evo_RC(udata: torch.Tensor, W_in, W_r,  r0=False, *W_b, **kwargs):
    device = torch.device("cuda:" + kwargs['GPUid'] if torch.cuda.is_available() else "cpu")
    alpha = kwargs.get("alpha")
    n = kwargs.get("res_nodes")
    len_washout = kwargs.get("warmup_steps")
    r_tmp = torch.zeros((n, udata.shape[1]+1)).to(device)
    if type(r0) == torch.Tensor:
        r_tmp[:, 0] = r0
    for ti in range(udata.shape[1]):
        x1 = torch.matmul(W_r, r_tmp[:, ti])
        x2 = torch.matmul(W_in, udata[:, ti])
        if type(W_b) == torch.Tensor:
            r_tmp[:, ti+1] = (1 - alpha) * r_tmp[:, ti] + alpha * afunc(x1+x2+W_b, **kwargs)
        else:
            r_tmp[:, ti + 1] = (1 - alpha) * r_tmp[:, ti] + alpha * afunc(x1+x2, **kwargs)
    r_all = r_tmp[:, 1+len_washout:]
    return r_all


# 计算图储备池状态（输入含 warm up data，输出时舍去）, 储备池输入：udata
def Res_evo_graphRC(udata: torch.Tensor, W_in, W_r, norm_Adj, r0=False, cp=False, *W_b, **kwargs):
    device = torch.device("cuda:" + kwargs['GPUid'] if torch.cuda.is_available() else "cpu")
    if type(norm_Adj) == np.ndarray:
        norm_Adj = torch.from_numpy(norm_Adj).float().to(device)
    alpha = kwargs.get("alpha")
    n = kwargs.get("res_nodes")
    len_washout = kwargs.get("warmup_steps")
    (features, nodes, timelength) = udata.shape
    h_tmp = torch.zeros((n, nodes, timelength+1)).to(device)
    if type(r0) == torch.Tensor:
        for ni in range(nodes):
            h_tmp[:, ni, 0] = r0
    for ti in range(timelength):
        if type(cp) == torch.Tensor:
            x1 = W_r @ h_tmp[:, :, ti] @ (cp[ti]*norm_Adj)
        else:
            x1 = W_r @ h_tmp[:, :, ti] @ norm_Adj
        x2 = torch.matmul(W_in, udata[:, :, ti])
        if type(W_b) == torch.Tensor:
            h_tmp[:, :, ti+1] = (1-alpha) * h_tmp[:, :, ti]+alpha * afunc(x1+x2+W_b, **kwargs)
        else:
            h_tmp[:, :, ti + 1] = (1-alpha) * h_tmp[:, :, ti]+alpha * afunc(x1+x2, **kwargs)
    r_all = h_tmp[:, :, 1+len_washout:]
    return r_all


def graphL(udata: torch.Tensor):
    [f, nodes, m] = udata.shape
    udata_reshaped = udata.permute(1, 0, 2)  # (nodes, f, m)
    diff = udata_reshaped.unsqueeze(1) - udata_reshaped.unsqueeze(0)  # (nodes, nodes, f, m)
    # 计算 Frobenius 范数（即对 (f, m) 两维平方后求和开根号）
    distance_matrix = torch.norm(diff, dim=(2, 3))  # (nodes, nodes)

    # 构建邻接矩阵，初始为全 0
    adj = torch.zeros_like(distance_matrix)
    # 动态设置 k，确保不超过节点数量
    k = min(5, nodes - 1)  # 至少留一个节点给自己
    for i in range(nodes):
        # 设置自身距离为无穷大，避免选到自己
        distance_matrix[i, i] = float('inf')
        # 获取最近的 k 个邻居索引
        knn_idx = torch.topk(distance_matrix[i], k=k, largest=False).indices
        # 设置邻接关系为 1
        adj[i, knn_idx] = 1.0
    # 可选：构造无向图（对称化）
    adj = torch.maximum(adj, adj.T)
    adj = adj.float()
    # 加单位矩阵（加自环）
    adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
    # 计算度矩阵 D
    degree = adj_hat.sum(dim=1)  # (nodes,)
    # 计算 D^{-1/2}
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0  # 防止除零
    # 构造归一化邻接矩阵
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    norm_Adj = D_inv_sqrt @ adj_hat @ D_inv_sqrt
    return norm_Adj


def Mapping(data_pack: dict, **arg):
    device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
    data = torch.from_numpy(data_pack["data_noise"]).float().to(device)
    ubase = torch.from_numpy(data_pack["ubase"]).float().to(device) if data_pack["ubase"] is not None else None
    coupling = torch.from_numpy(data_pack["coupling"]).float().to(device)
    if data.ndim != 2 and data.ndim != 3:
        print("Wrong data shape!")
        exit()
    else:
        Ws = RC_ge(**arg)
        W_in = torch.from_numpy(Ws["W_in"]).float().to(device)
        W_r = torch.from_numpy(Ws["W_r"]).float().to(device)
        W_b = torch.from_numpy(Ws["W_b"]).float().to(device)
        # ensure idx does not exceed available coupling length
        if data.shape[-1] != len(coupling):
            print("Warning: data length does not match coupling length.")
            exit()
        idx = np.arange(data.shape[-1])
        np.random.shuffle(idx)
        if data.ndim==2:
            warmup_data = data[:, idx[0:arg['warmup_steps']]]
            input_data = Res_evo_RC(torch.cat((warmup_data, data), dim=1), W_in=W_in, W_b=W_b, W_r=W_r, **arg)
        else:
            if ubase is None:
                adj = graphL(data)
            else:
                # adj_hat = ubase
                I = torch.eye(len(ubase), device=device)
                adj_hat = ubase + I
                adj = adj_hat / (adj_hat.sum(dim=1, keepdim=True) + 1e-6)  #行归一
            warmup_data = data[:, :, idx[0:arg['warmup_steps']]]
            # cp = torch.from_numpy(coupling).float().to(device)
            # ensure warmup_steps not greater than available indices
            arg['warmup_steps'] = min(arg['warmup_steps'], idx.shape[0])
            # Debugging: inspect idx slice before using it to index GPU tensor
            idx_w = idx[0:arg['warmup_steps']]
            cp = torch.cat((coupling[idx_w], coupling))
            cat_data = torch.cat((warmup_data, data), dim=2)

            normed_data = (cat_data-cat_data.mean(dim=-1, keepdim=True))/(cat_data.std(dim=-1, keepdim=True) + 1e-6)
            # normed_data = torch.cat((warmup_data, data), dim=2)/(torch.cat((warmup_data, data), dim=2).std()+1e-6)
            input_data = Res_evo_graphRC(normed_data, norm_Adj=adj, cp=cp,
                                         W_in=W_in, W_b=W_b, W_r=W_r, **arg)
        # print(normed_data.shape, input_data.shape, cat_data.shape, data.shape)
        print("Input:", torch.min(normed_data), torch.max(normed_data))
        print("After GC:", torch.min(input_data), torch.max(input_data))
        print("U_base stats:", adj.mean(), adj.std())
        print("Activation saturates?", torch.sum(normed_data > 5).item())

        return input_data

def arg_save(path, arg, groups):
    grouped_arg = group_args(arg, groups)
    # find_bad(grouped_arg)
    # Convert any remaining non-serializable objects to strings or lists
    def json_serializable(obj):
        if isinstance(obj, (list, tuple, dict, str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    with open(os.path.join(path, 'arguments.json'), 'w', encoding='utf-8') as f:
        json.dump(grouped_arg, f,
                  indent=4,
                  ensure_ascii=False,  # 保证非ASCII字符正常保存
                  separators=(',', ': '),
                  default=json_serializable)
    return

class Timer:
    def __init__(self, print_tmpl="Time elapsed: {:.2f} seconds"):
        self.print_tmpl = print_tmpl

    def __enter__(self):
        self.start_time = time.time()  # 记录开始时间
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time  # 计算耗时
        print(self.print_tmpl.format(elapsed_time))  # 打印耗时信息


def stgRC(data_pack, path, groups, **arg):
    step = arg['win_step']
    device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
    xx_noise = torch.from_numpy(data_pack["data_noise"]).float().to(device)
    # ubase = torch.from_numpy(data_pack["ubase"]).float().to(device) if ubase is not None else None
    # coupling = torch.from_numpy(data_pack["coupling"]).float().to(device)
    # ##################### data batching ########################
    if type(xx_noise) == np.ndarray:
        xx_noise = torch.from_numpy(xx_noise).float().to(device)
    if xx_noise.ndim != 3:
        print('Dimension mismatched!')
        exit()
    win_m = arg['win_m']
    [f, nodes, m] = xx_noise.shape # features, nodes, total_length
    arg['datashape'] = (f, nodes, m)
    if arg["num_win"] <= 0:
        num_zones = int((m - win_m) / step)  # 180
        arg["num_win"] = num_zones
    else:
        num_zones = arg["num_win"]
    myzones = []
    for i in range(num_zones):
        myzones.append(range(0 + i * step, 0 + i * step + win_m, 1))
    # ########################## algorithm start ###############################
    xx_input = Mapping(data_pack, **arg)
    batches = [xx_input[:, :, i * step : i * step + win_m].reshape(-1, win_m)
    for i in range(num_zones)
    ]
    print("Mapping done. ")
    # arg_save(path, arg, groups)
    print("Arguments: " + str(arg))
    print("-" * 80)

    def parallel(input_data):
        traindata = input_data - torch.mean(input_data, dim=1, keepdim=True)
        stPCA_results = stPCA(traindata, **arg)
        flat_z=stPCA_results.get('flat_z')
        temp_var=stPCA_results.get('var_y')
        return [flat_z, temp_var]

    with Timer(print_tmpl='Pool() takes {:.1f} seconds'):
        with ThreadPoolExecutor(6) as executor:
            results_pool = list(
                tqdm(executor.map(
                    parallel, batches),
                    total=num_zones, desc="Processing tasks"
                )
            )
            flatz_pool, var_y_pool = zip(*results_pool)
            var_y_pool = torch.tensor(var_y_pool)
            flatz_pool = torch.stack(flatz_pool)
    return {"sd(Z)": var_y_pool, "flat_z": flatz_pool}, arg


def stRC(data_pack, path, groups, **arg):
    step = arg['win_step']
    device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
    xx_noise = torch.from_numpy(data_pack["data_noise"]).float().to(device)
    # ubase = torch.from_numpy(data_pack["ubase"]).float().to(device) if ubase is not None else None
    # coupling = torch.from_numpy(data_pack["coupling"]).float().to(device)
    # ##################### data batching ########################
    if type(xx_noise) == np.ndarray:
        xx_noise = torch.from_numpy(xx_noise).float().to(device)
    if xx_noise.ndim != 2:
        print('Dimension mismatched, so former dimensions flattened.')
    win_m = arg['win_m']
    [n, m] = xx_noise.shape  # input_dimensions, total_length
    arg['datashape'] = (n, m)
    if arg["num_win"] <= 0:
        num_zones = int((m - win_m) / step)  # 180
        arg["num_win"] = num_zones
    else:
        num_zones = arg["num_win"]
    myzones = []
    for i in range(num_zones):
        myzones.append(range(0 + i * step, 0 + i * step + win_m, 1))
    batches_ori = [xx_noise[:, 0 + i * step: 0 + i * step + win_m] for i in range(num_zones)]
    # ########################## algorithm start ###############################
    xx_input = Mapping(data_pack, **arg)
    batches = [xx_input[:, 0 + i * step: 0 + i * step + win_m] for i in range(num_zones)]
    print("Mapping done.")
    arg_save(path, arg, groups)
    print("Arguments: " + str(arg))
    print("-" * 80)

    def parallel(batch, batch_ori):
        input_data = batch
        traindata = input_data - torch.mean(input_data, dim=1, keepdim=True)
        stRC_results = stPCA(traindata, ori_data=batch_ori,
                              **arg)  # Z.T in Manuscript ## X: n*m, W: n*L, Z: m*L
        temp_var_y = stRC_results.get('var_y')
        flat_z = stRC_results.get('flat_z')
        return [flat_z, temp_var_y]

    with Timer(print_tmpl='Pool() takes {:.1f} seconds'):
        with ThreadPoolExecutor(6) as executor:
            results_pool = list(
                tqdm(executor.map(
                    parallel, batches, batches_ori),
                    total=num_zones, desc="Processing tasks"
                )
            )
            flatz_pool, var_y_pool= zip(*results_pool)
            var_y_pool = torch.tensor(var_y_pool)
            flatz_pool = torch.stack(flatz_pool)
    return {"sd(Z)": var_y_pool, "flat_z": flatz_pool}, arg

def STI_pre(train_x, train_y, **arg):
    device = arg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    pred_len = arg["L"]
    # ---------- 输入转为 torch.Tensor 并放到 device ----------
    if isinstance(train_x, np.ndarray):
        train_x = torch.from_numpy(train_x)
    if isinstance(train_y, np.ndarray):
        train_y = torch.from_numpy(train_y)

    train_x = train_x.to(device=device, dtype=torch.float32)
    train_y = train_y.to(device=device, dtype=torch.float32)

    n, train_len = train_x.shape
    l = max(5, int(n / 100))  # randomly selected variables of matrix B
    w_flag = torch.zeros(n, device=device)                      # (n,)
    B = torch.zeros((n, pred_len), device=device)               # (n, pred_len)
    y_pred = torch.zeros(pred_len - 1, device=device)           # (pred_len-1,)
    errors = []
    # ============ 迭代开始 ============
    for iter in range(1000):
        # ----------------- 更新 B -----------------
        random_idx = torch.randint(low=0, high=n, size=(l,), device=device)
        # G_B: (train_len - pred_len, l)
        G_B = train_x[random_idx, : train_len - pred_len].T

        # 构造 Ym_known: 形状 (train_len - pred_len, pred_len)
        Ym_known_list = []
        for j in range(train_len - pred_len):
            Ym_known_list.append(train_y[j : j + pred_len])
        Ym_known = torch.stack(Ym_known_list, dim=0)  # (train_len - pred_len, pred_len)

        # 预先 pinv 一次
        # Ym_pinv = torch.linalg.pinv(Ym_known)  # (pred_len, train_len - pred_len)

        for i in range(l):
            # B_para = pinv(Ym_known) @ G_B[:, i]
            # Ym_pinv = torch.linalg.pinv(Ym_known)  # (pred_len, train_len - pred_len)
            # B_para = Ym_pinv @ G_B[:, i]  # (pred_len,)
            B_para = torch.linalg.lstsq(Ym_known, G_B[:, i]).solution
            idx = random_idx[i]
            B[idx, :] = (B[idx, :] + B_para + B_para * (1.0 - w_flag[idx])) * 0.5
            w_flag[idx] = 1.0

        # ------------- 基于 B 的临时预测 -------------
        # super_bb: (n*(pred_len-1),)
        # super_AA: (n*(pred_len-1), pred_len-1)
        super_bb = torch.zeros(n * (pred_len - 1), device=device)
        super_AA = torch.zeros(
            n * (pred_len - 1), pred_len - 1, device=device
        )

        for i in range(n):
            kt = 0
            # 对每个 i，构造 (pred_len - 1) 个方程
            for j in range(train_len - (pred_len - 1), train_len):
                # j 从 train_len - (pred_len-1) 到 train_len-1，共 pred_len-1 个
                bb_ij = train_x[i, j].clone()
                col_known_y_num = train_len - j  # 1,2,...,pred_len-1

                # 减去已知 y 部分
                for r in range(col_known_y_num):
                    bb_ij = bb_ij - B[i, r] * train_y[train_len - col_known_y_num + r]

                row_idx = i * (pred_len - 1) + kt
                super_bb[row_idx] = bb_ij
                # AA[kt, : pred_len - col_known_y_num] = B[i, col_known_y_num : pred_len]
                super_AA[row_idx, : pred_len - col_known_y_num] = B[
                    i, col_known_y_num:pred_len
                ]
                kt += 1

        # super_AA: (N_all, pred_len-1), super_bb: (N_all,)
        # pred_y_tmp: (pred_len-1,)
        pred_y_tmp = torch.linalg.lstsq(super_AA, super_bb).solution
        # pred_y_tmp = torch.linalg.pinv(super_AA) @ super_bb

        # ------------- 更新 A, Y 等 -------------
        tmp_y = torch.cat((train_y, pred_y_tmp), dim=0)  # (train_len + pred_len -1,)

        # Ym: (pred_len, train_len)
        Ym_list = []
        for j in range(pred_len):
            Ym_list.append(tmp_y[j : j + train_len])
        Ym = torch.stack(Ym_list, dim=0)

        BG = torch.cat((B, train_x), dim=1)  # (n, pred_len + train_len)
        IY = torch.cat(
            (torch.eye(pred_len, device=device), Ym), dim=1
        )  # (pred_len, pred_len + train_len)
        A = torch.linalg.lstsq(BG.T, IY.T).solution.T # (pred_len, n)
        # A = IY @ torch.linalg.pinv(BG)  # (pred_len, n)

        # final_ym = A @ train_x  ⇒ (pred_len, train_len)
        final_ym = A @ train_x

        tmp_pred = torch.zeros(pred_len - 1, device=device)
        for j1 in range(pred_len - 1):
            tmp_vals = []
            for row in range(j1 + 1, pred_len):
                col = train_len - row + j1
                tmp_vals.append(final_ym[row, col])
            tmp_pred[j1] = torch.mean(torch.stack(tmp_vals))

        # ------------- 收敛判断 -------------
        error = torch.sqrt(torch.mean((tmp_pred - y_pred) ** 2))
        errors.append(error.item())

        if error < 0.02:
            copy_y = torch.zeros(train_len, device=device)
            # 前 pred_len 段
            for j2 in range(pred_len):
                tmp_vals = []
                for row in range(j2 + 1):
                    col = j2 - row
                    tmp_vals.append(final_ym[row, col])
                copy_y[j2] = torch.mean(torch.stack(tmp_vals))

            # 后面部分
            for j3 in range(pred_len, train_len):
                tmp_sum = 0.0
                for row in range(pred_len):
                    col = j3 - row
                    tmp_sum = tmp_sum + final_ym[row, col]
                copy_y[j3] = tmp_sum / pred_len

            rmse_full = torch.sqrt(torch.mean((copy_y - train_y) ** 2))
            # 可以视需要保存 rmse_full
            break

        y_pred = tmp_pred
    errors = np.array(errors, dtype=np.float32)
    return {
        "y_pred": y_pred,
        "errors": errors,
        "copy_y": copy_y if "copy_y" in locals() else None,
        "B": B,
    }

def stgRC_pre(data_pack, path, groups, **arg):
    step = arg['win_step']
    device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
    xx_noise = torch.from_numpy(data_pack["data_noise"]).float().to(device)
    # ubase = torch.from_numpy(data_pack["ubase"]).float().to(device) if ubase is not None else None
    # coupling = torch.from_numpy(data_pack["coupling"]).float().to(device)
    # ##################### data batching ########################

    if type(xx_noise) == np.ndarray:
        xx_noise = torch.from_numpy(xx_noise).float().to(device)
    if xx_noise.ndim != 3:
        print('Dimension mismatched!')
        exit()
    win_m = arg['win_m']
    [f, nodes, m] = xx_noise.shape # features, nodes, total_length
    arg['datashape'] = (f, nodes, m)
    if arg["num_win"] <= 0:
        num_zones = int((m - win_m) / step)  # 180
        arg["num_win"] = num_zones
    else:
        num_zones = arg["num_win"]
    # myzones = []
    # for i in range(num_zones):
    #     myzones.append(range(0 + i * step, 0 + i * step + win_m, 1))
    # ########################## algorithm start ###############################
    if arg['NNkey'] == 'N':
        xx_input = xx_noise
    else:
        xx_input = Mapping(data_pack, **arg) #res_nodes*nodes*total_length
    batches_x = [xx_input[:, :, i * step : i * step + win_m].reshape(-1, win_m) #((res_nodes*nodes)*win_m)*num_win
    for i in range(num_zones)
    ]
    tar0, tar1 = arg['target']
    batches_y = [xx_noise[tar0, tar1, i * step: i * step + win_m] for i in range(num_zones)] # win_m*num_win
    print("Mapping done. ")
    arg_save(path, arg, groups)
    # print("Arguments: " + str(arg))
    print("-" * 80)

    def parallel(input_data, train_y):
        # f, nodes, m = input_data.shape
        traindata = input_data - torch.mean(input_data, dim=0, keepdim=True)
        pre_results = STI_pre(train_x=traindata, train_y=train_y, **arg) # {"y_pred": y_pred, "errors": errors, "copy_y": copy_y, "B": B}
        temp_y_pred=pre_results.get('y_pred')
        temp_copy_y=pre_results.get('copy_y')
        return [temp_y_pred, temp_copy_y]

    with Timer(print_tmpl='Pool() takes {:.1f} seconds'):
        with ThreadPoolExecutor(6) as executor:
            results_pool = list(
                tqdm(executor.map(
                    parallel, batches_x, batches_y),
                    total=num_zones, desc="Processing tasks"
                )
            )
            y_pred_pool, copy_y_pool = zip(*results_pool)
            copy_y_pool = [x for x in copy_y_pool if x is not None]
            copy_y_pool = torch.stack(copy_y_pool, dim=0)
            y_pred_pool = torch.stack(y_pred_pool)
    return {"y_preds": y_pred_pool, "copy_ys": copy_y_pool}, arg

def Save_mainresluts(main_results, path, **arg):
    sdZ = main_results["sd(Z)"]
    flatZs = main_results["flat_z"]
    win_id = np.arange(1, arg["num_win"] + 1, 1)
    # ################################      保存结果        ####################################
    summary = pd.DataFrame({"win_id": win_id,
                            "sd(Z)": sdZ.cpu().numpy()}
                           )
    summary.to_csv(path + "/summary_%s.txt" % arg["label"], index=False)

    # 保存 flatZs 为 TXT文件，添加多行注释
    comments = """# flatz - 数组\n# 行向量为窗口的flatz\n# 数据开始："""
    np.save(path + "/flatZs_%s.npy" % arg["label"], flatZs.cpu().numpy())
               # header=comments, comments='',  # 禁用默认的注释符
               # fmt='%.6f')  # 保留6位小数
    return

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T = len(data)
    log_R = -np.inf * np.ones((T + 1, T + 1))
    log_R[0, 0] = 0  # log 0 == 1
    pmean = np.empty(T)  # Model's predictive mean.
    pvar = np.empty(T)  # Model's predictive variance.
    log_message = np.array([0])  # log 0 == 1
    log_H = np.log(hazard)
    log_1mH = np.log(1 - hazard)

    for t in range(1, T + 1):
        # 2. Observe new datum.
        x = data[t - 1]

        # Make model predictions.
        pmean[t - 1] = np.sum(np.exp(log_R[t - 1, :t]) * model.mean_params[:t])
        pvar[t - 1] = np.sum(np.exp(log_R[t - 1, :t]) * model.var_params[:t])

        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t + 1] = new_log_joint
        log_R[t, :t + 1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar


class GaussianUnknownMean:

    def __init__(self, mean0, var0, varx):
        """Initialize model.

        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params = (self.mean_params * self.prec_params[:-1] +
                           (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx

# -----------------------------------------------------------------------------
def plot_posterior(T, data, cps, R, pmean, pvar, path, label='N'):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    ax1, ax2 = axes
    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r',
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)
    if type(cps) == int or np.int64 or np.int16 or np.int32:
        ax1.axvline(cps, c='red', ls='dotted')
        ax2.axvline(cps, c='red', ls='dotted')
    else:
        for cp in cps:
            ax1.axvline(cp, c='red', ls='dotted')
            ax2.axvline(cp, c='red', ls='dotted')
    plt.savefig(path + '/bocd_%s.svg' % label)
    plt.close()

# -----------------------------------------------------------------------------
def EWS_bocd(sdZ, path='', bif_window=-1, p_bocd = 1, step=10, if_plot=False, label='N'):
    hazard = 1 / 1000  # Constant prior on changepoint probability.
    T = len(sdZ) # Number of observations.
    mean0 = np.mean(sdZ[:T])  # The prior mean on the mean parameter.
    var0 = np.std(sdZ[:T])  # The prior variance for mean parameter.
    varx = np.std(sdZ[:T])  # The known variance of the data.
    model = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(sdZ[:T], model, hazard)
    maxp_idx = np.array([np.argmax(R[i, :]) for i in range(T)])
    for i in range(5, T, 1):
        if maxp_idx[i] != i and i - maxp_idx[i]+1 > 5:
            bif_window = i - maxp_idx[i]+1
            p_bocd = R[i, maxp_idx[i]]
            break
    return bif_window, p_bocd

def All_cps_detection(seq, path='', bif_window=-1, p_bocd = 1, step=10, if_plot=False, label='N'):
    hazard = 1 / 1000  # Constant prior on changepoint probability.
    T = len(seq) # Number of observations.
    mean0 = np.mean(seq[:T])  # The prior mean on the mean parameter.
    var0 = np.std(seq[:T])  # The prior variance for mean parameter.
    varx = np.std(seq[:T])  # The known variance of the data.
    model = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(seq[:T], model, hazard)
    maxp_idx = np.array([np.argmax(R[i, :]) for i in range(T)])
    bif_windows = []
    p_bocds = []    
    for i in range(5, T, 1):
        if maxp_idx[i] != i and i - maxp_idx[i]+1 > 5:
            bif_window = i - maxp_idx[i]+1
            p_bocd = R[i, maxp_idx[i]]
            bif_windows.append(bif_window)
            p_bocds.append(p_bocd)
    if if_plot == True:
        plot_posterior(T, seq[:T], bif_windows, R, pmean, pvar, path, label)
    return bif_windows, p_bocds


def plot_perf(results, data_show, ews_point, ews_window,
              data_xlims, win_xlims, path, color_map=[],
              subplot=None, if_axinsert=True,
              if_invert=[False, False, False, False], legend=True, **arg):
    if subplot is None:
        subplot = [True, True, True, True]
    if len(color_map) < 5:
        color_map = ['#DEF056', '#56A2F0', '#F06156', 'red', 0.2]
        # color_map = ['#f4a116', '#2c92d1', '#e60012'] #[earlywarning, para_bif, ob_bif]


    sdZ = results["sd(Z)"]
    bif_point_para = arg['bif_point_para']
    if bif_point_para > 0 and bif_point_para <= arg['win_m']+arg['win_step']*(arg['num_win']-1):
        bif_window_para = int(np.ceil((bif_point_para-arg['win_m'])/arg['win_step']+1))
    else:
        bif_window_para = -1
    bif_point_obsvt = arg['bif_point_obsvt']

    # #绘制数据变化
    if subplot[0] and arg['dataset'] != "Peter_lake":
        # 绘制 data, zorder控制元素位置，值越大元素越靠前
        fig, ax = plt.subplots()
        if type(data_show) == pd.DataFrame:
            colors=['#2372a9', '#2ca02c', '#9467bd', '#e377c2', '#17becf'] #blue, green, purple, pink, cran
            for i, col in enumerate(data_show.columns):
                color = colors[(i % len(colors))]  # 防止超出 colormap 范围自动循环
                ax.plot(data_show.index, data_show[col], label=col, color=color)
        else:
            ax.plot(data_xlims, data_show, linewidth=2)

        if bif_point_obsvt > 0:
            ax.axvspan(bif_point_obsvt, data_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(x=bif_point_obsvt, color=color_map[2], linestyle='--', linewidth=1.5, label="observed bifurcation")
        if bif_point_para > 0:
            ax.axvline(x=bif_point_para, color=color_map[1], linestyle='--', linewidth=0.75, label="parameter bifurcation")
            if bif_point_obsvt < 0:
                ax.axvspan(bif_point_para, data_xlims[-1], color=color_map[3], alpha=color_map[4])
        if ews_window > 0:
            if bif_point_para < 0 and bif_point_obsvt < 0:
                ax.axvspan(ews_point, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(x=ews_point, color=color_map[0], linestyle='--', linewidth=1.5, label="early warning signal")

        ax.set_title(arg['dataset'])
        if if_invert[0]:
            ax.invert_xaxis()
        if legend:
            ax.legend()
        plt.savefig(path + '/%s data.svg' % arg['dataset'])
        plt.close()

    # 绘制 sd(Z)
    if subplot[1]:
        fig, ax = plt.subplots()
        ax.plot(win_xlims, sdZ, linewidth=2, c="#2372a9")
        if bif_point_obsvt > 0:
            ax.axvspan(bif_point_obsvt, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(x=bif_point_obsvt, color=color_map[2], linestyle='--', linewidth=1.5, label="observed bifurcation")
        if bif_point_para > 0:
            ax.axvline(x=bif_point_para, color=color_map[1], linestyle='--', linewidth=0.75, label="parameter bifurcation")
            if bif_point_obsvt < 0:
                ax.axvspan(bif_point_para, win_xlims[-1], color=color_map[3], alpha=color_map[4])
        if ews_window > 0:
            if bif_point_para < 0 and bif_point_obsvt < 0:
                ax.axvspan(ews_point, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            # 获取当前 y 轴范围
            y_min, y_max = ax.get_ylim()  # 可能是 [-0.1, 2.1] 等自动扩展范围
            # 计算归一化的 ymax
            ymax_norm = float((sdZ[ews_window] - y_min) / (y_max - y_min))
            # 绘制垂直线
            ax.axvline(x=ews_point, color=color_map[0], linestyle='--', linewidth=1.5,
                       ymin=0,  # ymin=0 表示从 x 轴开始
                       ymax=ymax_norm,  # 归一化到当前 y 轴范围
                       label="early warning signal")
            ax.scatter(ews_point, sdZ[ews_window], color=color_map[0], s=150, zorder=5, marker="*")  # s 控制点大小
        # ax.set_xlim(left=0)
        if legend:
            ax.legend()
        ax.set_title('sd(Z)')
        if if_invert[1]:
            ax.invert_xaxis()
        if ews_window > 0 and if_axinsert==True:
            # # 定义要放大的区域（x范围）
            if bif_window_para>0 and bif_window_para >= ews_window:
                zoom_win1, zoom_win2 = ews_window - 20, bif_window_para + 5
            elif bif_window_para>0 and bif_window_para <= ews_window:
                zoom_win1, zoom_win2 = bif_window_para - 20, ews_window + 5
            else:
                zoom_win1, zoom_win2 = ews_window - 20, ews_window + 5
            # # 创建插入的子图（放大图）
            ax_inset = inset_axes(ax, width="40%", height="30%", loc='center left')
            ax_inset.plot(win_xlims[zoom_win1:zoom_win2 + 1], sdZ[zoom_win1:zoom_win2 + 1], c="#2372a9")
            if bif_point_para>0:
                ax_inset.axvline(x=bif_point_para, color=color_map[1], linestyle='--', linewidth=1.5, label="parameter bifurcation")
            y_min, y_max = ax_inset.get_ylim()
            ymax_norm = float((sdZ[ews_window] - y_min) / (y_max - y_min))
            ax_inset.axvline(x=ews_point, color=color_map[0], linestyle='--', linewidth=1.5,
                       ymin=0,  # ymin=0 表示从 x 轴开始
                       ymax=ymax_norm)  # 归一化到当前 y 轴范围
            ax_inset.scatter(ews_point, sdZ[ews_window], color=color_map[0], s=80, zorder=5, marker="*")  # s 控制点大小
            # ax_inset.set_yticks([0, np.ceil(max(sdZ[zoom_win1:zoom_win2 + 1]))])
            ax_inset.set_yticks([])
        plt.savefig(path + '/sd(Z).svg')
        plt.close()
    return


def plot_xeff_beta(outputs, ews_point, 
                   color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2], ratio=20, 
                   outpath='weight_reduction_xbeta.svg'):
    if outputs.size == 0:
        print('No outputs to plot x_eff vs beta_eff.')
        return
    x_eff = outputs[::ratio, 1]
    beta_eff = outputs[::ratio, 2]
    frac = outputs[::ratio, 3]
    fig, ax = plt.subplots()
    sc = ax.scatter(beta_eff, x_eff, c=frac, cmap='OrRd_r', edgecolor='k')
    if ews_point > 0:
        # 获取当前 y 轴范围
        y_min, y_max = ax.get_ylim()  # 可能是 [-0.1, 2.1] 等自动扩展范围
        # 计算归一化的 ymax
        ymax_norm = float((outputs[ews_point, 1] - y_min) / (y_max - y_min))
        ax.scatter(outputs[ews_point, 2], outputs[ews_point, 1], color=color_map[0], s=150, zorder=5, marker="*")  # s 控制点大小
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\beta_{eff}$')
    plt.ylabel(r'$x_{eff}$')
    plt.title('x_eff vs beta_eff (color = fraction_reduce)')
    cbar = plt.colorbar(sc)
    cbar.set_label('fraction_reduce')
    plt.tight_layout()
    plt.savefig(outpath, format='svg')
    plt.close()


def plot_fraction_mean(outputs, ews_point, ratio=1,
                       color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2], outpath='weight_reduction_f_mean.svg', **arg):
    if outputs.size == 0:
        print('No outputs to plot fraction vs mean.')
        return
    frac = outputs[:, 3][::ratio]
    mean_x = outputs[:, 0][::ratio] 
    order = np.argsort(frac)[::-1]
    fig, ax = plt.subplots()
    plt.plot(frac[order], mean_x[order], '-o')
    ax.invert_xaxis()
    if ews_point > 0:
        # 获取当前 y 轴范围
        y_min, y_max = ax.get_ylim()  # 可能是 [-0.1, 2.1] 等自动扩展范围
        # 计算归一化的 ymax
        ymax_norm = float((mean_x[ews_point] - y_min) / (y_max - y_min))
        ax.scatter(frac[ews_point], mean_x[ews_point], color=color_map[0], s=150, zorder=5, marker="*")  # s 控制点大

    plt.xlabel('fraction_reduce')
    plt.ylabel('mean state (mean_x)')
    plt.title('Mean system state vs fraction_reduce')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, format='svg')
    plt.close()

def plot_fraction_xeff(outputs, ews_point, ratio=1,
                       color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2], outpath='weight_reduction_f_mean.svg', **arg):
    if outputs.size == 0:
        print('No outputs to plot fraction vs mean.')
        return
    frac = outputs[:, 3][::ratio]
    x_eff = outputs[:, 1][::ratio]
    order = np.argsort(frac)[::-1]
    fig, ax = plt.subplots()
    plt.plot(frac[order], x_eff[order], '-o')
    ax.invert_xaxis()
    # bif_point_para = arg['bif_point_para']
    # bif_point_obsvt = arg['bif_point_obsvt']
    # if bif_point_obsvt > 0:
    #         ax.axvspan(frac[bif_point_obsvt], frac[order[-1]], color=color_map[3], alpha=color_map[4])
    #         ax.axvline(x=frac[bif_point_obsvt], color=color_map[2], linestyle='--', linewidth=1.5, label="observed bifurcation")
    # if bif_point_para > 0:
    #     ax.axvline(x=frac[bif_point_para], color=color_map[1], linestyle='--', linewidth=0.75, label="parameter bifurcation")
    #     if bif_point_obsvt < 0:
    #         ax.axvspan(frac[bif_point_para], frac[order[-1]], color=color_map[3], alpha=color_map[4])
    # if ews_point > 0:
    #     if bif_point_para < 0 and bif_point_obsvt < 0:
    #         ax.axvspan(frac[ews_point], frac[order[-1]], color=color_map[3], alpha=color_map[4])
    #     ax.axvline(x=frac[ews_point], color=color_map[0], linestyle='--', linewidth=1.5, label="early warning signal")
    if ews_point > 0:
        # 获取当前 y 轴范围
        y_min, y_max = ax.get_ylim()  # 可能是 [-0.1, 2.1] 等自动扩展范围
        # 计算归一化的 ymax
        ymax_norm = float((x_eff[ews_point] - y_min) / (y_max - y_min))
        ax.scatter(frac[ews_point], x_eff[ews_point], color=color_map[0], s=150, zorder=5, marker="*")  # s 控制点大

    plt.xlabel('fraction_reduce')
    plt.ylabel('x_eff')
    plt.title('x_eff vs fraction_reduce')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, format='svg')
    plt.close()
