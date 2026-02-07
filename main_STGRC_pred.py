# %% #import part
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import importlib
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
importlib.invalidate_caches()
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg
import sys
# 定义模块绝对路径
MODULE_PATH = "/home/yangna/JetBrains/graphRC"
MODULE_FILE = "func_STGRC.py"
FULL_PATH = os.path.join(MODULE_PATH, MODULE_FILE)
# 验证文件是否存在
if not os.path.exists(FULL_PATH):
    raise FileNotFoundError(f"找不到模块文件: {FULL_PATH}")
# 添加路径到Python搜索路径
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)  # 优先搜索
# 导入模块
try:
    import func_STGRC as fg
    print("✅ 模块导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"当前Python路径: {sys.path}")
import datetime
import numpy as np
import torch
#%% Settings
arg, groups = fg.args()
# ##################### Data Mark ######################
parent_doc = 'Prediction'
arg['dataset'] = 'mutualistic_weight_reduction'# gLV_oscillation_002, CLorenz_weight_reduction [2026-01-19  17:24:32], GRN_weight_reduction, mutualistic_weight_reduction
data_start = 0 # 0, 5700, 2500, 0
EWS_point =  893 # 1210, 6300, 1199+2500, 893
arg['target']=(0, 0)
arg['ns'] = 0.05 # noise strength of data
# ##################### Algorithm Setting ######################
arg['GPUid'] = '1'
arg['NNkey'] = 'graphRC'  # 'graphRC', 'RC', 'N'
device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
# ####################### Reservoir ########################
arg['fun_act'] = 'tanh'
arg['warmup_steps'] = 10
arg['aa'] = 1
arg['alpha'] = 0.5  # (1-alpha)r^t+alpha*f()
arg['res_nodes'] = 20
arg['deg'] = 0.2
arg['rho'] = 0.8
# ######################### stPCA ##########################
arg['win_m'] = 25
arg['L'] = 12
arg['lam'] = 0.2
# ######################### Prediction ##########################
arg['win_step'] = 2
arg['num_win'] = 900
EWS_idx = (EWS_point-data_start-arg["win_m"])//arg["win_step"]-1 if EWS_point>data_start else -1 # EWS window index
arg['EWS_win'] = EWS_idx
length_used = arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1) + arg['L']
length_pred = arg['num_win']*arg['L']
print("✅ 参数设置成功！")
# %% 主要部分
# for arg['NNkey'] in ['RC', 'graphRC']:
# for arg['ns'] in [0.02, 0.05]:
for iter in range(1):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    print('\nStart time: ', timestamp)
    # #############读取数据#############
    if any(s in arg['dataset'] for s in ["GRN", "mutualistic", "gLV", "CLorenz"]):
        # #########建立存储结果文件夹#########
        path = "/home/yangna/JetBrains/graphRC/Figures/%s/[%s] %s_ns=%.2f" % (parent_doc, timestamp, arg['dataset'], arg['ns'])
        if not os.path.exists(path):
            os.makedirs(path)
        # #########  读取data  ###############
        data_all = np.load('/home/yangna/JetBrains/graphRC/Data/%s.npz' %arg['dataset'])
        X_trim = data_all['X_trim']  # (time, nodes) / (time, nodes, f)
        outputs = data_all['outputs']  # (time, 4)
        if X_trim.ndim == 2:
            nodes = X_trim.shape[1]
            data = X_trim.T.reshape(1, nodes, -1)  # f*nodes*length
            data = data[:, :, data_start:]  # f*nodes*length
        else:
            data = np.transpose(X_trim, (2, 1, 0))  # (time, nodes, f) -> (f, nodes, time)
            data = data[:, :, data_start:]  # f*nodes*length
        if not any(s in arg['dataset'] for s in ['CLorenz']):
            ubase = data_all['A0']
            coupling = outputs[data_start:, 3]
        else:
            coupling = np.ones(len(outputs)-data_start)
        np.save(path+'/outputs.npy', outputs[data_start:,:])
    else:
        print("Finding no corresponding dataset.")
        exit()

    length_used = arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1)
    if "CLorenz" in arg['dataset']:
        stds = np.std(data[:, :, :length_used], axis=2)
        arg['target'] = np.unravel_index(np.argmax(stds), stds.shape)
    print("Target selected: ", arg['target'])
    [f, nodes, length] = np.shape(data)  # features, nodes, total_length

    data_noise = np.stack([
        data[i, j, :] + arg["ns"] * np.std(data[i, j, :length_used]) * np.random.randn(length)
        for i in range(f) for j in range(nodes)
    ]).reshape(f, nodes, length)

    
    comments = """# data with noise - 数组\n# 行向量为data的多维变量\n# 数据开始："""
    np.savez(path + '/data_noise_%s_ns=%.1f.npz' % (arg["label"], arg["ns"]), 
             data_noise=data_noise, 
             A0=data_all['A0'] if 'A0' in data_all else None, 
             coupling=coupling)
    
    data_pack = {"data_noise": data_noise,
                "ubase": data_all['A0'] if 'A0' in data_all else None,
                "coupling": coupling}

    main_results, arg = fg.stgRC_pre(data_pack, path, groups, **arg) # {"y_preds": y_pred_pool, "copy_ys": copy_y_pool}, arg
    print("✅ stgRC 预测完成！")
#%% ################################      保存结果        ####################################
    import importlib
    importlib.reload(fg)
    y_preds = main_results["y_preds"]
    y_preds = y_preds.detach().cpu().numpy()
    np.save(path + "/y_preds_%s.npy" % arg["label"], y_preds)
               # header=comments, comments='',  # 禁用默认的注释符
               # fmt='%.6f')  # 保留6位小数
    print("✅ stgRC 结果已保存！")

# %% 可视化
    # 图像元素
    tar0, tar1 = arg['target']
    batchx_idx = [np.arange(i * arg['win_step'], i * arg['win_step'] + arg['win_m'])
    for i in range(arg['num_win'])
    ]
    batchy_idx = [np.arange(i * arg['win_step']+arg['win_m']-1, i * arg['win_step']+arg['win_m']+arg['L']-1)
    for i in range(arg['num_win'])
    ]

    # 计算每个窗口的PCC和RMSE
    norm_term = np.std(data_noise[tar0, tar1, :arg['num_win']*arg['win_step']+arg['win_m']])
    window_metrics = []
    
    for win in range(arg['num_win']):
        show_x = data_noise[tar0, tar1, batchx_idx[win]]
        benchy = data_noise[tar0, tar1, batchy_idx[win]]
        predy = np.insert(y_preds[win, :], 0, benchy[0])
        
        [PCC, p_value] = pearsonr(predy, benchy)
        RMSE = np.sqrt(np.mean((predy-benchy) ** 2))/(norm_term+0.001)
        MAE = np.mean(np.abs(predy-benchy))/(norm_term+0.001)
        
        window_metrics.append({
            'win_idx': win + 1,
            'PCC': PCC,
            'RMSE': RMSE,
            'MAE': MAE
        })
        
        print("%i-th window prediction done and the PCC=%.4f, RMSE=%.4f" % (win+1, PCC, RMSE))

        if win==0:
            with open(path + '/Perf conclusion_%s.txt' % arg['label'], "w", encoding="utf-8") as f:
                f.write("win_idx\tPCC\tRMSE\tMAE\n")   
                f.write("%i\t%.6f\t%.6f\t%.6f\n" % (int(win+1), PCC, RMSE, MAE))
        else:
            with open(path + '/Perf conclusion_%s.txt' % arg['label'], "a", encoding="utf-8") as f: 
                f.write("%i\t%.6f\t%.6f\t%.6f\n" % (int(win+1), PCC, RMSE, MAE))

    # %%  对RMSE进行bocd检测

    # %%  可视化
    # 图1: 预测总图（灰色真实值，蓝色预测值）
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 15})
    
    # 计算预测值覆盖的时间范围
    pred_start_time = batchy_idx[0][0]
    pred_end_time = batchy_idx[-1][-1]
    
    # 绘制真实值（灰色）- 只展示预测值对应的部分
    time_points = np.arange(pred_start_time, pred_end_time + 1)
    ax.plot(time_points, data_noise[tar0, tar1, pred_start_time:pred_end_time+1], 
            color='gray', linewidth=1.5, label='True values', alpha=0.7)
    
    
    # 收集所有预测值，连接成一条连续的折线
    all_pred_times = []
    all_pred_values = []
    
    for win in range(arg['num_win']):
        benchy = data_noise[tar0, tar1, batchy_idx[win]]
        predy = np.insert(y_preds[win, :], 0, benchy[0])
        
        # 预测窗口的时间点
        pred_time = np.arange(batchy_idx[win][0], batchy_idx[win][-1] + 1)
        
        # 添加到总列表中
        all_pred_times.extend(pred_time)
        all_pred_values.extend(predy)
    df = pd.DataFrame({
    't': all_pred_times,
    'y': all_pred_values
    })

    df_mean = df.groupby('t', as_index=False).mean()

    ax.plot(df_mean['t'], df_mean['y'], color='blue', linewidth=1.5, alpha=0.6)
    # 绘制连接的预测值折线（蓝色）
    # ax.plot(all_pred_times, all_pred_values, color='blue', linewidth=1, alpha=0.6)
    
    # 标记临界点
    bif_point_obsvt = EWS_point - data_start
    ax.axvline(x=bif_point_obsvt, color='red', linestyle='--', linewidth=2, label='Critical point')
    
    # 标记临界期范围
    if EWS_idx >= 0 and EWS_idx < arg['num_win']:
        critical_start_time = batchy_idx[EWS_idx][0]
        critical_end_time = batchy_idx[EWS_idx][-1]
        ax.axvspan(critical_start_time, critical_end_time, alpha=0.2, color='yellow', label='Critical period')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Prediction Overview - {arg["dataset"]}\nσ={arg["ns"]}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path + '/Prediction_overview.svg')
    plt.close()

    # 图1.5: 真实总图（蓝色真实值）
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 15})
    
    # 计算预测值覆盖的时间范围
    pred_start_time = batchy_idx[0][0]
    pred_end_time = batchy_idx[-1][-1]
    
    # 绘制真实值（灰色）- 只展示预测值对应的部分
    time_points = np.arange(pred_start_time, pred_end_time + 1)
    ax.plot(time_points, data_noise[tar0, tar1, pred_start_time:pred_end_time+1], 
            color='gray', linewidth=1.5, label='True values', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(f'Real Data Overview - {arg["dataset"]}\nσ={arg["ns"]}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path + '/Real_data_overview.svg')
    plt.close()

    # 图2: 分窗口详细预测图（保持原有格式）
    for win in range(arg['num_win']):
        show_x = data_noise[tar0, tar1, batchx_idx[win]]
        benchy = data_noise[tar0, tar1, batchy_idx[win]]
        predy = np.insert(y_preds[win, :], 0, benchy[0])
        
        window_metrics_item = window_metrics[win]
        PCC = window_metrics_item['PCC']
        RMSE = window_metrics_item['RMSE']
        MAE = window_metrics_item['MAE']
        
        color_map = ['blue', 'cyan', 'red']
        plt.figure()
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({'font.size': 15})
        plt.plot(range(1, arg['win_m']+1, 1), show_x, '*-', color=color_map[0],
                markersize=2, linewidth=2, label='Observed data')
        plt.plot(range(arg['win_m'], arg['win_m']+arg['L'], 1), benchy, '*-', color=color_map[1],
                markersize=2, linewidth=2, label='Future data')
        plt.plot(range(arg['win_m'], arg['win_m']+arg['L'], 1), predy, 'o-', color=color_map[2],
                 markersize=2, linewidth=2,
                label='Prediction')
        plt.title('$\sigma$ = %.2f; PCC = %.4f, RMSE = %.4f, MAE = %.4f'
                    % (arg['ns'], PCC, RMSE, MAE), fontsize=15)
        plt.legend()
        plt.savefig(path + '/win=%i.svg' % (win+1))
        plt.close()

    # 图3: 预测误差随窗口变化图（折线图）
    preds_perf = pd.read_csv(path + '/Perf conclusion_%s.txt' % arg['label'], delimiter='\t')
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 15})
    
    # 提取RMSE值
    rmse_values = preds_perf['RMSE'].values
    win_indices = preds_perf['win_idx'].values
    
    # 5个窗口平滑
    window_size = 1
    smoothed_rmse = []
    smoothed_win_indices = []
    
    for i in range(len(rmse_values) - window_size + 1):
        smoothed_rmse.append(np.mean(rmse_values[i:i+window_size]))
        smoothed_win_indices.append(np.mean(win_indices[i:i+window_size]))
    point_indices = [
    i * arg['win_step'] + arg['win_m']
    for i in smoothed_win_indices
    ]
    # 绘制平滑后的折线图
    ax.plot(point_indices, smoothed_rmse, '-', c='#0B5D19', linewidth=2, label='RMSE')
    
    # 标记临界点
    ax.axvline(x=bif_point_obsvt+1, color='red', linestyle='--', linewidth=2, label="Critical point")
    
    ax.legend()
    ax.set_xlabel('Window Index')
    ax.set_ylabel('RMSE')
    ax.set_title('Prediction Errors (Smoothed by 5 Windows)')
    plt.tight_layout()
    plt.savefig(path + '/Prediction_errors_smoothed.svg')
    plt.close()
