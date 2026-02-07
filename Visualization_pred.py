# %% #import part
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
import numpy as np
from scipy.stats import pearsonr
#%% #reading files
path = "/home/yangna/JetBrains/graphRC/Figures/Prediction/[2026-01-29  18:09:48] CLorenz_weight_reduction [2026-01-29  16:00:27]_ns=0.00"

# 读取设置参数
def flatten_dict_generator(d, parent_key=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten_dict_generator(v, k)
        else:
            yield k, v
with open(path+'/arguments.json', 'r') as f:
    arguments = json.load(f)
arg = dict(flatten_dict_generator(arguments))
label = arg['label']
print(arg)
tar0, tar1 = arg['target']
EWS_idx = arg['EWS_win']
# #############读取数据#############
Model_list = ["Henon", "ADVP", "Lorenz"]
y_preds = np.load(path + '/y_preds_%s.npy'% arg["label"])
data_noise = np.load(path + '/data_noise_%s_ns=%.1f.npz' % (arg["label"], arg["ns"]))['data_noise']
norm_term = np.std(data_noise[tar0, tar1, :arg['num_win']*arg['win_step']+arg['win_m']])
print("✅ 读取结果完成！")

#%%  可视化
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

# # 收集所有预测值，连接成一条连续的折线
# all_pred_times = []
# all_pred_values = []

# for win in range(arg['num_win']):
#     benchy = data_noise[tar0, tar1, batchy_idx[win]]
#     predy = np.insert(y_preds[win, :], 0, benchy[0])
    
#     # 预测窗口的时间点
#     pred_time = np.arange(batchy_idx[win][0], batchy_idx[win][-1] + 1)
    
#     # 添加到总列表中
#     all_pred_times.extend(pred_time)
#     all_pred_values.extend(predy)

# # 绘制连接的预测值折线（蓝色）
# ax.plot(all_pred_times, all_pred_values, color='blue', linewidth=1, alpha=0.6)

# 标记临界点
EWS_point = (EWS_idx+1)*arg["win_step"]+arg["win_m"]
ax.axvline(x=EWS_point, color='red', linestyle='--', linewidth=2, label='Critical point')


ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title(f'Prediction Overview - {arg["dataset"]}\nσ={arg["ns"]}')
ax.legend()
plt.tight_layout()
plt.savefig(path + '/Prediction_overview.svg')
plt.close()

# 图2: 分窗口详细预测图（保持原有格式）
for win in range(arg['num_win']):
    show_x = data_noise[tar0, tar1, batchx_idx[win]]
    benchy = data_noise[tar0, tar1, batchy_idx[win]]
    predy = np.insert(y_preds[win, :], 0, benchy[0])
    
    window_metrics_item = window_metrics[win]
    PCC = window_metrics_item['PCC']
    RMSE = window_metrics_item['RMSE']
    
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
    plt.title('$\sigma$ = %.2f; PCC = %.4f, RMSE = %.4f'
                % (arg['ns'], PCC, RMSE), fontsize=15)
    plt.legend()
    plt.savefig(path + '/win=%i.svg' % (win+1))
    plt.close()

# 图3: 预测误差随窗口变化图（5个窗口平滑，折线图）
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

# 绘制平滑后的折线图
ax.plot(smoothed_win_indices, smoothed_rmse, 'b-', linewidth=2, label='RMSE')

# 标记临界点
ax.axvline(x=EWS_idx+1, color='red', linestyle='--', linewidth=2, label="Critical point")

ax.legend()
ax.set_xlabel('Window Index')
ax.set_ylabel('RMSE')
ax.set_title('Prediction Errors')
plt.tight_layout()
plt.savefig(path + '/Prediction_errors.svg')
plt.close()
