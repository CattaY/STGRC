# %% #import part
import importlib
import json

import pandas as pd
importlib.invalidate_caches()
import matplotlib
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
import datetime
import numpy as np
import torch
#%% #reading files
path = "/home/yangna/JetBrains/graphRC/Figures/EWS/[2026-01-18  18:07:29] gLV_oscillation_002"
color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2] #['#0EF0B1', '#0E9CF0', '#F06A0E']

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
# #############读取数据#############
Model_list = ["Henon", "ADVP", "Lorenz"]
summary = pd.read_csv(path+'/summary_%s.txt' % label, sep=',',
                      header=0, # 使用第一行作为列名: devs1_real,devs1_imag,devs1_str,
                                # devs2_real,devs2_imag,devs2_str,sd(Z)
                      dtype=None)      # 自动推断类型

sdZ = summary['sd(Z)']
print("✅ 读取结果完成！")
#%%  可视化
import importlib
importlib.reload(fg)  # 关键步骤！
if arg['bif_point_obsvt'] > 0 and arg['bif_point_obsvt'] <= arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1):
    bif_window_obsvt = int(np.ceil((arg['bif_point_obsvt'] - arg['win_m']) / arg['win_step'] + 1))
    # bif_window_obsvt = arg['bif_point_obsvt']
else:
    bif_window_obsvt = -1
if arg['bif_point_para'] > 0 and arg['bif_point_para'] <= arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1):
    bif_window_para = int(np.ceil((arg['bif_point_para'] - arg['win_m']) / arg['win_step'] + 1))
    # bif_window_para = arg['bif_point_para']
else:
    bif_window_para = -1
sdZ_ews = sdZ#[:bif_window_obsvt] if bif_window_obsvt>0 else sdZ
EWS_idx, pvalue = fg.EWS_bocd(sdZ_ews, path=path)
print("Early warning window: %i,\nBifurcation window: %i\n" % (EWS_idx, bif_window_para),
      "Observed bifurcation window: %i\n" % bif_window_obsvt)
ews_point = EWS_idx * arg["win_step"] + arg["win_m"]
print("Early warning signal: %i,\tParameter bifurcation point: %i,\tObserved bifurcation point: %i\t"
          %(ews_point, arg['bif_point_para'], arg['bif_point_obsvt']))
# 图像元素
step = arg['win_step']
ews_window = EWS_idx
if any(s in arg['dataset'] for s in ["GRN", "mutualistic", "gLV", "CLorenz"]):
    data_all = np.load(path + '/data_noise_%s_ns=%.1f.npz' % (label, arg['ns']))
    data = data_all['data_noise']
    show_length = arg['win_m']+step*(arg['num_win']-1)
    data_xlims = np.arange(0, show_length, 1) #(arg['num_win']-1)* arg["win_step"] + arg["win_m"]+1
    win_xlims = np.arange(arg['win_m'], show_length+1, step)

    tmp = data[:, 0, :show_length]
    data_show = (tmp.reshape(-1, show_length)).T
     # ################################      绘制图表        ####################################
    fg.plot_perf(summary, data_show, ews_point, ews_window,
                data_xlims, win_xlims, path, color_map, **arg)
    outputs = np.load(path + '/outputs.npy')
    outputs = outputs[:show_length, :]  # 去掉时间列
    if not any(s in arg['dataset'] for s in ["CLorenz"]):
        fg.plot_xeff_beta(outputs, ews_point, ratio=10,
                          outpath=os.path.join(path, '%s xbeta.svg'%arg['dataset']))
        fg.plot_fraction_xeff(outputs, ews_point, 
                          outpath=os.path.join(path, '%s f_x.svg'%arg['dataset']), **arg)
    fg.plot_fraction_mean(outputs, ews_point,
                          outpath=os.path.join(path, '%s f_mean.svg'%arg['dataset']), **arg)
    
else:
    print("Finding no corresponding dataset.")
    exit()
print("Done.")
