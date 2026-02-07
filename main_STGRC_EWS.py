# %% #import part
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# # 然后再 import numpy/scipy

import importlib
import pandas as pd
importlib.invalidate_caches()
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg
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
#%% Settings
arg, groups = fg.args()
# ##################### Data Mark ######################
parent_doc = 'EWS'
arg['dataset'] = 'mutualistic_weight_reduction' # 'mutualistic_weight_reduction', 'GRN_weight_reduction', 'gLV_oscillation_002'
data_start = 0 # 0, 2500, 0
arg['ns'] = 0.05 # noise strength of data
arg['bif_point_para'] = -1
arg['bif_point_obsvt'] = -1 #9250-data_start, 10500-data_start, -1
arg['win_step'] = 2
arg['num_win'] = 900
# ##################### Algorithm Setting ######################
arg['GPUid'] = '2'
arg['NNkey'] = 'graphRC'  # 'graphRC', 'RC'
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
print("✅ 参数设置成功！")
# %% 主要部分
# for arg['NNkey'] in ['RC', 'graphRC']:
for iter in range(1):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    print('\nStart time: ', timestamp)
    # #############读取数据#############
    if any(s in arg['dataset'] for s in ["GRN", "mutualistic", 'gLV', 'CLorenz']):
        # #########建立存储结果文件夹#########
        path = "/home/yangna/JetBrains/graphRC/Figures/%s/[%s] %s" % (parent_doc, timestamp, arg['dataset'])
        if not os.path.exists(path):
            os.makedirs(path)
        # #########  读取data  ###############
        data_all = np.load('/home/yangna/JetBrains/graphRC/Data/%s.npz' %arg['dataset'])
        X_trim = data_all['X_trim']  # (time, nodes) / (time, f, nodes)
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

    [f, nodes, length] = np.shape(data)  # features, nodes, total_length
    length_used = arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1)

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
    if arg['NNkey'] == 'RC':
        data_noise = data_noise.reshape(f*nodes, len)
        main_results, arg = fg.stRC(data_pack, path, groups, **arg)
    elif arg['NNkey'] == 'graphRC':
        main_results, arg = fg.stgRC(data_pack, path, groups, **arg)
    else:
        print("Wrong parameter for networks!")
        exit()


    print("✅ gtRC 计算完成！")
#%% ################################      保存结果        ####################################
    import importlib
    importlib.reload(fg)
    sdZ = main_results["sd(Z)"]
    fg.Save_mainresluts(main_results, path, **arg)
    summary = pd.read_csv(path + '/summary_%s.txt' % arg['label'], sep=',',
                               header=0,  # 使用第一行作为列名: devs1_real,devs1_imag,devs1_str,
                               # devs2_real,devs2_imag,devs2_str,sd(Z)
                               dtype=None)  # 自动推断类型
    print("✅ stARC 结果已保存！")
    # %% EWS
    bif_point_obsvt = arg['bif_point_obsvt']
    bif_point_para = arg['bif_point_para']
    if bif_point_obsvt > 0 and bif_point_obsvt <= arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1):
        bif_window_obsvt = int(np.ceil((bif_point_obsvt - arg['win_m']) / arg['win_step'] + 1))
    else:
        bif_window_obsvt = -1
    if bif_point_para > 0 and bif_point_para <= arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1):
        bif_window_para = int(np.ceil((bif_point_para - arg['win_m']) / arg['win_step'] + 1))
    else:
        bif_window_para = -1
    sdZ = sdZ.cpu().numpy()
    sdZ_ews = sdZ[:bif_window_obsvt] if bif_window_obsvt>0 else sdZ
    EWS_idx, pvalue = fg.EWS_bocd(sdZ_ews, path=path)
    print("Early warning window: %i,\nBifurcation window: %i\n" % (EWS_idx, bif_window_para))
    ews_point = EWS_idx * arg["win_step"] + arg["win_m"] ##
    arg['EWS_win'] = EWS_idx
    print(type(arg['EWS_win']))
    print("Early warning signal: %i,\nParameter bifurcation point: %i,\nObserved bifurcation point: %i\n"
          %(ews_point, arg['bif_point_para'], arg['bif_point_obsvt']))
    
    with open(path + '/EWS conclusion_%s.txt' % arg['label'], "w", encoding="utf-8") as f:
            f.write("Early warning window: %i,\nBifurcation window: %i,\n" % (EWS_idx, bif_window_para))
            f.write("Early warning signal of used data: %i,\nParameter bifurcation point: %i,\nObserved bifurcation point: %i\n"
                    %(ews_point, arg['bif_point_para'], arg['bif_point_obsvt']))
            f.write("Early warning signal of completed data: %i" %(ews_point+data_start))
    fg.arg_save(path, arg, groups)


    # %% 可视化
    # 图像元素
    show_length = arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1)
    data_xlims = np.arange(0, show_length, 1)  # (arg['num_win']-1)* arg["win_step"] + arg["win_m"]+1
    win_xlims = np.arange(arg['win_m'], show_length + 1, arg['win_step'])
    
    if arg['NNkey'] == 'RC':
        data_show = data_noise[:, :show_length].T
    elif arg['NNkey'] == 'graphRC':
        data_show = (data_noise[:, 0, :show_length].reshape(-1, show_length)).T

    step = arg['win_step']
    ews_window = EWS_idx
    ews_point = EWS_idx * arg["win_step"] + arg["win_m"]
    #       绘制图表        #
    if not any(s in arg['dataset'] for s in ['CLorenz']):
        fg.plot_xeff_beta(outputs[data_start:data_start+show_length, :], ews_point,
                        outpath=os.path.join(path, '%s xbeta.svg'%arg['dataset']))
    fg.plot_fraction_mean(outputs[data_start:data_start+show_length, :], ews_point,
                        outpath=os.path.join(path, '%s f_mean.svg'%arg['dataset']), **arg)
    color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2]
    fg.plot_perf(summary, data_show, ews_point, ews_window,
                data_xlims, win_xlims, path, color_map, **arg)
    print("✅ 单次实验操作全部完成！")
print("✅ 所有实验操作完成！")
