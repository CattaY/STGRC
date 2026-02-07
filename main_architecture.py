#!/usr/bin/env python3
"""
主架构文件：
1. 参数设置
2. 数据生成
3. EWS检测
4. 预测

可以直接运行，使用默认参数
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

# 添加模块路径
MODULE_PATH = "/home/yangna/JetBrains/graphRC"
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

try:
    import func_STGRC as fg
    from data_generation import generate_and_filter_data
    from ews_detection import detect_ews_batch, load_ews_results
    from prediction import predict_batch
    print("✅ 所有模块导入成功！")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def main():
    """主函数 - 使用默认参数"""
    # 获取func_graphRC的参数和分组
    arg, groups = fg.args()
    # ==================== 参数设置 ====================
    # 系统类型
    arg['system_type'] = 'gLV'  # 可选: 'gLV' 或 'CLorenz'
    arg['GPUid']='1'
    # 执行步骤控制
    arg['generate_data'] = False  # 是否生成数据
    arg['detect_ews'] = False     # 是否进行EWS检测
    arg['predict'] = True        # 是否进行预测
    
    # 数据生成参数
    arg['num_datasets'] = 0      # 需要生成的有效数据数量
    arg['n_nodes'] = 6            # 节点数量
    arg['n_features'] = 3         # 特征数量
    arg['total_length'] = 2000    # 数据总长度
    arg['seed'] = None            # 随机种子 (None表示随机)
    
    # EWS检测参数
    arg['runs_ews'] = 50          # 要检测的样本数量
    arg['data_start'] = 0         # 数据起始位置
    
    # 预测参数
    arg['runs_pred'] = 50          # 要预测的样本数量

    # 噪声参数（通用）
    arg['ns'] = 0             # 噪声强度（用于EWS和预测时的样本检测）
    
    # 网络参数
    arg['win_m'] = 25             # 窗口长度
    arg['L'] = 7                  # 嵌入维度
    arg['win_step'] = 10           # 窗口步长
    arg['num_win'] = 180          # 窗口数量
    arg['bif_point_para'] = -1    # 基于参数的临界点
    arg['bif_point_obsvt'] = -1   # 基于观测的临界点
    arg['GPUid'] = '1'            # GPU ID
    arg['NNkey'] = 'graphRC'      # 网络类型: 'graphRC' 或 'RC'
    arg['fun_act'] = 'tanh'       # 激活函数
    arg['warmup_steps'] = 10      # 预热步数
    arg['aa'] = 1                 # 缩放因子
    arg['alpha'] = 0.5            # 泄漏因子
    arg['res_nodes'] = 50         # 储备池节点数
    arg['deg'] = 0.2              # 平均度
    arg['rho'] = 0.8              # 谱半径
    arg['lam'] = 0.2              # lambda参数
    arg['target'] = (1, 1)        # 预测目标
    
    # ==================== 开始执行 ====================
    print("=" * 80)
    print(f"系统类型: {arg['system_type']}")
    print(f"生成数据: {arg['generate_data']}")
    print(f"EWS检测: {arg['detect_ews']}")
    print(f"预测: {arg['predict']}")
    print("=" * 80)
    
    # 步骤1: 生成数据
    if arg['generate_data']:
        print("\n[步骤1] 生成数据...")
        print("-" * 80)
        valid_count, metadata = generate_and_filter_data(
            system_type=arg['system_type'],
            num_datasets=arg['num_datasets'],
            n_nodes=arg['n_nodes'],
            n_features=arg['n_features'],
            total_length=arg['total_length'],
            seed=arg['seed']
        )
        print(f"✓ 生成完成: {valid_count} 条有效数据")
    
    # 步骤2: EWS检测
    if arg['detect_ews']:
        print("\n[步骤2] EWS检测...")
        print("-" * 80)
        ews_results = detect_ews_batch(
            runs=arg['runs_ews'],
            groups=groups,
            **arg
        )
        print(f"✓ EWS检测完成: {len(ews_results)} 条数据")
    
    # 步骤3: 预测
    if arg['predict']:
        print("\n[步骤3] 预测...")
        print("-" * 80)
        pred_results = predict_batch(
            runs=arg['runs_pred'],
            groups=groups,
            **arg
        )
        print(f"✓ 预测完成: {len(pred_results)} 个样本")
    
    print("\n" + "=" * 80)
    print("所有步骤完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
