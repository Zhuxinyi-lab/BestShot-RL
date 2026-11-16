# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
# One MCS: N*N uniform random node draws WITH replacement; updates apply immediately.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
import networkx as nx
import time
from tqdm import tqdm

# ================= 1. 全局配置与参数初始化 =================
N = 100  # 100 x 100 periodic lattice (10,000 agents)
SIZE = N * N
L = 1.0
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.005

ITERATIONS = 8000  # MCS
RUNS = 5                     # 每个参数组合跑 5 次独立实验

# 需要扫描的 w 和 c 列表
W_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
C_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))

# 预计算邻居 (采用周期性边界条件)
def init_grid():
    G = nx.grid_2d_graph(N, N, periodic=True)
    node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
    neighbors = [[] for _ in range(SIZE)]
    for node in G.nodes():
        u = node_to_id[node]
        for v_node in G.neighbors(node):
            neighbors[u].append(node_to_id[v_node])
    # 转为numpy数组加速索引
    return np.array(neighbors, dtype=np.int32)

NEIGHBORS = init_grid()

# 状态映射提速: nb_inv=[0,1]->0, nb_inv=[2]->1, nb_inv=[3,4]->2
STATE_MAP = np.array([2, 2, 1, 0, 0], dtype=np.int32)

# ================= 2. 核心异步演化任务 =================
def run_single_combination(args):
    """ 处理单一 (w, c) 组合，内部循环 5 次随机初态 """
    w_val, c_val = args
    
    # 提前定义本任务的收益矩阵
    P_LOOKUP = np.array([
        [-L, 0.0],
        [-c_val, -c_val]
    ])

    run_payoffs = []
    run_investments = []
    
    # 5次不同随机初态独立运行
    for run_idx in range(RUNS):
        # 保证每次初态彻底随机
        np.random.seed(42 + run_idx + 100 * int(round(w_val * 100)) + 10000 * int(round(c_val * 100)))
        x = np.random.choice([0.0, 1.0], size=SIZE)
        
        nb_counts = np.zeros(SIZE, dtype=np.int32)
        for i in range(SIZE):
            nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)
            
        q_table = np.zeros((SIZE, 3, 2)) 
        
        pay_last_100 = []
        inv_last_100 = []

        for t in range(1, ITERATIONS + 1):
            # Random asynchronous updates, sampled with replacement.
            for _ in range(SIZE):
                i = np.random.randint(SIZE)
                old_action = x[i]
                s_curr = STATE_MAP[nb_counts[i]]
                
                # epsilon-greedy
                if np.random.rand() < EPSILON:
                    a_idx = np.random.randint(2)
                else:
                    a_idx = np.random.choice(np.flatnonzero(q_table[i, s_curr] == np.max(q_table[i, s_curr])))
                new_action = float(a_idx)
                
                # 状态即时更新
                if new_action != old_action:
                    x[i] = new_action
                    diff = 1 if new_action > 0.5 else -1
                    for nb in NEIGHBORS[i]:
                        nb_counts[nb] += diff
                        
                # Q表更新
                pay_i = P_LOOKUP[int(x[i]), 1 if nb_counts[i] > 0 else 0]
                pi_nbs_sum = sum(P_LOOKUP[int(x[nb]), 1 if nb_counts[nb] > 0 else 0] for nb in NEIGHBORS[i])
                reward = (1 - w_val) * pay_i + w_val * (pi_nbs_sum / 4.0)
                    
                s_next = STATE_MAP[nb_counts[i]]
                a_idx_new = 1 if new_action > 0.5 else 0
                target = reward + GAMMA * np.max(q_table[i, s_next])
                q_table[i, s_curr, a_idx_new] += ALPHA * (target - q_table[i, s_curr, a_idx_new])

        # 将本次 run 最后 100 步的平均值记录下来
            if t > ITERATIONS - 100:
                pay_last_100.append(np.mean(P_LOOKUP[x.astype(int), (nb_counts > 0).astype(int)]))
                inv_last_100.append(np.mean(x))

        run_payoffs.append(np.mean(pay_last_100))
        run_investments.append(np.mean(inv_last_100))
        from pathlib import Path
        out = Path(__file__).resolve().parent / 'results'
        out.mkdir(exist_ok=True)
        seed = 42 + run_idx + 100 * int(round(w_val * 100)) + 10000 * int(round(c_val * 100))
        stem = f'w{w_val}_c{c_val}_run{run_idx}_seed{seed}'
        np.save(out / (stem + '_final.npy'), x.astype(np.uint8).reshape(N,N))
        np.savetxt(out / (stem + '_window.csv'),
                   np.column_stack((np.arange(max(1,ITERATIONS-99),ITERATIONS+1),pay_last_100,inv_last_100)),
                   delimiter=',', header='MCS,average_social_payoff,investing_fraction', comments='')

    # 返回该 (w, c) 组合下，5次 run 的最终均值
    return {
        'w': w_val,
        'c': c_val,
        'Payoff': np.mean(run_payoffs),
        'Investment': np.mean(run_investments)
    }

# ================= 3. 高质量结果绘图 =================
def plot_and_save_data(df):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # 定义 7 种不同的颜色和 Marker 以区分 7 条线
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 将 DataFrame 透视，方便遍历画线
    df_payoff = df.pivot(index='w', columns='c', values='Payoff')
    df_inv = df.pivot(index='w', columns='c', values='Investment')

    # 遍历画 7 条线
    for i, c_val in enumerate(C_VALUES):
        # 收益图
        ax1.plot(df_payoff.index, df_payoff[c_val], marker=markers[i], color=colors[i], 
                 lw=2, markersize=7, label=f'c = {c_val}')
        # 投资图
        ax2.plot(df_inv.index, df_inv[c_val], marker=markers[i], color=colors[i], 
                 lw=2, markersize=7, label=f'c = {c_val}')

    # --- 设置左图 (收益) ---
    ax1.set_title('Steady-State Payoff vs. Social Preference ($w$)', fontsize=16)
    ax1.set_xlabel('Social Preference Weight ($w$)', fontsize=14)
    ax1.set_ylabel('Average Social Payoff', fontsize=14)
    ax1.set_xticks(W_VALUES)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(loc='lower right', fontsize=12)

    # --- 设置右图 (投资率) ---
    ax2.set_title('Steady-State Investment vs. Social Preference ($w$)', fontsize=16)
    ax2.set_xlabel('Social Preference Weight ($w$)', fontsize=14)
    ax2.set_ylabel('Average Investment Density', fontsize=14)
    ax2.set_xticks(W_VALUES)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(loc='lower left', fontsize=12)

    # 主标题
    plt.suptitle('State Design: Group Majority (5 Runs Average)', fontsize=20, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # 存图
    img_path = os.path.join(RESULT_DIR, 'SteadyState_7Lines.png')
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n🎉 演化图像已生成: {img_path}")
    plt.show()

# ================= 4. 主程序 =================
if __name__ == "__main__":
    print(f"🚀 系统就绪！配置：100x100网格 | 8000步 | 7条C线 | {RUNS}次独立实验求均值")
    
    # 构建所有的 (w, c) 参数对 (11 * 7 = 77 个任务)
    tasks = [(w, c) for w in W_VALUES for c in C_VALUES]
    
    num_procs = max(1, cpu_count() - 2)
    print(f"💻 开启 {num_procs} 个核心执行并发演化，请耐心等待 (总计算量较大)...\n")
    
    start_time = time.time()
    
    all_results = []
    with Pool(processes=num_procs) as pool:
        # 使用 tqdm 显示进度条
        for res in tqdm(pool.imap_unordered(run_single_combination, tasks), total=len(tasks), desc="Simulating"):
            all_results.append(res)
            
    print(f"\n✅ 所有模拟运行完毕！总耗时: {(time.time() - start_time) / 60:.2f} 分钟.")
    
    # ================= 5. 数据整理与 Origin 文件保存 =================
    df_raw = pd.DataFrame(all_results)
    df_raw = df_raw.sort_values(by=['c', 'w']).reset_index(drop=True)
    
    # 保存原始长表数据
    raw_csv = os.path.join(RESULT_DIR, 'Raw_Data_LongFormat.csv')
    df_raw.to_csv(raw_csv, index=False)
    
    # 特别制作适合 Origin 绘图的宽表 (列名为不同C值)
    df_payoff_origin = df_raw.pivot(index='w', columns='c', values='Payoff').add_prefix('Payoff_c=')
    df_inv_origin = df_raw.pivot(index='w', columns='c', values='Investment').add_prefix('Inv_c=')
    
    df_origin = pd.concat([df_payoff_origin, df_inv_origin], axis=1).reset_index()
    origin_csv = os.path.join(RESULT_DIR, 'Data_For_Origin.csv')
    df_origin.to_csv(origin_csv, index=False)
    
    print(f"👉 原始总表已保存: {raw_csv}")
    print(f"👉 Origin 专属表格已保存: {origin_csv} (请直接复制该表导入Origin作图)")

    # 绘图输出
    plot_and_save_data(df_raw)
