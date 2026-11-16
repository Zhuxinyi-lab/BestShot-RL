# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
# One MCS: N*N uniform random node draws WITH replacement; updates apply immediately.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import time

# ================= 1. 全局配置与参数 =================
STEP_W0 = 50         # w=0 时的演化步数
STEP_W08 = 1900      # w=0.8 时的演化步数

N = 100  # 100 x 100 periodic lattice (10,000 agents)
SIZE = N * N
C = 0.5              # 投资成本
L = 1.0              # 损失
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.005 

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'

# ================= 2. 网络与状态初始化 =================
def init_grid():  
    G = nx.grid_2d_graph(N, N, periodic=True)
    node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
    neighbors = [[] for _ in range(SIZE)]
    for node in G.nodes():
        u = node_to_id[node]
        for v_node in G.neighbors(node):
            neighbors[u].append(node_to_id[v_node])
    return neighbors

NEIGHBORS = init_grid()

def get_majority_state(nb_inv):
    if nb_inv > 2: return 0
    elif nb_inv == 2: return 1
    else: return 2

# ================= 3. 核心演化算法 =================
def run_snapshot_simulation(w_val, target_steps):
    print(f"⏳ 正在运行 w={w_val} 的模拟，目标步数: {target_steps} MCS...")
    
    np.random.seed(43)
    x = np.random.choice([0.0, 1.0], size=SIZE)
    
    nb_counts = np.zeros(SIZE, dtype=np.int32)
    for i in range(SIZE):
        nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)

    q_table = np.zeros((SIZE, 3, 2))
    P_LOOKUP = np.array([
        [-L, 0.0],
        [-C, -C]
    ])

    start_time = time.time()
    for t in range(target_steps):
        for _ in range(SIZE):
            i = np.random.randint(SIZE)
            old_action = x[i]
            s_curr = get_majority_state(nb_counts[i])
            
            if np.random.rand() < EPSILON:
                a_idx = np.random.randint(2)
            else:
                a_idx = np.random.choice(np.flatnonzero(q_table[i, s_curr] == np.max(q_table[i, s_curr])))
            new_action = float(a_idx)
            
            if new_action != old_action:
                x[i] = new_action
                diff = 1 if new_action > 0.5 else -1
                for nb in NEIGHBORS[i]:
                    nb_counts[nb] += diff
            
            pay_i = P_LOOKUP[int(x[i]), 1 if nb_counts[i] > 0 else 0]
            if w_val > 0:
                pi_nbs_sum = sum(P_LOOKUP[int(x[nb]), 1 if nb_counts[nb] > 0 else 0] for nb in NEIGHBORS[i])
                reward = (1 - w_val) * pay_i + w_val * (pi_nbs_sum / 4.0)
            else:
                reward = pay_i
                
            s_next = get_majority_state(nb_counts[i])
            target = reward + GAMMA * np.max(q_table[i, s_next])
            q_table[i, s_curr, int(new_action)] += ALPHA * (target - q_table[i, s_curr, int(new_action)])

    print(f"✅ 完成！耗时: {time.time() - start_time:.2f} 秒")
    from pathlib import Path
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    grid = x.astype(np.uint8).reshape(N, N)
    stem = f'w{w_val}_MCS{target_steps}_seed43'
    np.save(out / (stem + '.npy'), grid)
    np.savetxt(out / (stem + '.csv'), grid, fmt='%d', delimiter=',')
    return grid

# ================= 4. 单图渲染与保存引擎 =================
def plot_and_save_single_snapshot(grid2d, w_val, steps):
    """
    生成单张网络图并独立保存
    - 黑色圆圈：投资 (1)
    - 白色圆圈：不投资 (0)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    
    # 1. 绘制浅灰色背景连线
    for r in range(N):
        y = N - 1 - r
        ax.plot([-0.5, N-0.5], [y, y], color='lightgray', lw=1.5, zorder=1)
    for c in range(N):
        ax.plot([c, c], [-0.5, N-0.5], color='lightgray', lw=1.5, zorder=1)

    # 2. 绘制节点
    for r in range(N):
        for c in range(N):
            y = N - 1 - r
            x = c
            val = grid2d[r, c]
            
            # 颜色判定：黑色=投资，白色=不投资
            fc = '#2f6fe4' if val > 0.5 else '#d94b4b'
            ec = 'black' 
            
            circle = patches.Circle((x, y), radius=0.4, facecolor=fc, edgecolor=ec, lw=1.5, zorder=3)
            ax.add_patch(circle)

    ax.set_xlim(-0.8, N - 0.2)
    ax.set_ylim(-0.8, N - 0.2)
    ax.set_title(f'w = {w_val} (MCS = {steps})', fontsize=22, pad=15)
    ax.axis('off')

    plt.tight_layout()
    
    # 根据 w 值动态命名文件
    save_path = f"Network_Raw_w{str(w_val).replace('.', '')}_MCS{steps}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig) # 生成后关闭画板，防止重叠
    print(f"🎨 独立快照已成功生成并保存为: {save_path}")

# ================= 5. 启动入口 =================
def plot_snapshots():
    # 运行并生成 w = 0.0 的单独图片
    grid_w0 = run_snapshot_simulation(w_val=0.0, target_steps=STEP_W0)
    plot_and_save_single_snapshot(grid_w0, w_val=0.0, steps=STEP_W0)
    
    # 运行并生成 w = 0.8 的单独图片
    grid_w08 = run_snapshot_simulation(w_val=0.8, target_steps=STEP_W08)
    plot_and_save_single_snapshot(grid_w08, w_val=0.8, steps=STEP_W08)

if __name__ == "__main__":
    plot_snapshots()
