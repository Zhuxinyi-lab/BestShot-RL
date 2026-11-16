# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
# One MCS: N*N uniform random node draws WITH replacement; updates apply immediately.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing as mp

# ================= 1. 全局配置与参数 =================
N = 100  # 100 x 100 periodic lattice (10,000 agents)
SIZE = N * N
C = 0.5
L = 1.0
ROUNDS = 8000  # MCS

# 颜色配置 (RGB规范化到 0-1)
COLORS = {
    0.2: (181/255, 136/255, 163/255),
    0.3: (120/255, 163/255, 194/255),
    0.4: (241/255, 103/255, 84/255),
    0.5: (126/255, 193/255, 186/255)
}

DASH_COLOR_1 = (149/255, 149/255, 149/255)
DASH_COLOR_2 = (111/255, 126/255, 142/255)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ================= 2. 网络底层与严格的纳什初态构建 =================
def get_neighbors(i):
    r, c = divmod(i, N)
    return [((r - 1) % N) * N + c, ((r + 1) % N) * N + c,
            r * N + ((c - 1) % N), r * N + ((c + 1) % N)]
NEIGHBORS = [get_neighbors(i) for i in range(SIZE)]

def random_argmax(values):
    max_val = np.max(values)
    candidates = np.flatnonzero(values == max_val)
    return np.random.choice(candidates)

def generate_base_state(density):
    """生成严格满足纳什均衡(IDS)的初态"""
    if density == 0.2:
        state = np.zeros(SIZE)
        for r in range(N):
            for c in range(N):
                if (r * 2 + c) % 5 == 0: state[r*N + c] = 1.0
        return state
        
    elif density == 0.5:
        state = np.zeros(SIZE)
        for r in range(N):
            for c in range(N):
                if (r + c) % 2 == 0: state[r*N + c] = 1.0
        return state
        
    else:
        # 0.3 和 0.4 密度：通过从0.5(棋盘格)进行"安全降维剥离"来精确寻找纳什解
        target_ones = int(SIZE * density)
        while True:
            # 初始化为0.5棋盘格
            state = np.zeros(SIZE)
            for r in range(N):
                for c in range(N):
                    if (r + c) % 2 == 0: state[r*N + c] = 1.0
            
            current_ones = int(np.sum(state))
            stuck_counter = 0
            
            while current_ones > target_ones:
                zero_indices = np.where(state == 0)[0]
                np.random.shuffle(zero_indices)
                flipped = False
                
                # 尝试寻找一个能够安全减少总投资点数的操作
                for z_idx in zero_indices:
                    one_nbs = [nb for nb in NEIGHBORS[z_idx] if state[nb] == 1.0]
                    reduction = len(one_nbs) - 1
                    
                    if reduction > 0 and current_ones - reduction >= target_ones:
                        # 检查安全约束：被剥离的1变成0后，它们原来的0邻居是否还有其他的1保护？
                        safe = True
                        for o_idx in one_nbs:
                            for w_idx in NEIGHBORS[o_idx]:
                                if w_idx != z_idx:
                                    w_one_nbs = [nb for nb in NEIGHBORS[w_idx] if state[nb] == 1.0]
                                    if len(w_one_nbs) <= 1:
                                        safe = False
                                        break
                            if not safe: break
                        
                        if safe:
                            state[z_idx] = 1.0
                            for nb in one_nbs: state[nb] = 0.0
                            current_ones -= reduction
                            flipped = True
                            stuck_counter = 0
                            break
                
                # 如果没有减少数量的路径，尝试平移(中性步)以逃离局部死锁
                if not flipped:
                    for z_idx in zero_indices:
                        one_nbs = [nb for nb in NEIGHBORS[z_idx] if state[nb] == 1.0]
                        if len(one_nbs) == 1:
                            o_idx = one_nbs[0]
                            safe = True
                            for w_idx in NEIGHBORS[o_idx]:
                                if w_idx != z_idx:
                                    w_one_nbs = [nb for nb in NEIGHBORS[w_idx] if state[nb] == 1.0]
                                    if len(w_one_nbs) <= 1:
                                        safe = False
                                        break
                            if safe:
                                state[z_idx] = 1.0
                                state[o_idx] = 0.0
                                flipped = True
                                stuck_counter += 1
                                break
                                
                if not flipped or stuck_counter > 1500:
                    break # 陷入死锁，重置重试
                    
            if current_ones == target_ones:
                return state

def get_perturbed_state(base_state, H):
    """依据汉明距离精准翻转指定数量的节点"""
    state = base_state.copy()
    if H > 0:
        idx = np.random.choice(SIZE, H, replace=False)
        state[idx] = 1.0 - state[idx]
    return state

# ================= 3. 动力学演化算法 =================
def run_BR(init_state):
    x = init_state.copy()
    nb_counts = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)

    for _ in range(ROUNDS):
        for _ in range(SIZE):
            i = np.random.randint(SIZE)
            old_x = x[i]
            new_x = 1.0 if nb_counts[i] == 0 else 0.0
            if new_x != old_x:
                x[i] = new_x
                diff = 1 if new_x > 0.5 else -1
                for nb in NEIGHBORS[i]:
                    nb_counts[nb] += diff

    from pathlib import Path
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    import hashlib
    key = hashlib.sha256(np.asarray(init_state, dtype=np.uint8).tobytes()).hexdigest()[:16]
    np.savez_compressed(out / ("BR_" + key + '.npz'),
                        initial=np.asarray(init_state, dtype=np.uint8).reshape(N, N),
                        final=x.astype(np.uint8).reshape(N, N))
    final_density = np.mean(x)
    P_LOOKUP = np.array([[-L, 0.0], [-C, -C]])
    has_nb_inv = (nb_counts > 0).astype(int)
    self_inv = x.astype(int)
    final_payoff = np.mean(P_LOOKUP[self_inv, has_nb_inv])
    return final_density, final_payoff

def run_QL(init_state, w_val=0.8):
    x = init_state.copy()
    nb_counts = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)

    q_table = np.zeros((SIZE, 3, 2))
    P_LOOKUP = np.array([[-L, 0.0], [-C, -C]])
    STATE_MAP = np.array([2, 2, 1, 0, 0], dtype=int)

    alpha = 0.1
    gamma = 0.95
    epsilon = 0.005

    for _ in range(ROUNDS):
        for _ in range(SIZE):
            i = np.random.randint(SIZE)
            old_x = x[i]
            s_curr = STATE_MAP[nb_counts[i]]

            if np.random.rand() < epsilon:
                a_idx = np.random.randint(2)
            else:
                a_idx = random_argmax(q_table[i, s_curr])

            new_x = float(a_idx)
            if new_x != old_x:
                x[i] = new_x
                diff = 1 if new_x > 0.5 else -1
                for nb in NEIGHBORS[i]:
                    nb_counts[nb] += diff

            pi_i = P_LOOKUP[int(x[i]), 1 if nb_counts[i] > 0 else 0]
            pi_avg = np.mean([P_LOOKUP[int(x[nb]), 1 if nb_counts[nb] > 0 else 0] for nb in NEIGHBORS[i]])
            reward = (1 - w_val) * pi_i + w_val * pi_avg

            s_next = STATE_MAP[nb_counts[i]]
            target = reward + gamma * q_table[i, s_next, random_argmax(q_table[i, s_next])]
            q_table[i, s_curr, int(new_x)] += alpha * (target - q_table[i, s_curr, int(new_x)])

    from pathlib import Path
    import hashlib
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    key = hashlib.sha256(np.asarray(init_state, dtype=np.uint8).tobytes()).hexdigest()[:16]
    np.savez_compressed(out / ('QL_' + key + '.npz'),
                        initial=np.asarray(init_state, dtype=np.uint8).reshape(N, N),
                        final=x.astype(np.uint8).reshape(N, N))
    final_density = np.mean(x)
    has_nb_inv = (nb_counts > 0).astype(int)
    self_inv = x.astype(int)
    final_payoff = np.mean(P_LOOKUP[self_inv, has_nb_inv])
    return final_density, final_payoff

def simulate_task(task):
    alg, density, H, base_state = task
    np.random.seed(42 + int(round(density * 100)) * 10000 + H)
    
    init_state = get_perturbed_state(base_state, H)
    init_density = np.mean(init_state)
    
    nb_counts_init = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        nb_counts_init[i] = sum(1 for nb in NEIGHBORS[i] if init_state[nb] > 0.5)
    P_LOOKUP = np.array([[-L, 0.0], [-C, -C]])
    init_payoff = np.mean(P_LOOKUP[init_state.astype(int), (nb_counts_init > 0).astype(int)])

    if alg == 'BR':
        final_density, final_payoff = run_BR(init_state)
    else:
        final_density, final_payoff = run_QL(init_state, 0.8)

    print(f"✅ 完成: {alg}, 密度={density}, 汉明={H}")
    return (alg, density, H, init_density, init_payoff, final_density, final_payoff)

# ================= 4. 并发调度与高级出图引擎 =================
def main():
    np.random.seed(42)
    densities = [0.2, 0.3, 0.4, 0.5]
    Hs = [0, 100, 500, 1000, 2000]
    algs = ['BR', 'QL']

    print("🧩 正在通过剥离算法求解精确纳什基准状态...")
    base_states = {d: generate_base_state(d) for d in densities}
    for density, base in base_states.items():
        counts = np.sum(base[np.asarray(NEIGHBORS)], axis=1)
        assert np.isclose(np.mean(base), density)
        assert np.all(counts[base == 1] == 0), 'Initial investors must be independent'
        assert np.all(counts[base == 0] > 0), 'Initial pattern must cover all nodes'
    tasks = [(alg, d, H, base_states[d]) for alg in algs for d in densities for H in Hs]

    print("🚀 启动多进程并发演化计算 (共 40 组)...")
    num_cores = max(1, mp.cpu_count() - 1)
    with mp.Pool(num_cores) as pool:
        results = pool.map(simulate_task, tasks)

    from pathlib import Path
    import csv
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    with (out / 'figure7.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'base_density', 'H', 'initial_density', 'initial_payoff', 'final_density', 'final_payoff'])
        writer.writerows(results)
    print("🎨 数据收集完毕，开始高精度渲染图表...")
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    
    for i, alg in enumerate(['BR', 'QL']):
        ax_payoff = axes[i, 0]
        ax_invest = axes[i, 1]

        ax_payoff.axhline(-0.1, color='black', linestyle='--', lw=2, zorder=1)
        ax_payoff.axhline(-0.25, color=DASH_COLOR_1, linestyle='--', lw=2, zorder=1)
        
        ax_invest.axhline(0.5, color=DASH_COLOR_1, linestyle='--', lw=2, zorder=1)
        ax_invest.axhline(0.2, color='black', linestyle='--', lw=2, zorder=1)

        for res in results:
            r_alg, d, H, init_den, init_pay, fin_den, fin_pay = res
            if r_alg != alg: continue
            color = COLORS[d]

            ax_payoff.annotate("", xy=(init_den, fin_pay), xytext=(init_den, init_pay),
                               arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color=color, lw=2.5), zorder=2)
            ax_invest.annotate("", xy=(init_den, fin_den), xytext=(init_den, init_den),
                               arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.4", color=color, lw=2.5), zorder=2)

            ax_payoff.scatter(init_den, init_pay, s=150, facecolors='white', edgecolors=color, linewidth=2.5, zorder=3)
            ax_payoff.scatter(init_den, fin_pay, s=150, color=color, zorder=3)

            ax_invest.scatter(init_den, init_den, s=150, facecolors='white', edgecolors=color, linewidth=2.5, zorder=3)
            ax_invest.scatter(init_den, fin_den, s=150, color=color, zorder=3)

            ax_payoff.text(init_den + 0.005, init_pay, f"H={H}", fontsize=17, fontfamily='Times New Roman', color='#666666', va='center', zorder=4)
            ax_invest.text(init_den + 0.005, init_den, f"H={H}", fontsize=17, fontfamily='Times New Roman', color='#666666', va='center', zorder=4)

        for ax, ylabel in zip([ax_payoff, ax_invest], ['Average social payoff', 'Proportion of investing agents']):
            ax.set_xlabel("initial investment", fontsize=31, fontfamily='Times New Roman')
            ax.set_ylabel(ylabel, fontsize=31, fontfamily='Times New Roman')
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            legend_elements = [
                Line2D([0], [0], color=COLORS[0.2], lw=4, label='Cluster 0.2'),
                Line2D([0], [0], color=COLORS[0.3], lw=4, label='Cluster 0.3'),
                Line2D([0], [0], color=COLORS[0.4], lw=4, label='Cluster 0.4'),
                Line2D([0], [0], color=COLORS[0.5], lw=4, label='Cluster 0.5'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markerfacecolor='white', markersize=12, markeredgewidth=2, label='Initial State'),
                Line2D([0], [0], marker='o', color='w', markeredgecolor='none', markerfacecolor='black', markersize=12, label='Final State')
            ]
            ax.legend(handles=legend_elements, loc='best', prop={'family': 'Times New Roman', 'size': 20}, frameon=True, edgecolor='black')

    axes[0, 0].text(-0.12, 1.02, '(a)', transform=axes[0, 0].transAxes, fontsize=33, fontfamily='Times New Roman', va='top')
    axes[1, 0].text(-0.12, 1.02, '(b)', transform=axes[1, 0].transAxes, fontsize=33, fontfamily='Times New Roman', va='top')

    plt.tight_layout()
    save_path = 'Basin_of_Attraction_Final.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🎉 演化结束，高精度图片已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
