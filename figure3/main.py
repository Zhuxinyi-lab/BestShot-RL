# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
from pathlib import Path
import json
# One MCS: N*N uniform random node draws WITH replacement; updates apply immediately.
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count
import networkx as nx
import time

plt.rcParams['font.family'] = 'Times New Roman'

# ================= 1. 配置与环境初始化 =================
N = 100  # 100 x 100 periodic lattice (10,000 agents)
SIZE = N * N
C = 0.5
L = 1.0
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.005
ROUNDS = 8000  # MCS
W_LIST = [0.0, 0.8, 0.99]
NUM_AVERAGE_RUNS = 1

# 记录快照的具体 MCS 步数 (T=0 会在循环外单独记录)
SNAPSHOT_STEPS = [20, 200, 1500, 8000]

# 预计算邻居 (采用周期性边界条件)
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

def random_argmax(values):
    max_val = np.max(values)
    candidates = np.flatnonzero(values == max_val)
    return np.random.choice(candidates)

# ================= 2. 核心异步演化任务 =================
def run_simulation(task):
    w_val, run_idx = task
    print(f"🚀 进程启动：开始执行 W={w_val} 的 {ROUNDS} 步演化...")
    np.random.seed(42 + run_idx)  # 固定种子保证可复现
    
    # 🌟 修改点：初态设置为随机分布 (50% 投资，50% 不投资)
    x = (np.random.rand(SIZE) > 0.5).astype(float)
    
    nb_counts = np.zeros(SIZE, dtype=np.int32)
    for i in range(SIZE):
        nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)
        
    # 收益矩阵提速: P_LOOKUP[自己投不投, 邻居有没有人投]
    P_LOOKUP = np.array([
        [-L, 0.0],
        [-C, -C]
    ])

    q_table = np.zeros((SIZE, 3, 2)) 

    payoff_history = np.zeros(ROUNDS)
    
    # 在演化开始前，先记录 T=0 的初态快照
    snapshots = [x.copy().reshape((N, N))]

    for t in range(1, ROUNDS + 1):
        # --- 随机异步更新 ---
        for _ in range(SIZE):
            i = np.random.randint(SIZE)
            old_action = x[i]
            s_curr = get_majority_state(nb_counts[i])
            
            if np.random.rand() < EPSILON:
                a_idx = np.random.randint(2)
            else:
                a_idx = random_argmax(q_table[i, s_curr])
            new_action = float(a_idx)
            
            # --- 状态更新 ---
            if new_action != old_action:
                x[i] = new_action
                diff = 1 if new_action > 0.5 else -1
                for nb in NEIGHBORS[i]:
                    nb_counts[nb] += diff
                    
            # --- Q表更新 ---
            pay_i = P_LOOKUP[int(x[i]), 1 if nb_counts[i] > 0 else 0]
            pi_nbs_sum = sum(P_LOOKUP[int(x[nb]), 1 if nb_counts[nb] > 0 else 0] for nb in NEIGHBORS[i])
            reward = (1 - w_val) * pay_i + w_val * (pi_nbs_sum / 4.0)
                
            s_next = get_majority_state(nb_counts[i])
            a_idx_new = 1 if new_action > 0.5 else 0
            target = reward + GAMMA * np.max(q_table[i, s_next])
            q_table[i, s_curr, a_idx_new] += ALPHA * (target - q_table[i, s_curr, a_idx_new])

        # --- 记录全局宏观数据 ---
        has_nb_inv = (nb_counts > 0).astype(int)
        self_inv = x.astype(int)
        payoff_history[t-1] = np.mean(P_LOOKUP[self_inv, has_nb_inv])
        
        # --- 记录快照 ---
        if t in SNAPSHOT_STEPS:
            snapshots.append(x.copy().reshape((N, N)))
            
        if t % 8000 == 0:
            print(f"   [W={w_val}] -> 已完成 {t} 步")

    from pathlib import Path
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    stem = f'w{w_val}_run{run_idx}_seed{42 + run_idx}'
    np.savetxt(out / (stem + '_history.csv'),
               np.column_stack((np.arange(1, ROUNDS + 1), payoff_history)),
               delimiter=',', header='MCS,average_social_payoff', comments='')
    np.savez_compressed(out / (stem + '_states.npz'),
                        mcs=np.array([0, *SNAPSHOT_STEPS]), states=np.array(snapshots, dtype=np.uint8))
    return w_val, run_idx, payoff_history, snapshots

# ================= 3. 高质量绘图引擎 =================
def plot_results(results_dict, output_path=None, *, w_values=None,
                 snapshot_steps=None, save_data=True):
    """原版三行六列布局：正方形快照紧凑对齐，放大左侧坐标字号。"""
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent
    output = Path(output_path) if output_path else output_dir / "Comparisons_Payoff_and_Snapshots_Fixed.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    w_values = W_LIST if w_values is None else w_values
    snapshot_mcs = (0, *SNAPSHOT_STEPS) if snapshot_steps is None else snapshot_steps
    if len(w_values) != 3 or len(snapshot_mcs) != 5:
        raise ValueError("Figure 3 requires three rows and five snapshots per row")

    if save_data:
        provenance = {
            "data_kind": "original_simulation_payoff_averages_and_first_run_snapshots",
            "N": N, "C": C, "L": L, "alpha": ALPHA, "gamma": GAMMA,
            "epsilon": EPSILON, "rounds": ROUNDS,
            "number_of_runs": NUM_AVERAGE_RUNS,
            "w_values": w_values, "snapshot_mcs": snapshot_mcs,
        }
        save_archive(results_dict, output_dir / "figure3_raw_results.npz",
                     provenance, w_values, snapshot_mcs)

    # 下面的尺寸均按论文里最终显示的英寸/pt设置，避免缩图后字变小。
    figure_width = 6.69
    left_margin, right_margin = 0.59, 0.015
    line_height = 0.9628
    line_snapshot_gap = 0.055
    snapshot_gap = 0.065
    row_gap = 0.17
    bottom_margin, top_margin = 0.34, 0.11
    tick_fontsize, label_fontsize, panel_fontsize = 8.5, 9, 10

    snapshot_size = (figure_width - left_margin - right_margin - line_snapshot_gap - 4 * snapshot_gap) / 6
    line_width = snapshot_size
    snapshot_left = left_margin + line_width + line_snapshot_gap
    row_pitch = line_height + row_gap
    figure_height = bottom_margin + 3 * line_height + 2 * row_gap + top_margin
    style = {
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "path.simplify": False,
        "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.55,
    }

    with plt.rc_context(style):
        fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
        cmap_snap = ListedColormap(["#d94b4b", "#2f6fe4"])
        for row, w_val in enumerate(w_values):
            result = results_dict[w_val]
            payoff_hist = np.asarray(result["payoff_avg"])
            steps_arr = np.asarray(result.get("mcs", np.arange(1, len(payoff_hist) + 1)))
            row_bottom = bottom_margin + (2 - row) * row_pitch

            # 曲线框与快照严格等宽，保留原字号、纵向尺寸和快照间隔。
            ax_line = fig.add_axes([left_margin / figure_width, row_bottom / figure_height,
                                    line_width / figure_width, line_height / figure_height])
            ax_line.plot(steps_arr, payoff_hist, color="orangered", lw=0.85)
            ax_line.set_xlim(0, snapshot_mcs[-1])
            ax_line.set_ylim(-1, 0)
            ax_line.set_ylabel("Average social payoff", fontsize=label_fontsize, labelpad=0.5)
            ax_line.xaxis.set_major_locator(MultipleLocator(4000))
            ax_line.xaxis.set_minor_locator(MultipleLocator(500))
            ax_line.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_line.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax_line.tick_params(which="major", labelsize=tick_fontsize,
                                length=2.6, width=0.55, pad=1)
            ax_line.tick_params(which="minor", length=1.3, width=0.5)
            if row == 2:
                ax_line.set_xlabel("MCS", fontsize=label_fontsize, labelpad=1)
            # (a)(b)(c)仍位于图左端，时间与w沿用原图图注说明。
            fig.text(0.018 / figure_width,
                     (row_bottom + line_height + 0.015) / figure_height,
                     f"({chr(97 + row)})", fontsize=panel_fontsize, ha="left", va="top")

            snapshots = result["snapshots"]
            if len(snapshots) != 5:
                raise ValueError("Five snapshots including the initial state are required")
            for col, snapshot in enumerate(snapshots):
                x = snapshot_left + col * (snapshot_size + snapshot_gap)
                snapshot_bottom = row_bottom + (line_height - snapshot_size) / 2
                ax_snap = fig.add_axes([x / figure_width, snapshot_bottom / figure_height,
                                       snapshot_size / figure_width, snapshot_size / figure_height])
                ax_snap.imshow(snapshot, cmap=cmap_snap, vmin=0, vmax=1,
                               interpolation="none", origin="upper", aspect="equal")
                ax_snap.set_xticks([])
                ax_snap.set_yticks([])
                for spine in ax_snap.spines.values():
                    spine.set_color("black")
                    spine.set_linewidth(0.55)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
        boxes += [text.get_window_extent(renderer) for text in fig.texts]
        for box in boxes:
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                plt.close(fig)
                raise ValueError(f"Figure 3 text outside the canvas: {box}")
        fig.savefig(output, format="pdf", dpi=600, facecolor="white")
        fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
        plt.close(fig)
    print(f"Figure saved: {output}")

# ================= 4. 主程序启动 =================
def save_archive(results, output: Path, provenance: dict,
                 w_values=(0.0,0.8,0.99), snapshot_steps=(0,20,200,1500,8000)):
    arrays = {"w_values": np.array(w_values), "snapshot_mcs": np.array(snapshot_steps)}
    for row, w in enumerate(w_values):
        result = results[w]
        arrays[f"mcs_{row}"] = np.asarray(result.get("mcs", np.arange(1, len(result["payoff_avg"]) + 1)))
        arrays[f"payoff_{row}"] = np.asarray(result["payoff_avg"])
        arrays[f"snapshots_{row}"] = np.stack(result["snapshots"])
    arrays["provenance_json"] = np.array(json.dumps(provenance, ensure_ascii=False))
    np.savez_compressed(output, **arrays)
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

if __name__ == "__main__":
    start_time = time.time()
    num_procs = min(3, cpu_count())
    
    results = {w_val: {'payoff_runs': [], 'snapshots': None} for w_val in W_LIST}
    tasks = [(w_val, run_idx) for w_val in W_LIST for run_idx in range(NUM_AVERAGE_RUNS)]
    with Pool(processes=num_procs) as pool:
        for w_val, run_idx, p_hist, snaps in pool.imap_unordered(run_simulation, tasks):
            results[w_val]['payoff_runs'].append(p_hist)
            if run_idx == 0:
                results[w_val]['snapshots'] = snaps
    
    for w_val in W_LIST:
        results[w_val]['payoff_avg'] = np.mean(results[w_val]['payoff_runs'], axis=0)
            
    print(f"✅ 所有模拟运行完毕！总耗时: {time.time() - start_time:.2f} 秒.")
    
    # 绘制图像
    plot_results(results)
