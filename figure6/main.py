# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
from pathlib import Path
import json
# One MCS: N*N uniform random node draws WITH replacement; updates apply immediately.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
from collections import Counter 

# ================= 1. 全局配置参数 =================
N = 100  # 100 x 100 periodic lattice (10,000 agents)
SIZE = N * N
C = 0.5
L = 1.0
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.005 
ROUNDS = 8000  # MCS

# 设置全局字体和刻度朝内，确保包含负号等数学符号
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'Times New Roman'  # 修复：改为新罗马体
plt.rcParams['mathtext.fontset'] = 'custom'      # 修复：确保负号等数学符号也使用正常字体
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# ================= 2. 构型模板与物理底层 =================
# 构型 (a): W=0.0 的 12节点棋盘格
MOTIF_A = [
    (0,0,1), (0,1,0), (0,2,1), (0,3,0),
    (1,0,0), (1,1,1), (1,2,0), (1,3,1),
    (2,0,1), (2,1,0), (2,2,1), (2,3,0)
]

# 构型 (b1): W=0.8 的 12节点菱形核
MOTIF_B1 = [
    (0,1,1), (0,2,0),
    (1,0,0), (1,1,0), (1,2,0), (1,3,1),
    (2,0,1), (2,1,0), (2,2,0), (2,3,0),
    (3,1,0), (3,2,1)
]

# 构型 (b2): W=0.8 的 12节点菱形核
MOTIF_B2 = [
    (0,1,0), (0,2,1),
    (1,0,1), (1,1,0), (1,2,0), (1,3,0),
    (2,0,0), (2,1,0), (2,2,0), (2,3,1),
    (3,1,1), (3,2,0)
]

def get_neighbors(i):
    r, c = divmod(i, N)
    return [
        ((r - 1) % N) * N + c, ((r + 1) % N) * N + c,
        r * N + ((c - 1) % N), r * N + ((c + 1) % N)
    ]

NEIGHBORS = [get_neighbors(i) for i in range(SIZE)]

def get_state(nb_inv):
    if nb_inv > 2: return 0
    elif nb_inv == 2: return 1
    else: return 2

def random_argmax(values):
    max_val = np.max(values)
    candidates = np.flatnonzero(values == max_val)
    return np.random.choice(candidates)

def init_worst_nash_state(size, n):
    x = np.zeros(size)
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0: x[r*n + c] = 1.0
    return x

def get_motif_mask(g2d, motif_def):
    match = np.ones_like(g2d, dtype=bool)
    for dr, dc, val in motif_def:
        shifted = np.roll(g2d, shift=(-dr, -dc), axis=(0, 1))
        match &= (shifted == val)
        
    node_mask = np.zeros_like(g2d, dtype=bool)
    for dr, dc, _ in motif_def:
        node_mask |= np.roll(match, shift=(dr, dc), axis=(0, 1))
    return node_mask

# ================= 3. 核心演化与统计算法 =================
def run_simulation(w_val):
    np.random.seed(43)  # 固定种子
    
    x = np.random.choice([0.0, 1.0], size=SIZE)

    nb_counts = np.zeros(SIZE, dtype=int)
    for i in range(SIZE):
        nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)

    q_table = np.zeros((SIZE, 3, 2))
    history = np.zeros((ROUNDS, SIZE), dtype=np.uint8)
    history_ratio = np.zeros(ROUNDS)

    for r in range(ROUNDS):
        # 🌟 修复：使用随机异步更新
        for _ in range(SIZE):
            i = np.random.randint(SIZE)
            old_s = get_state(nb_counts[i])
            if np.random.rand() < EPSILON:
                a = np.random.randint(2)
            else:
                a = random_argmax(q_table[i, old_s])
            
            old_xi = x[i]
            x[i] = float(a)
            if x[i] != old_xi:
                diff = 1 if x[i] > 0.5 else -1
                for nb in NEIGHBORS[i]: nb_counts[nb] += diff
            
            def get_pi(idx):
                if x[idx] > 0.5: return -C
                return 0.0 if nb_counts[idx] > 0 else -L
            
            pi_i = get_pi(i)
            pi_avg = np.mean([get_pi(nb) for nb in NEIGHBORS[i]])
            reward = (1 - w_val) * pi_i + w_val * pi_avg
            
            new_s = get_state(nb_counts[i])
            future_val = q_table[i, new_s, random_argmax(q_table[i, new_s])]
            old_q = q_table[i, old_s, a]
            
            q_table[i, old_s, a] += ALPHA * (reward + GAMMA * future_val - old_q)
            
        history[r] = x.copy()
        
        # 统计宏观构型比例
        g2d = x.copy().reshape((N, N))
        if w_val == 0.0:
            mask = get_motif_mask(g2d, MOTIF_A)
            history_ratio[r] = np.sum(mask) / SIZE * 100.0
        else:
            mask = get_motif_mask(g2d, MOTIF_B1) | get_motif_mask(g2d, MOTIF_B2)
            history_ratio[r] = np.sum(mask) / SIZE * 100.0
            
    from pathlib import Path
    out = Path(__file__).resolve().parent / 'results'
    out.mkdir(exist_ok=True)
    stem = f'w{w_val}_seed43'
    np.savez_compressed(out / (stem + '_states.npz'),
                        mcs=np.arange(1, ROUNDS + 1), states=history.reshape(ROUNDS, N, N))
    np.savetxt(out / (stem + '_motif.csv'),
               np.column_stack((np.arange(1, ROUNDS + 1), history_ratio)),
               delimiter=',', header='MCS,stable_fraction_percent', comments='')
    return history, history_ratio

def get_solidified_nodes(history, start_r, end_r=2500):
    # 此函数在本版本中未使用，但遵照“其余不要改动”的指令予以保留
    total_steps = end_r - start_r + 1
    threshold = 0.95
    
    structure_centers = set()
    structure_members = set()
    
    for i in range(SIZE):
        patterns = []
        for r in range(start_r, end_r + 1):
            grid = history[r]
            pat = tuple([grid[i]] + [grid[nb] for nb in NEIGHBORS[i]])
            patterns.append(pat)
            
        most_common_pat, count = Counter(patterns).most_common(1)[0]
        if count / total_steps >= threshold:
            structure_centers.add(i)
            
    for center in structure_centers:
        structure_members.add(center)
        for nb in NEIGHBORS[center]:
            structure_members.add(nb)
            
    return structure_members

# 🌟 修改点：严格使用与折线图相对应的颜色的 RGBA 数值
def get_colored_grid(grid2d, motif_mask):
    img_data = np.zeros((N, N, 4))
    
    color_white = np.array([1.0, 1.0, 1.0, 1.0])
    color_black = np.array([0.0, 0.0, 0.0, 1.0])
    # 利用 to_rgba 将十六进制颜色精准转换为矩阵渲染所需的数值
    color_blue  = np.array(to_rgba('#2f6fe4'))
    color_red   = np.array(to_rgba('#d94b4b'))

    for r in range(N):
        for c in range(N):
            val = grid2d[r, c]
            is_in_motif = motif_mask[r, c]
            
            if is_in_motif:
                img_data[r, c] = color_black if val > 0.5 else color_white
            else:
                img_data[r, c] = color_blue if val > 0.5 else color_red
                
    return img_data

# ================= 4. 高质量出图引擎 =================
def plot_results(results_dict, output_path=None, *, save_data=False):
    """恢复原GridSpec样式，仅简化轴名并放大坐标文字。"""
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent
    output = Path(output_path) if output_path else output_dir / "2Diffusion_Evolution_DelayedSnapshots_Colored.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    w_values = (0.0, 0.8)
    if save_data:
        save_results_archive(results_dict, output_dir / "figure6_raw_results.npz", {
            "data_kind": "original_full_simulation_ratios_and_selected_colored_snapshots",
            "N": N, "C": C, "L": L, "alpha": ALPHA, "gamma": GAMMA,
            "epsilon": EPSILON, "rounds": ROUNDS, "random_seed": 43,
            "w_values": w_values,
            "snapshot_indices": ((100, 300, 500, 700, 1000), (1500, 1550, 1600, 1650, 2000)),
        })

    # 沿用当前GridSpec样式，仅将六列改为等宽；保留原字号、配色和横向间隔。
    # 画布直接按论文6.69英寸宽生成，放大坐标文字后适配必要的边距和行距。
    width, height = 6.69, 2.68
    tick_fontsize, axis_fontsize = 8.5, 9.5
    style = {
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42, "path.simplify": False,
        "xtick.direction": "in", "ytick.direction": "in", "axes.linewidth": 0.5,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width, height), facecolor="white")
        gs = gridspec.GridSpec(
            2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1],
            wspace=0.12, hspace=0.32,
            left=0.55 / width, right=0.995,
            bottom=0.33 / height, top=1 - 0.10 / height,
        )
        for row, w in enumerate(w_values):
            result = results_dict[w]
            ratio = np.asarray(result["ratio"])
            mcs = np.asarray(result.get("mcs", np.arange(len(ratio))))
            ax_line = fig.add_subplot(gs[row, 0])
            ax_line.plot(mcs, ratio, color=("#e74c3c", "#3498db")[row], lw=0.6)
            ax_line.set_xlim(0, ROUNDS)
            ax_line.set_ylim(-5, 105)
            ax_line.set_xlabel("MCS", fontsize=axis_fontsize, labelpad=1)
            ax_line.set_ylabel("Stable fraction(%)", fontsize=axis_fontsize, labelpad=1)
            ax_line.xaxis.set_major_locator(MultipleLocator(4000))
            ax_line.xaxis.set_minor_locator(MultipleLocator(500))
            ax_line.yaxis.set_major_locator(MultipleLocator(20))
            ax_line.yaxis.set_minor_locator(MultipleLocator(10))
            ax_line.tick_params(which="major", labelsize=tick_fontsize,
                                length=2.5, width=0.5, pad=1)
            ax_line.tick_params(which="minor", length=1.2, width=0.5)
            bounds = ax_line.get_position()
            fig.text(0.018 / width, bounds.y1 + 0.015 / height,
                     f"({chr(97 + row)})", fontsize=9.5, fontweight="bold",
                     ha="left", va="top")
            if len(result["snapshots"]) != 5:
                raise ValueError("Figure 6 requires five snapshots per row")
            for col, snapshot in enumerate(result["snapshots"]):
                ax_snap = fig.add_subplot(gs[row, col + 1])
                ax_snap.imshow(snapshot, interpolation="none", origin="upper", aspect="equal")
                ax_snap.set_xticks([])
                ax_snap.set_yticks([])
                for spine in ax_snap.spines.values():
                    spine.set_color("black")
                    spine.set_linewidth(0.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
        boxes += [text.get_window_extent(renderer) for text in fig.texts]
        for box in boxes:
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                plt.close(fig)
                raise ValueError(f"Figure 6 text outside the canvas: {box}")
        fig.savefig(output, format="pdf", dpi=600, facecolor="white")
        fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
        plt.close(fig)
    print(f"Figure saved: {output}")


def plot_final_figure():
    # 演化逻辑与快照索引不变；只把数据准备与可视化拆开，以便复用已有结果。
    print("Running W=0.0 simulation...")
    hist_w0, ratio_w0 = run_simulation(w_val=0.0)
    print("Running W=0.8 simulation...")
    hist_w08, ratio_w08 = run_simulation(w_val=0.8)
    results = {}
    for w, history, ratio, indices in [
        (0.0, hist_w0, ratio_w0, (100, 300, 500, 700, 1000)),
        (0.8, hist_w08, ratio_w08, (1500, 1550, 1600, 1650, 2000)),
    ]:
        snapshots = []
        for draw_r in indices:
            grid = history[draw_r - 1].reshape(N, N)
            mask = get_motif_mask(grid, MOTIF_A) if w == 0.0 else (
                get_motif_mask(grid, MOTIF_B1) | get_motif_mask(grid, MOTIF_B2))
            snapshots.append(get_colored_grid(grid, mask))
        results[w] = {"ratio": ratio, "snapshots": snapshots}
    plot_results(results, save_data=True)

# ================= 5. 启动入口 =================
def save_results_archive(results, output, provenance):
    output = Path(output)
    arrays = {"w_values": np.array([0.0,0.8]), "snapshot_mcs": np.array([[100,300,500,700,1000],[1500,1550,1600,1650,2000]])}
    for row, w in enumerate([0.0,0.8]):
        result = results[w]
        arrays[f"mcs_{row}"] = np.asarray(result.get("mcs", np.arange(len(result["ratio"]))))
        arrays[f"ratio_{row}"] = np.asarray(result["ratio"])
        arrays[f"snapshots_{row}"] = np.stack(result["snapshots"])
    arrays["provenance_json"] = np.array(json.dumps(provenance, ensure_ascii=False))
    np.savez_compressed(output, **arrays)
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    plot_final_figure()
