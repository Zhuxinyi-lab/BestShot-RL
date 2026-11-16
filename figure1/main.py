# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
# One MCS: N*N uniform random node draws WITH replacement; updates apply immediately.
import os
from multiprocessing import Pool, cpu_count

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


# ================= 1. Configuration =================
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

N = 100  # 100 x 100 periodic lattice (10,000 agents)
SIZE = N * N
L = 1.0
ITERATIONS = 8000  # MCS
SAMPLE_INTERVAL = 1
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.005
W_WEIGHT = 0.8
FERMI_K = 0.1
NUM_RUNS = 5

C_VALUES = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_results")


def init_grid():
    G = nx.grid_2d_graph(N, N, periodic=True)
    node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
    neighbors = [[] for _ in range(SIZE)]
    for node in G.nodes():
        u = node_to_id[node]
        for v_node in G.neighbors(node):
            neighbors[u].append(node_to_id[v_node])
    return neighbors, np.array(neighbors)


NEIGHBORS, NEIGHBOR_ARR = init_grid()


def get_majority_state(nb_inv):
    if nb_inv > 2:
        return 0
    if nb_inv == 2:
        return 1
    return 2


def run_algorithm(args):
    c, algo_name, x_init = args

    x = x_init.copy()
    nb_counts = np.zeros(SIZE, dtype=np.int32)
    for i in range(SIZE):
        nb_counts[i] = sum(1 for nb in NEIGHBORS[i] if x[nb] > 0.5)

    payoff_lookup = np.array([
        [-L, 0.0],
        [-c, -c],
    ])

    q_table = np.zeros((SIZE, 3, 2))

    inv_history = []
    reward_history = []

    for t in range(ITERATIONS + 1):
        if t % SAMPLE_INTERVAL == 0:
            current_inv = np.mean(x)
            has_nb_inv = (nb_counts > 0).astype(int)
            self_inv = x.astype(int)
            base_payoffs = payoff_lookup[self_inv, has_nb_inv]

            if algo_name == "Q-Mixed":
                nb_payoffs_sum = np.sum(base_payoffs[NEIGHBOR_ARR], axis=1)
                current_reward = np.mean(
                    (1 - W_WEIGHT) * base_payoffs
                    + W_WEIGHT * (nb_payoffs_sum / 4.0)
                )
            else:
                current_reward = np.mean(base_payoffs)

            inv_history.append(current_inv)
            reward_history.append(current_reward)

        if t == ITERATIONS:
            break

        for _ in range(SIZE):
            i = np.random.randint(SIZE)
            old_action = x[i]
            new_action = old_action

            if algo_name == "Fermi":
                pay_i = payoff_lookup[int(old_action), 1 if nb_counts[i] > 0 else 0]
                j = np.random.choice(NEIGHBORS[i])
                pay_j = payoff_lookup[int(x[j]), 1 if nb_counts[j] > 0 else 0]

                prob = 1.0 / (1.0 + np.exp((pay_i - pay_j) / FERMI_K))
                if np.random.rand() < prob:
                    new_action = x[j]

            elif algo_name == "BR":
                pay_1 = payoff_lookup[1, 1 if nb_counts[i] > 0 else 0]
                pay_0 = payoff_lookup[0, 1 if nb_counts[i] > 0 else 0]
                new_action = 1.0 if pay_1 > pay_0 else 0.0

            else:
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

            if "Q" in algo_name:
                pay_i = payoff_lookup[int(x[i]), 1 if nb_counts[i] > 0 else 0]

                if algo_name == "Q-Mixed":
                    pi_nbs_sum = sum(
                        payoff_lookup[int(x[nb]), 1 if nb_counts[nb] > 0 else 0]
                        for nb in NEIGHBORS[i]
                    )
                    reward = (1 - W_WEIGHT) * pay_i + W_WEIGHT * (pi_nbs_sum / 4.0)
                else:
                    reward = pay_i

                s_next = get_majority_state(nb_counts[i])
                a_idx_new = 1 if new_action > 0.5 else 0
                target = reward + GAMMA * np.max(q_table[i, s_next])
                q_table[i, s_curr, a_idx_new] += ALPHA * (
                    target - q_table[i, s_curr, a_idx_new]
                )

    avg_inv = np.mean(inv_history[-100:])
    avg_reward = np.mean(reward_history[-100:])
    from pathlib import Path
    import hashlib
    import json
    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)
    key = hashlib.sha256(np.asarray(x_init, dtype=np.uint8).tobytes()).hexdigest()[:16]
    stem = f'{algo_name}_c{c}_initial{key}'
    np.savez_compressed(out / (stem + '_states.npz'),
                        initial=np.asarray(x_init, dtype=np.uint8).reshape(N,N),
                        final=x.astype(np.uint8).reshape(N,N))
    np.savetxt(out / (stem + '_history.csv'),
               np.column_stack((np.arange(len(inv_history))*SAMPLE_INTERVAL, reward_history, inv_history)),
               delimiter=',', header='MCS,average_social_payoff,investing_fraction', comments='')
    return algo_name, avg_inv, avg_reward


def process_single_c(c):
    algos = ["Q-Mixed", "Q-Indiv", "BR", "Fermi"]
    run_results = {algo: {"inv": [], "reward": []} for algo in algos}

    for run_idx in range(NUM_RUNS):
        np.random.seed(42 + run_idx)
        x_init = np.random.choice([0.0, 1.0], size=SIZE)
        tasks = [(c, algo, x_init) for algo in algos]

        for task in tasks:
            if task[1] == "BR":
                # Independent BR update stream, matching the supplied BR runs.
                np.random.seed(1000000 + 42 + run_idx)
            algo_name, avg_inv, avg_reward = run_algorithm(task)
            run_results[algo_name]["inv"].append(avg_inv)
            run_results[algo_name]["reward"].append(avg_reward)

    results = {}
    for algo in algos:
        results[algo] = {
            "inv": np.mean(run_results[algo]["inv"]),
            "reward": np.mean(run_results[algo]["reward"]),
        }

    return c, results


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_procs = max(1, cpu_count() - 2)
    print(f"Starting {num_procs} worker processes for evolution...\n")

    all_results = {}
    with Pool(processes=num_procs) as pool:
        for c, res_dict in tqdm(
            pool.imap_unordered(process_single_c, C_VALUES),
            total=len(C_VALUES),
            desc="Processing C values",
        ):
            all_results[c] = res_dict

    print("\nExperiment finished. Generating figure and CSV...")

    c_array = np.array(sorted(C_VALUES))

    plot_data = {
        "Q-Mixed": {
            "inv": [],
            "reward": [],
            "label": f"Q-learning(w={W_WEIGHT})",
            "color": "#f39c12",
            "marker": "^",
        },
        "Q-Indiv": {
            "inv": [],
            "reward": [],
            "label": "Q-learning(w=0.0)",
            "color": "#e74c3c",
            "marker": "s",
        },
        "BR": {
            "inv": [],
            "reward": [],
            "label": "BR",
            "color": "#5DADE2",
            "marker": "o",
        },
        "Fermi": {
            "inv": [],
            "reward": [],
            "label": "Fermi",
            "color": "#48C9B0",
            "marker": "D",
        },
    }

    for c in c_array:
        for algo in plot_data.keys():
            plot_data[algo]["inv"].append(all_results[c][algo]["inv"])
            plot_data[algo]["reward"].append(all_results[c][algo]["reward"])

    df_data = {"c": c_array}
    for algo, data in plot_data.items():
        df_data[f"{algo}_reward"] = data["reward"]
        df_data[f"{algo}_investment"] = data["inv"]

    csv_path = os.path.join(OUTPUT_DIR, "figure1_new_results.csv")
    pd.DataFrame(df_data).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved steady-state CSV: {csv_path}")

    bne_inv = np.full_like(c_array, 0.2)
    bne_reward = -0.2 * c_array
    wne_inv = np.full_like(c_array, 0.5)
    wne_reward = -0.5 * c_array

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for algo, data in plot_data.items():
        ax.plot(
            c_array,
            data["reward"],
            label=data["label"],
            color=data["color"],
            marker=data["marker"],
            linestyle="-",
            markersize=7,
        )

    ax.plot(c_array, bne_reward, label="BNE", color="black", linestyle="--", linewidth=2)
    ax.plot(c_array, wne_reward, label="WNE", color="grey", linestyle="--", linewidth=2)

    ax.set_xlabel("c", fontsize=16)
    ax.set_ylabel("average social reward", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12, loc="lower left")

    ax = axes[1]
    for algo, data in plot_data.items():
        ax.plot(
            c_array,
            data["inv"],
            label=data["label"],
            color=data["color"],
            marker=data["marker"],
            linestyle="-",
            markersize=7,
        )

    ax.plot(c_array, bne_inv, label="BNE", color="black", linestyle="--", linewidth=2)
    ax.plot(c_array, wne_inv, label="WNE", color="grey", linestyle="--", linewidth=2)

    ax.set_xlabel("c", fontsize=16)
    ax.set_ylabel("average investment", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    figure_path = os.path.join(OUTPUT_DIR, "figure1_new_results.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved figure: {figure_path}")
