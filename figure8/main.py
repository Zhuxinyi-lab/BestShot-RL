# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
"""Figure 8 reproduction using Darvariu et al.'s original UCT implementation."""
import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "uct_source.zip"))
from relnet.agent.mcts.mcts_agent import MonteCarloTreeSearchAgent
from relnet.environment.graph_mis_env import GraphMISEnv
from relnet.objective_functions.objective_functions import SocialWelfare
from relnet.state.graph_state import S2VGraph

N = 100
LOSS = 1.0
ITERATIONS = 8000
STEADY_STATE_LAST_STEPS = 100
ALPHA, GAMMA, EPSILON, W_WEIGHT, FERMI_K = 0.1, 0.95, 0.005, 0.8, 0.1

def get_state(nb_inv, degree):
    nb_non_inv = degree - nb_inv
    if nb_inv > nb_non_inv:
        return 0
    if nb_inv == nb_non_inv:
        return 1
    return 2

def evaluate_user_payoff(graph, x_or_investors, c):
    if isinstance(x_or_investors, np.ndarray):
        x = x_or_investors.astype(int)
    else:
        x = np.zeros(graph.number_of_nodes(), dtype=int)
        x[list(x_or_investors)] = 1

    total_payoff = 0.0
    attacked_count = 0
    for i in range(graph.number_of_nodes()):
        if x[i] == 1:
            total_payoff -= c
        else:
            has_investing_neighbor = any(x[j] == 1 for j in graph.neighbors(i))
            if not has_investing_neighbor:
                total_payoff -= LOSS
                attacked_count += 1

    return {
        "pay": float(total_payoff / graph.number_of_nodes()),
        "inv": float(np.mean(x)),
        "investor_count": int(np.sum(x)),
        "attacked_count": int(attacked_count),
    }

def compute_average_payoff_from_counts(x, nb_counts, c):
    total_payoff = 0.0
    for i in range(len(x)):
        if x[i] > 0.5:
            total_payoff -= c
        elif nb_counts[i] == 0:
            total_payoff -= LOSS
    return float(total_payoff / len(x))

def run_dynamics_once(graph, c, algo_name, run_idx):
    rng_seed = 42 + run_idx
    rng = np.random.default_rng(rng_seed)

    neighbors = [list(graph.neighbors(i)) for i in range(N)]
    degrees = [len(neighbors[i]) for i in range(N)]

    x = rng.choice([0.0, 1.0], size=N)
    nb_counts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        nb_counts[i] = sum(1 for nb in neighbors[i] if x[nb] > 0.5)

    q_table = np.zeros((N, 3, 2), dtype=np.float64)
    last_inv = []
    last_pay = []

    for t in range(ITERATIONS):
        for _ in range(N):
            i = int(rng.integers(N))
            old_action = x[i]
            new_action = old_action

            if algo_name == "Fermi":
                pay_i = -c if old_action > 0.5 else (0.0 if nb_counts[i] > 0 else -LOSS)
                j = rng.choice(neighbors[i]) if neighbors[i] else i
                pay_j = -c if x[j] > 0.5 else (0.0 if nb_counts[j] > 0 else -LOSS)
                prob = 1.0 / (1.0 + np.exp((pay_i - pay_j) / FERMI_K))
                if rng.random() < prob:
                    new_action = x[j]

            elif algo_name == "BR":
                u_s = -c
                u_n = 0.0 if nb_counts[i] > 0 else -LOSS
                new_action = float(rng.integers(2)) if u_s == u_n else float(u_s > u_n)

            else:
                s_curr = get_state(nb_counts[i], degrees[i])
                if rng.random() < EPSILON:
                    a_idx = rng.integers(2)
                else:
                    values = q_table[i, s_curr]
                    a_idx = int(rng.choice(np.flatnonzero(values == values.max())))
                new_action = float(a_idx)

            if new_action != old_action:
                x[i] = new_action
                diff = 1 if new_action > 0.5 else -1
                for nb in neighbors[i]:
                    nb_counts[nb] += diff

            if algo_name in ("SPQ", "IPQ"):
                pay_i = -c if x[i] > 0.5 else (0.0 if nb_counts[i] > 0 else -LOSS)
                if algo_name == "SPQ":
                    nb_pay_sum = 0.0
                    for nb in neighbors[i]:
                        nb_pay_sum += -c if x[nb] > 0.5 else (0.0 if nb_counts[nb] > 0 else -LOSS)
                    reward = (1 - W_WEIGHT) * pay_i + W_WEIGHT * (nb_pay_sum / degrees[i] if degrees[i] else 0.0)
                else:
                    reward = pay_i

                s_next = get_state(nb_counts[i], degrees[i])
                a_idx_new = 1 if new_action > 0.5 else 0
                target = reward + GAMMA * np.max(q_table[i, s_next])
                q_table[i, s_curr, a_idx_new] += ALPHA * (
                    target - q_table[i, s_curr, a_idx_new]
                )

        if t >= ITERATIONS - STEADY_STATE_LAST_STEPS:
            last_inv.append(float(np.mean(x)))
            last_pay.append(compute_average_payoff_from_counts(x, nb_counts, c))

    return {
        "pay": float(np.mean(last_pay)),
        "inv": float(np.mean(last_inv)),
        "final_x": x.copy(),
    }

def uct_solution(graph, cost, seed, budget=20):
    state = S2VGraph(graph)
    state.effort_costs = np.full(len(graph), cost, dtype=np.float32)
    env = GraphMISEnv(SocialWelfare(), {}, heterogenous_cost=False)
    env.setup([state], [0.0], training=False)
    agent = MonteCarloTreeSearchAgent(env)
    params = agent.get_default_hyperparameters()
    params["expansion_budget_modifier"] = budget
    agent.setup({"random_seed": seed, "num_simulations": 1,
                 "draw_trees": False, "log_progress": False}, params)
    actions, welfare = agent.run_trajectory_collection()
    investors = sorted(map(int, env.g_list[0].selected_nodes))
    assert all(not graph.has_edge(i, j) for i in investors for j in investors)
    assert set(investors).union(*(set(graph.neighbors(i)) for i in investors)) == set(graph)
    result = evaluate_user_payoff(graph, investors, cost)
    # Original paper's benefit-one utility equals our loss-one payoff plus one.
    assert abs(welfare - (1 + result["pay"])) < 1e-6
    assert result["attacked_count"] == 0
    return result, investors, list(map(int, actions))



import pandas as pd
def export_results(directory):
    directory = Path(directory)
    data = pd.read_csv(directory / "runs.csv")
    summary = data.groupby(["network", "cost", "algorithm"])[["pay", "inv"]].agg(["mean", "std", "count"])
    summary.to_csv(directory / "summary.csv")
    flat = summary.copy()
    flat.columns = [f"{metric}_{statistic}" for metric, statistic in flat.columns]
    flat = flat.reset_index()
    statistics = ["pay_mean", "pay_std", "inv_mean", "inv_std", "pay_count"]
    for network, frame in flat.groupby("network"):
        wide = frame.pivot(index="cost", columns="algorithm", values=statistics)
        wide.columns = [f"{algorithm}_{statistic}" for statistic, algorithm in wide.columns]
        columns = [f"{algorithm}_{statistic}"
                   for algorithm in ["SPQ", "IPQ", "BR", "Fermi", "UCT"]
                   for statistic in statistics
                   if f"{algorithm}_{statistic}" in wide.columns]
        wide = wide[columns].sort_index().rename_axis("c").reset_index()
        wide.to_csv(directory / f"{network}_origin.csv", index=False, encoding="utf-8-sig")

def plot_results(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    RESULTS = Path(directory)
    data = pd.read_csv(RESULTS / "runs.csv")
    keys = ["network", "network_seed", "cost", "algorithm", "run"]
    if data.duplicated(keys).any() or not np.isfinite(data[["pay", "inv"]]).all().all():
        raise ValueError("Duplicate runs or nonfinite results")
    counts = data.groupby(["network", "cost", "algorithm"]).size()
    if len(data) != 1650 or len(counts) != 165 or not counts.eq(10).all():
        raise ValueError("Expected 3 networks x 11 costs x 5 algorithms x 10 runs")
    means = data.groupby(["network", "cost", "algorithm"])[["pay", "inv"]].mean()
    styles = {
        "SPQ": ("#e6554f", "o", "SPQ"),
        "IPQ": ("#f1a340", "^", "IPQ"),
        "BR": ("#5185b5", "s", "BR"),
        "Fermi": ("#68bbb5", "D", "FR"),
        "UCT": ("#ae81a7", "*", "UCT"),
    }
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "axes.labelsize": 12, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.2), layout="constrained")
    titles = {"WS": "(a) Watts–Strogatz (WS)", "BA": "(b) Barabási–Albert (BA)",
              "ER": "(c) Erdős–Rényi (ER)"}
    for col, network in enumerate(["WS", "BA", "ER"]):
        axes[0, col].set_title(titles[network], fontsize=13, pad=10)
        for row, metric in enumerate(["pay", "inv"]):
            ax = axes[row, col]
            for algorithm, (color, marker, label) in styles.items():
                values = means.xs((network, algorithm), level=("network", "algorithm")).sort_index()
                ax.plot(values.index, values[metric], color=color, marker=marker,
                        markersize=4.8 if algorithm != "UCT" else 7,
                        linewidth=1.25, label=label)
            ax.set_xlabel("Investment cost, $c$")
            ax.set_ylabel("Average social payoff" if row == 0 else "Proportion of investing agents")
            ax.set_xlim(-.025, 1.015)
            ax.set_xticks(np.arange(0, 1.01, .2))
            ax.set_ylim((-1.05, .05) if row == 0 else (-.025, .75))
            ax.tick_params(direction="in", top=True, right=True)
            ax.grid(alpha=.16, linewidth=.55)
            ax.legend(loc="lower left" if row == 0 else "upper right", ncol=2,
                      fontsize=9.5, framealpha=.9, columnspacing=1.0, handlelength=1.7)
    fig.savefig(RESULTS / "figure8_results.png", dpi=200)
    fig.savefig(RESULTS / "figure8_results.svg")
    plt.close(fig)
    print("Validated 1650 runs; saved figure8_results.png and figure8_results.svg")

def main():
    global N, ITERATIONS, STEADY_STATE_LAST_STEPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Plot the supplied runs.csv without running experiments")
    parser.add_argument("--input", type=Path, default=ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--uct-only", action="store_true")
    parser.add_argument("--costs", nargs="+", type=float,
                        default=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
    parser.add_argument("--network-seeds", nargs="+", type=int, default=[2])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.input)
        return
    if args.runs < 1 or args.steps < 1 or any(not 0 <= c < 1 for c in args.costs):
        parser.error("runs/steps must be positive and costs must satisfy 0 <= c < 1")
    N = 12 if args.smoke else 100
    ITERATIONS = 20 if args.smoke else args.steps
    STEADY_STATE_LAST_STEPS = min(100, ITERATIONS)
    runs = 2 if args.smoke else args.runs
    budget = 2 if args.smoke else 20
    output = args.output or ROOT / ("smoke_results" if args.smoke else "results")
    if output.exists() and any(output.iterdir()):
        parser.error("output must be empty to prevent mixing separate experiments")
    output.mkdir(parents=True, exist_ok=True)
    commit = "30428dec0783aa71a8ff955862e95ae5b3cca9b4"
    if (ROOT / "upstream" / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    metadata = {"upstream_commit": commit, "python": platform.python_version(),
                "numpy": np.__version__, "networkx": nx.__version__,
                "arguments": {**vars(args), "input": str(args.input), "output": str(output)},
                "n": N, "steps": ITERATIONS, "steady_window_mcs": STEADY_STATE_LAST_STEPS,
                "uct": {**MonteCarloTreeSearchAgent.get_default_hyperparameters(),
                        "expansion_budget_modifier": budget},
                "network_parameters": {"WS": {"k": 4, "p": .1},
                                       "BA": {"m": 2}, "ER": {"p": .2}},
                "learning": {"alpha": ALPHA, "gamma": GAMMA, "epsilon": EPSILON,
                             "w_spq": W_WEIGHT, "w_ipq": 0, "fermi_k": FERMI_K},
                "update_rule": "N uniformly random node selections with replacement per MCS",
                "seed_rule": "network seed fixed before results; algorithm seed 42 + run"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    algorithms = ["UCT"] if args.uct_only else ["UCT", "Fermi", "BR", "SPQ", "IPQ"]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["network", "network_seed", "cost", "algorithm", "run", "seed", "pay", "inv"])
        writer.writeheader()
        for network_seed in args.network_seeds:
            graphs = {"WS": nx.watts_strogatz_graph(N, 4, .1, seed=network_seed),
                      "BA": nx.barabasi_albert_graph(N, 2, seed=network_seed),
                      "ER": nx.erdos_renyi_graph(N, .2, seed=network_seed)}
            for name, graph in graphs.items():
                nx.write_graphml(graph, output / f"{name}_seed{network_seed}.graphml")
                for cost in args.costs:
                    for algorithm in algorithms:
                        for run in range(runs):
                            seed = 42 + run
                            if algorithm == "UCT":
                                result, investors, actions = uct_solution(graph, cost, seed, budget)
                                final_x = np.isin(np.arange(N), investors).astype(int)
                                extra = {"investors": investors, "selected_actions": actions}
                            else:
                                result = run_dynamics_once(graph, cost, algorithm, run)
                                final_x = result["final_x"]
                                extra = {}
                            stem = f"{name}_g{network_seed}_c{cost:g}_{algorithm}_r{run}"
                            (output / (stem + ".json")).write_text(json.dumps({
                                "final_x": final_x.astype(int).tolist(), **extra}), encoding="utf-8")
                            writer.writerow({"network": name, "network_seed": network_seed,
                                "cost": cost, "algorithm": algorithm, "run": run, "seed": seed,
                                "pay": result["pay"], "inv": result["inv"]})
                            file.flush()
                        print(name, cost, algorithm, "complete", flush=True)
    export_results(output)


if __name__ == "__main__":
    main()
