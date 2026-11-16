# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Xinyi Zhu and contributors
"""Draw SPQ terminal configurations from the saved Figure8 experiments."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'figure8')
    parser.add_argument('--cost', type=float, default=0.5)
    parser.add_argument('--run', type=int, default=0,
                        help='Fixed repetition index, selected before inspecting outcomes')
    parser.add_argument('--network-seed', type=int, default=2)
    parser.add_argument('--layout-seed', type=int, default=2)
    parser.add_argument('--output', type=Path, default=ROOT / 'new_results')
    args = parser.parse_args()
    names = ['WS', 'BA', 'ER']
    with (args.data / 'runs.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [row for row in csv.DictReader(f)
                if row['algorithm'] == 'SPQ'
                and int(row['run']) == args.run
                and int(row['network_seed']) == args.network_seed
                and np.isclose(float(row['cost']), args.cost, rtol=0, atol=1e-12)]
    if len(rows) != 3 or {row['network'] for row in rows} != set(names):
        parser.error('Exactly one saved SPQ run per topology is required for the requested parameters')
    keys = {name: f'{name}_g{args.network_seed}_c{args.cost:g}_SPQ_r{args.run}' for name in names}
    states = {}
    with gzip.open(args.data / 'final_states.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['record_id'] in keys.values():
                states[record['record_id']] = np.asarray(record['final_x'], dtype=np.uint8)
    if set(states) != set(keys.values()):
        parser.error('One or more requested final configurations are missing')
    graphs = {}
    with zipfile.ZipFile(args.data / 'networks.zip') as archive:
        for name in names:
            filename = f'{name}_seed{args.network_seed}.graphml'
            graphs[name] = nx.read_graphml(io.BytesIO(archive.read(filename)), node_type=int)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    exported = {}
    for panel, name in enumerate(names):
        graph = graphs[name]
        state = states[keys[name]]
        assert len(state) == graph.number_of_nodes() and np.isin(state, [0, 1]).all()
        nodes = sorted(graph.nodes())
        assert nodes == list(range(len(state)))
        positions = nx.spring_layout(graph, seed=args.layout_seed,
                                     k=2.0 / np.sqrt(len(state)), iterations=300)
        nx.draw_networkx_edges(graph, positions, ax=axes[panel],
                               edge_color='#4a4a4a', width=0.65, alpha=0.55)
        nx.draw_networkx_nodes(graph, positions, ax=axes[panel], nodelist=nodes,
                               node_color=['#2f6fe4' if state[i] else '#d94b4b' for i in nodes],
                               node_size=48, edgecolors='white', linewidths=0.35)
        axes[panel].set_title(f'({chr(97 + panel)}) {name}', fontsize=12)
        axes[panel].set_axis_off()
        row = next(row for row in rows if row['network'] == name)
        payoff = np.mean([-args.cost if state[i] else
                          (0.0 if any(state[j] for j in graph.neighbors(i)) else -1.0)
                          for i in nodes])
        # Stored pay is a last-100-MCS mean; terminal payoff can differ.
        exported[name] = {'record_id': keys[name], 'final_x': state.tolist(),
                          'positions': {str(i): positions[i].tolist() for i in nodes},
                          'terminal_investing_fraction': float(state.mean()),
                          'last_window_mean_investing_fraction': float(row['inv']),
                          'terminal_payoff': float(payoff),
                          'last_window_mean_payoff': float(row['pay'])}
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / 'figure9_new.pdf', facecolor='white')
    fig.savefig(args.output / 'figure9_new.png', dpi=220, facecolor='white')
    plt.close(fig)
    (args.output / 'selected_states.json').write_text(json.dumps({
        'cost': args.cost, 'run': args.run, 'network_seed': args.network_seed,
        'layout_seed': args.layout_seed, 'selection': 'fixed run index, not best-performing run',
        'states': exported}, indent=2), encoding='utf-8')
    print('Saved new snapshots from existing Figure8 states:', args.output)


if __name__ == '__main__':
    main()
