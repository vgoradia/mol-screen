"""
Loads tox21.csv, converts every molecule to a graph, and builds the
full dataset list ready for train/val/test splitting.

Missing labels (NaN) per task are masked out during loss computation,
not imputed, since these are genuinely missing assay results, not zeros.
"""

import pandas as pd
import torch
import pickle
from featurize import smiles_to_graph

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
          'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
          'SR-MMP', 'SR-p53']


def build_dataset(csv_path='tox21.csv'):
    df = pd.read_csv(csv_path)

    graphs = []
    failed = 0

    for idx, row in df.iterrows():
        smiles = row['smiles']
        labels = row[TASKS].values.astype(float)  # NaN preserved for missing
        label_tensor = torch.tensor(labels, dtype=torch.float)

        g = smiles_to_graph(smiles, label=label_tensor)
        if g is None:
            failed += 1
            continue
        g.mol_id = row['mol_id']
        graphs.append(g)

    print(f"Total molecules in CSV: {len(df)}")
    print(f"Successfully converted: {len(graphs)}")
    print(f"Failed to parse: {failed}")

    return graphs


if __name__ == "__main__":
    graphs = build_dataset()

    with open('tox21_graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)

    print(f"Saved {len(graphs)} graphs to tox21_graphs.pkl")

    # sanity check a few
    print("\nSample graph:")
    g = graphs[0]
    print(f"  SMILES: {g.smiles}")
    print(f"  Nodes: {g.x.shape[0]}, Node feat dim: {g.x.shape[1]}")
    print(f"  Edges: {g.edge_index.shape[1]}")
    print(f"  Labels: {g.y}")
