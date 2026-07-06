"""
MolScreen explainability: identifies which atoms/substructures in a
molecule drove a given toxicity prediction.

Method: gradient-based saliency on node features, analogous to PyroSight's
input-gradient saliency maps but adapted to graphs. For a given task t,
we compute d(logit_t)/d(node_features) for each atom, take the L2 norm
per atom as its importance score, then normalize to [0,1] for visualization.

This directly addresses a stated limitation in the comparable published
SSL-GCN paper (Chen et al., J Cheminform 2021), which explicitly notes:
"the interpretability of our graph convolution model has not been
explored... in the next step of our study, we will focus on the
interpretability of the graph convolutional neural network."
"""

import torch
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from model import MolScreenGNN

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
          'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
          'SR-MMP', 'SR-p53']


def compute_atom_importance(model, data, task_idx):
    """
    Computes per-atom importance scores for a single molecule's
    prediction on a specific task, via input-gradient saliency.

    Returns: numpy array of shape (num_atoms,), normalized to [0, 1].
    """
    model.eval()
    x = data.x.clone().detach().requires_grad_(True)
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    batch = torch.zeros(x.shape[0], dtype=torch.long)  # single graph

    logits = model(x, edge_index, edge_attr, batch)
    target_logit = logits[0, task_idx]

    model.zero_grad()
    target_logit.backward()

    # L2 norm of gradient per atom = importance
    grad = x.grad  # shape (num_atoms, feat_dim)
    importance = grad.norm(dim=1).detach().numpy()

    # normalize to [0, 1] for visualization
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min())
    else:
        importance = np.zeros_like(importance)

    confidence = torch.sigmoid(target_logit).item()
    return importance, confidence


def render_molecule_with_importance(smiles, importance, out_path):
    """
    Draws the molecule with atoms colored by importance score
    (red = high importance, matching PyroSight's saliency color scheme).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_colors = {}
    for i, score in enumerate(importance):
        # red intensity scales with importance; low importance = light gray
        r = 1.0
        g = 1.0 - float(score) * 0.85
        b = 1.0 - float(score) * 0.85
        atom_colors[i] = (r, g, b)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=list(range(len(importance))),
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    with open(out_path, 'wb') as f:
        f.write(drawer.GetDrawingText())
    return out_path


def get_top_contributing_atoms(mol, importance, top_k=3):
    """Returns the top-k most important atoms with their element symbol and index."""
    idx_sorted = np.argsort(importance)[::-1][:top_k]
    results = []
    for idx in idx_sorted:
        atom = mol.GetAtomWithIdx(int(idx))
        results.append({
            'index': int(idx),
            'element': atom.GetSymbol(),
            'aromatic': atom.GetIsAromatic(),
            'in_ring': atom.IsInRing(),
            'importance': float(importance[idx]),
        })
    return results


if __name__ == "__main__":
    from featurize import smiles_to_graph

    model = MolScreenGNN()
    model.load_state_dict(torch.load('molscreen_best.pt'))

    # test on a real Tox21 molecule
    with open('tox21_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    test_graph = graphs[5]
    smiles = test_graph.smiles
    mol = Chem.MolFromSmiles(smiles)

    task_idx = 2  # NR-AhR
    importance, confidence = compute_atom_importance(model, test_graph, task_idx)

    print(f"SMILES: {smiles}")
    print(f"Task: {TASKS[task_idx]}")
    print(f"Predicted probability of toxicity: {confidence:.3f}")
    print(f"Number of atoms: {len(importance)}")
    print(f"Importance scores: {importance}")

    top_atoms = get_top_contributing_atoms(mol, importance, top_k=3)
    print("\nTop contributing atoms:")
    for a in top_atoms:
        print(f"  Atom {a['index']} ({a['element']}, aromatic={a['aromatic']}, "
              f"ring={a['in_ring']}): importance={a['importance']:.3f}")

    render_molecule_with_importance(smiles, importance, 'sample_explainability.png')
    print("\nSaved visualization to sample_explainability.png")
