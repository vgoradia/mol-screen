"""
MolScreen featurization: converts SMILES strings into molecular graphs
(PyTorch Geometric Data objects) for use with a GNN.

Atom features (per node):
  - atom type (one-hot, common organic elements)
  - degree (number of bonds)
  - formal charge
  - number of hydrogens
  - hybridization (one-hot)
  - aromaticity (bool)
  - in-ring (bool)

Bond features (per edge):
  - bond type (one-hot: single/double/triple/aromatic)
  - conjugated (bool)
  - in-ring (bool)
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
             'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
             'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
             'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(value, allowed_set):
    """One-hot encode a value against an allowed set; unseen values map to last bucket."""
    if value not in allowed_set:
        value = allowed_set[-1]
    return [1.0 if value == s else 0.0 for s in allowed_set]


def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetSymbol(), ATOM_LIST)
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    feats += [atom.GetFormalCharge()]
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)
    feats += [1.0 if atom.GetIsAromatic() else 0.0]
    feats += [1.0 if atom.IsInRing() else 0.0]
    return feats


def bond_features(bond):
    feats = []
    feats += one_hot(bond.GetBondType(), BOND_TYPE_LIST)
    feats += [1.0 if bond.GetIsConjugated() else 0.0]
    feats += [1.0 if bond.IsInRing() else 0.0]
    return feats


def smiles_to_graph(smiles, label=None):
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    Returns None if the SMILES is invalid (so callers can skip/log it).
    label: optional tensor of shape (num_tasks,) with NaN for missing labels.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edges (undirected -> both directions) + edge features
    edge_indices = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_feats += [feat, feat]

    if len(edge_indices) == 0:
        # Single-atom molecule edge case: no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(bond_features.__defaults__ or []) or 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = label.unsqueeze(0)  # shape (1, num_tasks)
    data.smiles = smiles
    return data


def node_feature_dim():
    """Returns the dimensionality of the atom feature vector (for model init)."""
    return len(ATOM_LIST) + 7 + 1 + 5 + len(HYBRIDIZATION_LIST) + 1 + 1


def edge_feature_dim():
    """Returns the dimensionality of the bond feature vector (for model init)."""
    return len(BOND_TYPE_LIST) + 1 + 1


if __name__ == "__main__":
    # Quick smoke test
    test_smiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    g = smiles_to_graph(test_smiles)
    print(f"SMILES: {test_smiles}")
    print(f"Node feature shape: {g.x.shape}")
    print(f"Edge index shape: {g.edge_index.shape}")
    print(f"Edge attr shape: {g.edge_attr.shape}")
    print(f"Expected node dim: {node_feature_dim()}, actual: {g.x.shape[1]}")
    print(f"Expected edge dim: {edge_feature_dim()}, actual: {g.edge_attr.shape[1]}")
