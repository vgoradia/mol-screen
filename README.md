# MolScreen

**AI early-stage molecular toxicity screening with substructure-level explainability and LLM-generated research triage notes.**

MolScreen is a graph neural network (GNN) trained on 7,823 compounds from the NIH/EPA Tox21 dataset, predicting toxicity risk across 12 biological endpoints directly from molecular structure. Across 10 independent training runs, MolScreen achieves a mean test ROC-AUC of 0.8442 under random splitting and 0.7716 under Bemis-Murcko scaffold splitting, outperforming the best published baseline on this dataset (Chen et al., 2021 SSL-GCN: 0.757) under the same scaffold split protocol. Beyond classification, MolScreen highlights which specific atoms drove each prediction and generates a plain-English research triage note to help screening chemists prioritize compounds for further testing.

**Live App:** https://mol-screen-32jrti5gpuajmd8wse5u3f.streamlit.app/

---

## Features

- **GNN toxicity prediction** across 12 Tox21 endpoints (NR-AR, NR-AhR, NR-ER, SR-MMP, and more)
- **Substructure-level explainability** — gradient-based atom attribution highlights which atoms the model focused on
- **LLM triage notes** — Claude API translates raw predictions into plain-English research recommendations
- **No ML expertise required** — paste a SMILES string, get results in under 0.5 seconds

---

## Model

| Metric | Value |
|---|---|
| Architecture | 4-layer GINEConv GNN with residual connections |
| Training data | 6,258 molecules (80% of Tox21) |
| Test mean ROC-AUC | 0.8442 |
| Parameters | ~374,000 |
| Inference time | ~0.4 seconds end-to-end |
| Comparison baseline | Chen et al. (2021) SSL-GCN: 0.757 mean ROC-AUC |

---

## Dataset

[Tox21](https://tripod.nih.gov/tox21/challenge/) — NIH/EPA Toxicology in the 21st Century initiative. 7,823 compounds with binary labels across 12 nuclear receptor and stress response toxicity assays. Heavy class imbalance handled via masked multi-label binary cross-entropy loss (missing labels are masked out, not imputed as negative).

---

## Installation

```bash
git clone https://github.com/vgoradia/mol-screen.git
cd mol-screen
pip install streamlit anthropic torch torch_geometric rdkit
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Run the app:
```bash
streamlit run app.py
```

---

## Project Structure

```
mol-screen/
├── app.py              # Streamlit web application
├── model.py            # GINEConv GNN architecture
├── featurize.py        # SMILES → molecular graph featurization
├── train.py            # Training loop with early stopping
├── build_dataset.py    # Dataset preprocessing
├── explainability.py   # Gradient-based atom attribution
├── llm_triage.py       # LLM prompt construction and response handling
├── molscreen_best.pt   # Trained model checkpoint
└── tox21.csv           # Raw Tox21 dataset
```

---

## Usage

Paste any valid SMILES string into the app and click **Analyze Compound**. MolScreen returns:

1. Predicted toxicity probability across all 12 Tox21 endpoints
2. A molecular structure visualization with atoms colored by importance
3. A plain-English research triage note recommending whether to advance, modify, or deprioritize the compound

---

## Disclaimer

MolScreen is a computational screening aid, not a diagnostic or regulatory tool. Predictions should be used to help prioritize compounds for further laboratory testing and are not a replacement for experimental toxicology.

---

## Author

Veer Goradia 
GitHub: [vgoradia](https://github.com/vgoradia) | Hugging Face: [vgoradia](https://huggingface.co/vgoradia)
