"""
MolScreen LLM decision support layer.

Takes the GNN's raw multi-task toxicity predictions plus the
explainability output (top contributing atoms/substructures) and
generates a plain-English research triage note for a screening
chemist -- mirroring PyroSight's emergency-responder decision
support layer, but for early-stage compound screening.

Like PyroSight's LLM layer, this is NOT trained or fine-tuned. It is
a prompted API call applied to the model's structured output. The
research contribution is the GNN + explainability; this layer's
contribution is purely to system usability / accessibility, framed
explicitly as such in the paper (same framing used for PyroSight,
per Dr. Akhloufi's review feedback).
"""

import json

TASK_DESCRIPTIONS = {
    'NR-AR': 'androgen receptor signaling',
    'NR-AR-LBD': 'androgen receptor ligand-binding domain',
    'NR-AhR': 'aryl hydrocarbon receptor signaling',
    'NR-Aromatase': 'aromatase enzyme activity',
    'NR-ER': 'estrogen receptor signaling',
    'NR-ER-LBD': 'estrogen receptor ligand-binding domain',
    'NR-PPAR-gamma': 'PPAR-gamma nuclear receptor signaling',
    'SR-ARE': 'antioxidant response element stress pathway',
    'SR-ATAD5': 'DNA damage / genotoxic stress response',
    'SR-HSE': 'heat shock stress response',
    'SR-MMP': 'mitochondrial membrane potential disruption',
    'SR-p53': 'p53 tumor suppressor stress pathway',
}


def build_triage_prompt(smiles, predictions, top_atoms_per_flagged_task, threshold=0.5):
    """
    predictions: dict of {task_name: probability}
    top_atoms_per_flagged_task: dict of {task_name: [list of atom dicts]}
    """
    flagged = {t: p for t, p in predictions.items() if p >= threshold}
    flagged_sorted = sorted(flagged.items(), key=lambda x: -x[1])

    if not flagged_sorted:
        summary_lines = ["No toxicity pathways were flagged above the 0.5 confidence threshold."]
    else:
        summary_lines = []
        for task, prob in flagged_sorted:
            desc = TASK_DESCRIPTIONS.get(task, task)
            atoms = top_atoms_per_flagged_task.get(task, [])
            atom_str = ", ".join([f"{a['element']} (atom {a['index']})" for a in atoms[:3]])
            summary_lines.append(
                f"- {task} ({desc}): {prob:.0%} confidence. "
                f"Most influential atoms: {atom_str if atom_str else 'not computed'}."
            )

    prompt = f"""A molecular toxicity screening model has analyzed the following compound:

SMILES: {smiles}

Predicted toxicity flags (tasks at or above 50% confidence):
{chr(10).join(summary_lines)}

Write a 3-sentence research triage note for a medicinal chemist evaluating whether to advance this compound for further testing. Sentence 1: state which toxicity pathway(s), if any, were flagged and at what confidence. Sentence 2: explain in plain terms what structural feature(s) the model attributed this to, without overstating certainty (this is a computational screening flag, not a confirmed toxicological finding). Sentence 3: give a brief, actionable recommendation (e.g., deprioritize, consider structural modification of the flagged region, or proceed with standard caution if no pathways were flagged)."""

    return prompt


def mock_llm_response(smiles, predictions, top_atoms_per_flagged_task, threshold=0.5):
    """
    Placeholder for local testing without an API call.
    The real deployed app calls the Claude API with build_triage_prompt().
    This mock lets us verify the full pipeline logic end-to-end first.
    """
    flagged = {t: p for t, p in predictions.items() if p >= threshold}
    if not flagged:
        return ("No toxicity pathways were flagged above the screening threshold for this compound. "
                "The model did not identify structural features strongly associated with any of the "
                "12 assayed toxicity endpoints. Standard screening caution is still recommended before advancing to in vitro testing.")

    top_task = max(flagged, key=flagged.get)
    prob = flagged[top_task]
    atoms = top_atoms_per_flagged_task.get(top_task, [])
    atom_desc = atoms[0]['element'] if atoms else "an unspecified"

    return (f"This compound was flagged for {TASK_DESCRIPTIONS.get(top_task, top_task)} "
            f"with {prob:.0%} model confidence. The prediction was primarily attributed to a "
            f"{atom_desc}-containing substructure, though this is a computational screening "
            f"signal, not a confirmed toxicological finding. Recommend deprioritizing or "
            f"considering structural modification of the flagged region before advancing this "
            f"compound to in vitro testing.")


if __name__ == "__main__":
    import torch
    import pickle
    from model import MolScreenGNN
    from explainability import compute_atom_importance, get_top_contributing_atoms, TASKS
    from rdkit import Chem

    model = MolScreenGNN()
    model.load_state_dict(torch.load('molscreen_best.pt'))

    with open('tox21_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    test_graph = graphs[5]
    smiles = test_graph.smiles
    mol = Chem.MolFromSmiles(smiles)

    predictions = {}
    top_atoms = {}
    for i, task in enumerate(TASKS):
        importance, confidence = compute_atom_importance(model, test_graph, i)
        predictions[task] = confidence
        if confidence >= 0.5:
            top_atoms[task] = get_top_contributing_atoms(mol, importance, top_k=3)

    prompt = build_triage_prompt(smiles, predictions, top_atoms)
    print("=== GENERATED PROMPT ===")
    print(prompt)
    print()
    print("=== MOCK LLM RESPONSE (pipeline test) ===")
    print(mock_llm_response(smiles, predictions, top_atoms))
