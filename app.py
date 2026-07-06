"""
MolScreen: AI-Powered Molecular Toxicity Screening
Streamlit deployment app.

User pastes a SMILES string -> GNN predicts toxicity across 12 Tox21
endpoints -> explainability highlights contributing atoms -> LLM
generates a plain-English research triage note.

Mirrors PyroSight's deployment architecture: model inference,
visual explainability, and LLM decision support in a single pipeline,
accessible without any ML or cheminformatics expertise.
"""

import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import time
import os

from featurize import smiles_to_graph
from model import MolScreenGNN
from explainability import compute_atom_importance, get_top_contributing_atoms
from llm_triage import build_triage_prompt, TASK_DESCRIPTIONS

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
          'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
          'SR-MMP', 'SR-p53']

st.set_page_config(page_title="MolScreen", page_icon="\U0001F9EA", layout="wide")


@st.cache_resource
def load_model():
    model = MolScreenGNN()
    model.load_state_dict(torch.load('molscreen_best.pt', map_location='cpu'))
    model.eval()
    return model


def render_molecule_image(mol, importance=None):
    drawer = rdMolDraw2D.MolDraw2DCairo(450, 450)
    if importance is not None:
        atom_colors = {}
        for i, score in enumerate(importance):
            r = 1.0
            g = 1.0 - float(score) * 0.85
            b = 1.0 - float(score) * 0.85
            atom_colors[i] = (r, g, b)
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=list(range(len(importance))),
            highlightAtomColors=atom_colors,
        )
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def call_claude_api(prompt):
    """
    Calls the Claude API for the plain-English triage note.
    Requires ANTHROPIC_API_KEY to be set in the environment.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"(LLM summary unavailable: {e})"


st.title("\U0001F9EA MolScreen")
st.markdown("**AI-powered early-stage molecular toxicity screening**, with substructure-level "
            "explainability and plain-English research triage notes.")

with st.expander("About this tool"):
    st.markdown("""
    MolScreen is a graph neural network trained on the Tox21 dataset (7,823 compounds,
    12 toxicity assay endpoints from the NIH/EPA Toxicology in the 21st Century initiative).
    It predicts toxicity risk directly from molecular structure, highlights which atoms drove
    each prediction, and generates a plain-English triage note for screening researchers.

    **This is a computational screening aid, not a diagnostic or regulatory tool.** Predictions
    should be used to help prioritize compounds for further laboratory testing, not as a
    replacement for experimental toxicology.
    """)

smiles_input = st.text_input(
    "Enter a SMILES string",
    value="CC(=O)Oc1ccccc1C(=O)O",
    help="Example shown is aspirin. Paste any valid SMILES string."
)

threshold = st.slider("Toxicity flag confidence threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Analyze Compound", type="primary"):
    mol = Chem.MolFromSmiles(smiles_input)

    if mol is None:
        st.error("Invalid SMILES string. Please check the input and try again.")
    else:
        start_time = time.time()
        model = load_model()
        graph = smiles_to_graph(smiles_input)

        predictions = {}
        all_importance = {}
        for i, task in enumerate(TASKS):
            importance, confidence = compute_atom_importance(model, graph, i)
            predictions[task] = confidence
            all_importance[task] = importance

        elapsed = time.time() - start_time

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Molecule Structure")
            img_bytes = render_molecule_image(mol)
            st.image(img_bytes, use_container_width=True)
            st.caption(f"Inference completed in {elapsed:.3f} seconds")

        with col2:
            st.subheader("Toxicity Predictions (12 Tox21 Endpoints)")
            flagged_tasks = {t: p for t, p in predictions.items() if p >= threshold}

            for task in TASKS:
                prob = predictions[task]
                desc = TASK_DESCRIPTIONS[task]
                color = "\U0001F534" if prob >= threshold else "\U0001F7E2"
                st.markdown(f"{color} **{task}** ({desc}): {prob:.1%}")

        st.divider()

        if flagged_tasks:
            top_flagged_task = max(flagged_tasks, key=flagged_tasks.get)
            st.subheader(f"Substructure Explainability: {top_flagged_task}")
            importance = all_importance[top_flagged_task]
            img_bytes_highlighted = render_molecule_image(mol, importance)
            st.image(img_bytes_highlighted, width=450)
            st.caption("Red intensity indicates atoms the model weighted most heavily for this prediction.")

            top_atoms = get_top_contributing_atoms(mol, importance, top_k=3)
            top_atoms_per_task = {top_flagged_task: top_atoms}
        else:
            top_atoms_per_task = {}

        st.divider()
        st.subheader("Research Triage Note")

        prompt = build_triage_prompt(smiles_input, predictions, top_atoms_per_task, threshold)

        if os.environ.get("ANTHROPIC_API_KEY"):
            with st.spinner("Generating triage note..."):
                response_text = call_claude_api(prompt)
            st.info(response_text)
        else:
            st.warning("ANTHROPIC_API_KEY not configured. Showing the structured prompt that "
                       "would be sent to the LLM in the deployed version:")
            st.code(prompt, language=None)

st.divider()
st.caption("MolScreen | Trained on the Tox21 dataset (NIH/EPA) | "
           "Mean test ROC-AUC: 0.8525 across 12 toxicity endpoints")
