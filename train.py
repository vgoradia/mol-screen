"""
MolScreen training script.

Split strategy: random 80/10/10 train/val/test split, consistent with
the original Tox21 / MoleculeNet benchmark protocol (random split is
the standard for Tox21 in the literature, as opposed to scaffold split
which is more common for datasets like BBBP). This keeps our comparison
to Kong-style published baselines apples-to-apples.

Training: Adam optimizer, masked multi-label BCE loss, ReduceLROnPlateau
scheduler, early stopping on validation mean ROC-AUC across tasks
(the standard Tox21 evaluation metric in MoleculeNet).
"""

import pickle
import random
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from model import MolScreenGNN, masked_bce_loss

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
          'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
          'SR-MMP', 'SR-p53']


def split_dataset(graphs, train_frac=0.8, val_frac=0.1):
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]
    return train, val, test


def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_logits.append(logits.cpu())
            all_targets.append(batch.y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits).numpy()
    targets = targets.numpy()

    per_task_auc = {}
    for i, task in enumerate(TASKS):
        mask = ~np.isnan(targets[:, i])
        y_true = targets[mask, i]
        y_pred = probs[mask, i]
        # AUC undefined if only one class present in this split for this task
        if len(np.unique(y_true)) < 2:
            continue
        per_task_auc[task] = roc_auc_score(y_true, y_pred)

    mean_auc = float(np.mean(list(per_task_auc.values())))
    return mean_auc, per_task_auc


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open('tox21_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)

    train_graphs, val_graphs, test_graphs = split_dataset(graphs)
    print(f"Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64)
    test_loader = DataLoader(test_graphs, batch_size=64)

    model = MolScreenGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_val_auc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    max_epochs = 60

    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = masked_bce_loss(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / len(train_graphs)
        val_auc, _ = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_mean_auc={val_auc:.4f} | lr={current_lr:.6f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_auc': val_auc, 'lr': current_lr})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'molscreen_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    print(f"\nBest validation mean AUC: {best_val_auc:.4f}")

    # final test evaluation using best checkpoint
    model.load_state_dict(torch.load('molscreen_best.pt'))
    test_auc, per_task_auc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Mean ROC-AUC: {test_auc:.4f}")
    print("\nPer-task Test ROC-AUC:")
    for task, auc in per_task_auc.items():
        print(f"  {task:15s}: {auc:.4f}")

    with open('training_history.pkl', 'wb') as f:
        pickle.dump({'history': history, 'test_auc': test_auc, 'per_task_auc': per_task_auc}, f)

    return model, test_auc, per_task_auc


if __name__ == "__main__":
    train()
