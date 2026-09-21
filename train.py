"""
Training Pipeline for ConvLSTM2D and ST-GNN Wildfire Risk Forecasting Models.

Executes end-to-end model training, validation, metric evaluation, and checkpoint saving.
"""

import os
import time
import json
import torch
import numpy as np
from src.data.wildfire_dataset import get_wildfire_dataloaders
from src.models.convlstm import ConvLSTM2D
from src.models.st_gnn import SpatiotemporalGNN
from src.models.losses import CombinedFocalDiceLoss
from src.evaluation.metrics import compute_iou, compute_pr_auc


def train_model(model, train_loader, val_loader, epochs: int = 5, lr: float = 1e-3, device: str = 'cpu', save_path: str = 'checkpoints/model.pt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedFocalDiceLoss()

    best_val_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

    print(f"\n--- Starting Training: {model.__class__.__name__} ({epochs} epochs) ---")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_risk = model(bx)
            loss = criterion(out_risk, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out_risk = model(bx)
                loss = criterion(out_risk, by)
                val_loss += loss.item() * bx.size(0)

                val_preds.append(out_risk.cpu().numpy())
                val_targets.append(by.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        # Average IoU across all horizons
        val_iou = compute_iou(val_preds, val_targets, threshold=0.35)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Time: {elapsed:.2f}s")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val IoU: {val_iou:.4f})")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # Prepare DataLoaders
    print("Generating spatiotemporal dataset split...")
    dataloaders = get_wildfire_dataloaders(batch_size=4, seq_len=5, total_days=60, seed=42)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    # 1. Train Primary Model: Stacked ConvLSTM2D
    convlstm_model = ConvLSTM2D(in_channels=7, hidden_dims=[32, 16], out_horizons=3)
    history_convlstm = train_model(
        convlstm_model, train_loader, val_loader,
        epochs=5, lr=1e-3, device=device,
        save_path='checkpoints/convlstm_best.pt'
    )

    # 2. Train Comparative Model: ST-GNN
    st_gnn_model = SpatiotemporalGNN(in_channels=7, patch_size=4, hidden_dim=32, out_horizons=3)
    history_st_gnn = train_model(
        st_gnn_model, train_loader, val_loader,
        epochs=5, lr=1e-3, device=device,
        save_path='checkpoints/st_gnn_best.pt'
    )

    # Save summary log
    os.makedirs('checkpoints', exist_ok=True)
    summary = {
        'convlstm_best_iou': max(history_convlstm['val_iou']),
        'st_gnn_best_iou': max(history_st_gnn['val_iou'])
    }
    with open('checkpoints/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Training Completed Successfully ===")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
