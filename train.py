"""
Training Pipeline for ConvLSTM2D and ST-GNN Wildfire Risk Forecasting Models.

Executes end-to-end model training, validation, metric evaluation, and checkpoint saving.
Saves full per-epoch history (loss, IoU, accuracy) to JSON for dashboard and notebook use.
"""

import sys
import os

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import json
import torch
import numpy as np
from src.data.wildfire_dataset import get_wildfire_dataloaders
from src.models.convlstm import ConvLSTM2D
from src.models.st_gnn import SpatiotemporalGNN
from src.models.losses import CombinedFocalDiceLoss
from src.evaluation.metrics import compute_iou, compute_pr_auc, compute_pixel_accuracy


def train_model(model, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3,
                device: str = 'cpu', save_path: str = 'checkpoints/model.pt'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = CombinedFocalDiceLoss()

    best_val_iou = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_iou': [], 'val_pr_auc': [],
        'val_accuracy': [], 'val_balanced_accuracy': [],
        'val_f1': [], 'val_precision': [], 'val_recall': []
    }

    print(f"\n--- Starting Training: {model.__class__.__name__} ({epochs} epochs) ---")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # â”€â”€ Training phase â”€â”€
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out_risk = model(bx)
            loss = criterion(out_risk, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # â”€â”€ Validation phase â”€â”€
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
        val_preds   = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)

        # â”€â”€ Metrics â”€â”€
        val_iou    = compute_iou(val_preds, val_targets, threshold=0.35)
        val_pr_auc = compute_pr_auc(val_preds, val_targets)
        acc_dict   = compute_pixel_accuracy(val_preds, val_targets, threshold=0.35)

        # Per-pixel precision / recall / F1
        p_flat = (val_preds.ravel() >= 0.35).astype(np.uint8)
        t_flat = (val_targets.ravel() >= 0.5).astype(np.uint8)
        tp = float(np.logical_and(p_flat, t_flat).sum())
        fp = float(np.logical_and(p_flat == 1, t_flat == 0).sum())
        fn = float(np.logical_and(p_flat == 0, t_flat == 1).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val IoU: {val_iou:.4f} | Val Acc: {acc_dict['overall_accuracy']:.4f} "
              f"| F1: {f1:.4f} | Time: {elapsed:.2f}s")

        history['train_loss'].append(round(train_loss, 6))
        history['val_loss'].append(round(val_loss, 6))
        history['val_iou'].append(round(val_iou, 4))
        history['val_pr_auc'].append(round(val_pr_auc, 4))
        history['val_accuracy'].append(round(acc_dict['overall_accuracy'], 4))
        history['val_balanced_accuracy'].append(round(acc_dict['balanced_accuracy'], 4))
        history['val_f1'].append(round(f1, 4))
        history['val_precision'].append(round(precision, 4))
        history['val_recall'].append(round(recall, 4))

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best checkpoint to {save_path} (Val IoU: {val_iou:.4f})")

    return history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    print("Generating spatiotemporal dataset split...")
    dataloaders = get_wildfire_dataloaders(batch_size=4, seq_len=5, total_days=35, seed=42)

    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    print(f"Dataset split: train={dataloaders['shapes']['train_len']} | "
          f"val={dataloaders['shapes']['val_len']} | "
          f"test={dataloaders['shapes']['test_len']} samples")

    # 1. Train ConvLSTM2D (10 epochs)
    convlstm_model = ConvLSTM2D(in_channels=7, hidden_dims=[32, 16], out_horizons=3)
    history_convlstm = train_model(
        convlstm_model, train_loader, val_loader,
        epochs=10, lr=1e-3, device=device,
        save_path='checkpoints/convlstm_best.pt'
    )

    # 2. Train ST-GNN (10 epochs)
    st_gnn_model = SpatiotemporalGNN(in_channels=7, patch_size=4, hidden_dim=32, out_horizons=3)
    history_st_gnn = train_model(
        st_gnn_model, train_loader, val_loader,
        epochs=10, lr=1e-3, device=device,
        save_path='checkpoints/st_gnn_best.pt'
    )

    # Save full training history
    os.makedirs('checkpoints', exist_ok=True)
    summary = {
        'convlstm': {
            'best_iou':           max(history_convlstm['val_iou']),
            'best_accuracy':      max(history_convlstm['val_accuracy']),
            'best_f1':            max(history_convlstm['val_f1']),
            'best_pr_auc':        max(history_convlstm['val_pr_auc']),
            'history':            history_convlstm
        },
        'st_gnn': {
            'best_iou':           max(history_st_gnn['val_iou']),
            'best_accuracy':      max(history_st_gnn['val_accuracy']),
            'best_f1':            max(history_st_gnn['val_f1']),
            'best_pr_auc':        max(history_st_gnn['val_pr_auc']),
            'history':            history_st_gnn
        }
    }
    with open('checkpoints/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== Training Completed ===")
    print(f"ConvLSTM2D  -> Best Val IoU: {summary['convlstm']['best_iou']:.4f}  "
          f"| Accuracy: {summary['convlstm']['best_accuracy']:.4f}  "
          f"| F1: {summary['convlstm']['best_f1']:.4f}")
    print(f"ST-GNN      -> Best Val IoU: {summary['st_gnn']['best_iou']:.4f}  "
          f"| Accuracy: {summary['st_gnn']['best_accuracy']:.4f}  "
          f"| F1: {summary['st_gnn']['best_f1']:.4f}")


if __name__ == "__main__":
    main()

