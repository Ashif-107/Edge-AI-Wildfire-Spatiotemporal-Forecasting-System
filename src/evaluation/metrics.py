"""
Evaluation Metrics Engine for Wildfire Risk Forecasting.

Calculates:
  1. Precision-Recall AUC (PR-AUC)
  2. Intersection over Union (IoU / Jaccard Index)
  3. Per-horizon Confusion Matrix (TP, FP, TN, FN, Precision, Recall, F1)
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix


def compute_iou(pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.35) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index) for binary fire prediction.
    """
    preds_binary = (pred_probs >= threshold).astype(np.uint8)
    targets_binary = (targets >= 0.5).astype(np.uint8)

    intersection = np.logical_and(preds_binary, targets_binary).sum()
    union = np.logical_or(preds_binary, targets_binary).sum()

    if union == 0:
        return 1.0  # Perfect prediction if no fire present in ground truth and model predicted no fire
    return float(intersection / (union + 1e-8))


def compute_pr_auc(pred_probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes Area Under Precision-Recall Curve (PR-AUC).
    """
    pred_flat = pred_probs.ravel()
    target_flat = (targets.ravel() >= 0.5).astype(np.uint8)

    precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
    pr_auc_score = auc(recall, precision)
    return float(pr_auc_score)


def compute_horizon_metrics(pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.35):
    """
    Computes metrics breakdown per forecast horizon (24h, 48h, 72h).
    Inputs:
      pred_probs: shape [N, 3, H, W]
      targets: shape [N, 3, H, W]
    Returns dictionary with metrics per horizon (horizon 0=24h, 1=48h, 2=72h).
    """
    horizons = ["24h", "48h", "72h"]
    results = {}

    for idx, h_name in enumerate(horizons):
        p_h = pred_probs[:, idx, :, :]
        t_h = targets[:, idx, :, :]

        iou = compute_iou(p_h, t_h, threshold=threshold)
        pr_auc = compute_pr_auc(p_h, t_h)

        p_flat = (p_h.ravel() >= threshold).astype(np.uint8)
        t_flat = (t_h.ravel() >= 0.5).astype(np.uint8)

        cm = confusion_matrix(t_flat, p_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))

        results[h_name] = {
            "IoU": round(iou, 4),
            "PR_AUC": round(pr_auc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "ConfusionMatrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
        }

    return results


if __name__ == "__main__":
    dummy_pred = np.random.uniform(0, 1, size=(5, 3, 128, 128))
    dummy_target = (np.random.uniform(0, 1, size=(5, 3, 128, 128)) > 0.9).astype(np.float32)
    metrics = compute_horizon_metrics(dummy_pred, dummy_target)
    print("Metrics per horizon:", metrics)
