"""
Evaluation Metrics Engine for Wildfire Risk Forecasting.

Calculates:
  1. Precision-Recall AUC (PR-AUC)
  2. Intersection over Union (IoU / Jaccard Index)
  3. Per-horizon Confusion Matrix (TP, FP, TN, FN)
  4. Pixel-level Accuracy, Balanced Accuracy
  5. Precision, Recall, F1-Score per horizon
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    balanced_accuracy_score, accuracy_score
)


def compute_iou(pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.35) -> float:
    """
    Computes Intersection over Union (IoU / Jaccard Index) for binary fire prediction.
    """
    preds_binary = (pred_probs >= threshold).astype(np.uint8)
    targets_binary = (targets >= 0.5).astype(np.uint8)

    intersection = np.logical_and(preds_binary, targets_binary).sum()
    union = np.logical_or(preds_binary, targets_binary).sum()

    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-8))


def compute_pr_auc(pred_probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes Area Under Precision-Recall Curve (PR-AUC).
    """
    pred_flat = pred_probs.ravel()
    target_flat = (targets.ravel() >= 0.5).astype(np.uint8)

    if target_flat.sum() == 0:
        return 0.0

    precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
    pr_auc_score = auc(recall, precision)
    return float(pr_auc_score)


def compute_pixel_accuracy(pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.35) -> dict:
    """
    Computes pixel-level classification accuracy metrics.
    Returns overall accuracy and balanced accuracy (handles class imbalance).
    """
    preds_flat = (pred_probs.ravel() >= threshold).astype(np.uint8)
    targets_flat = (targets.ravel() >= 0.5).astype(np.uint8)

    overall_acc = float(accuracy_score(targets_flat, preds_flat))
    balanced_acc = float(balanced_accuracy_score(targets_flat, preds_flat))
    fire_pixel_acc = 0.0
    fire_count = targets_flat.sum()
    if fire_count > 0:
        correct_fire = ((preds_flat == 1) & (targets_flat == 1)).sum()
        fire_pixel_acc = float(correct_fire / fire_count)

    return {
        "overall_accuracy": round(overall_acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "fire_pixel_recall": round(fire_pixel_acc, 4)
    }


def compute_horizon_metrics(pred_probs: np.ndarray, targets: np.ndarray, threshold: float = 0.35):
    """
    Computes full metrics breakdown per forecast horizon (24h, 48h, 72h).
    Inputs:
      pred_probs: shape [N, 3, H, W]
      targets:    shape [N, 3, H, W]
    Returns dictionary with all metrics per horizon.
    """
    horizons = ["24h", "48h", "72h"]
    results = {}

    for idx, h_name in enumerate(horizons):
        p_h = pred_probs[:, idx, :, :]
        t_h = targets[:, idx, :, :]

        iou = compute_iou(p_h, t_h, threshold=threshold)
        pr_auc = compute_pr_auc(p_h, t_h)
        acc_metrics = compute_pixel_accuracy(p_h, t_h, threshold=threshold)

        p_flat = (p_h.ravel() >= threshold).astype(np.uint8)
        t_flat = (t_h.ravel() >= 0.5).astype(np.uint8)

        cm = confusion_matrix(t_flat, p_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = float(tp / (tp + fp + 1e-8))
        recall    = float(tp / (tp + fn + 1e-8))
        f1        = float(2 * precision * recall / (precision + recall + 1e-8))
        specificity = float(tn / (tn + fp + 1e-8))

        results[h_name] = {
            "IoU":                round(iou, 4),
            "PR_AUC":             round(pr_auc, 4),
            "Accuracy":           acc_metrics["overall_accuracy"],
            "Balanced_Accuracy":  acc_metrics["balanced_accuracy"],
            "Fire_Pixel_Recall":  acc_metrics["fire_pixel_recall"],
            "Precision":          round(precision, 4),
            "Recall":             round(recall, 4),
            "Specificity":        round(specificity, 4),
            "F1_Score":           round(f1, 4),
            "ConfusionMatrix":    {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
        }

    return results


if __name__ == "__main__":
    dummy_pred   = np.random.uniform(0, 1, size=(5, 3, 128, 128))
    dummy_target = (np.random.uniform(0, 1, size=(5, 3, 128, 128)) > 0.9).astype(np.float32)
    metrics = compute_horizon_metrics(dummy_pred, dummy_target)
    for h, m in metrics.items():
        print(f"\n--- {h} ---")
        for k, v in m.items():
            print(f"  {k}: {v}")
