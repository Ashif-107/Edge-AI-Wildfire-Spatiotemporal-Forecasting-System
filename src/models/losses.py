"""
Focal Loss + Dice Loss for Extreme Class Imbalance (<1% Fire-Pixel Sparsity).

Formulation:
  Loss = alpha * FocalLoss(p, y) + beta * DiceLoss(p, y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for rare positive class detection.
    Focuses learning on hard, misclassified rare fire pixels.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: predicted probabilities [0, 1]
        targets: binary targets {0, 1}
        """
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        bce_loss = - (targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))

        # Focal term: (1 - pt)^gamma
        p_t = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting for positive vs negative class
        alpha_weight = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss / Jaccard similarity loss for spatial overlap optimization.
    """

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1.0 - dice


class CombinedFocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss objective for extreme wildfire pixel sparsity.
    """

    def __init__(self, focal_alpha: float = 0.85, focal_gamma: float = 2.0, alpha_weight: float = 0.6, beta_weight: float = 0.4):
        super(CombinedFocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.alpha_weight = alpha_weight
        self.beta_weight = beta_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.dice(inputs, targets)

        total_loss = self.alpha_weight * loss_focal + self.beta_weight * loss_dice
        return total_loss


if __name__ == "__main__":
    criterion = CombinedFocalDiceLoss()
    pred = torch.sigmoid(torch.randn(2, 3, 128, 128))
    target = (torch.rand(2, 3, 128, 128) > 0.95).float()  # Highly imbalanced
    loss = criterion(pred, target)
    print(f"Combined Focal+Dice Loss on imbalanced tensor: {loss.item():.4f}")
