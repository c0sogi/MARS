import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PearsonLoss(nn.Module):
    """
    Loss function based on the Pearson Correlation Coefficient.
    Loss = 1 - Pearson_Correlation(predictions, targets)
    """

    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted scores of shape (B, 1) or (B,).
            targets (torch.Tensor): Ground truth scores of shape (B, 1) or (B,).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Flatten tensors
        x = preds.view(-1)
        y = targets.view(-1)

        # Pearson correlation is undefined for batch size < 2
        if x.size(0) < 2:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        # Calculate means
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Calculate covariance and variances
        # Add epsilon for numerical stability
        cost = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8
        )

        # Constrain cost to [-1, 1] range to avoid numerical errors going slightly out of bounds
        cost = torch.clamp(cost, min=-1.0, max=1.0)

        # We want to maximize correlation, so minimize (1 - correlation)
        return 1.0 - cost


class HybridLoss(nn.Module):
    """
    Hybrid Loss combining MSE, CrossEntropy, and Pearson Correlation.
    Expects model output to be a dictionary with 'logits' (regression) and 'class_logits' (classification).
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.pearson = PearsonLoss()

        # Load weights from Config
        self.mse_weight = Config.mse_weight
        self.ce_weight = Config.ce_weight
        self.pearson_weight = Config.pearson_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing:
                - 'logits': Regression output (B, 1)
                - 'class_logits': Classification output (B, 5)
            targets (torch.Tensor): Ground truth scores (B,) containing floats [0.0, 0.25, 0.5, 0.75, 1.0]

        Returns:
            dict: Dictionary containing 'loss' (total) and individual components for logging.
        """
        pred_score = outputs["logits"].view(-1)
        pred_classes = outputs["class_logits"]

        # Ensure targets are on the correct device and shape
        targets = targets.to(pred_score.device).view(-1)

        # 1. MSE Loss (Regression)
        loss_mse = self.mse(pred_score, targets)

        # 2. Pearson Loss (Regression)
        loss_pearson = self.pearson(pred_score, targets)

        # 3. Cross Entropy Loss (Classification)
        # Convert continuous scores (0.0, 0.25, ...) to class indices (0, 1, 2, 3, 4)
        # Formula: class_idx = score * 4
        # We round to handle potential floating point inaccuracies
        target_classes = (targets * 4).round().long()
        loss_ce = self.ce(pred_classes, target_classes)

        # Weighted Sum
        total_loss = (
            (self.mse_weight * loss_mse)
            + (self.ce_weight * loss_ce)
            + (self.pearson_weight * loss_pearson)
        )

        return {
            "loss": total_loss,
            "mse": loss_mse,
            "ce": loss_ce,
            "pearson": loss_pearson,
        }
