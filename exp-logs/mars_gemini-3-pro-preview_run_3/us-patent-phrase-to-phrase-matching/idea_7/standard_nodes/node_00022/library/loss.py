import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson Correlation Loss.
    Computes 1 - Pearson Correlation Coefficient.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # Flatten tensors to ensure 1D vectors
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Center the vectors
        preds_mean = preds - torch.mean(preds)
        targets_mean = targets - torch.mean(targets)

        # Compute covariance
        cov = torch.sum(preds_mean * targets_mean)

        # Compute standard deviations (L2 norms of centered vectors)
        # Add epsilon to prevent division by zero
        preds_std = torch.sqrt(torch.sum(preds_mean**2) + 1e-8)
        targets_std = torch.sqrt(torch.sum(targets_mean**2) + 1e-8)

        # Compute Pearson Correlation
        pearson = cov / (preds_std * targets_std)

        # Return Loss (1 - correlation)
        return 1.0 - pearson


class CompositeLoss(nn.Module):
    """
    Composite Loss Function combining MSE, Cross-Entropy, and Pearson Correlation.
    L_Total = L_MSE + lambda1 * L_CE + lambda2 * (1 - Pearson)
    """

    def __init__(self, config):
        super().__init__()
        self.mse_weight = config.loss_mse_weight
        self.ce_weight = config.loss_ce_weight
        self.pearson_weight = config.loss_pearson_weight

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.pearson = PearsonLoss()

    def forward(self, reg_logits, cls_logits, targets):
        """
        Args:
            reg_logits (torch.Tensor): Regression predictions (batch_size, 1) or (batch_size,)
            cls_logits (torch.Tensor): Classification logits (batch_size, num_classes)
            targets (torch.Tensor): Ground truth similarity scores [0, 1] (batch_size,)

        Returns:
            loss (torch.Tensor): The weighted composite loss.
            metrics (dict): Dictionary containing individual loss components.
        """
        # Ensure targets are on the correct device
        targets = targets.to(reg_logits.device)

        # 1. MSE Loss (Regression)
        loss_mse = self.mse(reg_logits.view(-1), targets.view(-1))

        # 2. Cross-Entropy Loss (Classification)
        # Convert continuous scores to class indices: 0.0->0, 0.25->1, ..., 1.0->4
        # We multiply by 4 and round to handle potential float precision issues
        class_targets = (targets * 4).round().long()
        loss_ce = self.ce(cls_logits, class_targets)

        # 3. Pearson Loss (Regression)
        loss_pearson = self.pearson(reg_logits, targets)

        # Weighted Sum
        total_loss = (
            (self.mse_weight * loss_mse)
            + (self.ce_weight * loss_ce)
            + (self.pearson_weight * loss_pearson)
        )

        metrics = {
            "loss_mse": loss_mse.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_pearson": loss_pearson.detach(),
            "loss_total": total_loss.detach(),
        }

        return total_loss, metrics
