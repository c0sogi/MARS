import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CFG


class PearsonLoss(nn.Module):
    """
    Loss function based on the Pearson Correlation Coefficient.
    Minimizes 1 - rho, where rho is the correlation between preds and targets.
    """

    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, preds, targets):
        # Flatten tensors to 1D
        x = preds.view(-1)
        y = targets.view(-1)

        # Center the data (subtract mean)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Compute covariance and variances
        # Add small epsilon to avoid division by zero
        cov = torch.sum(vx * vy)
        var_x = torch.sum(vx**2)
        var_y = torch.sum(vy**2)

        # Calculate Pearson Correlation Coefficient
        denom = torch.sqrt(var_x * var_y) + 1e-8
        rho = cov / denom

        # Return loss (1 - correlation)
        return 1.0 - rho


class HybridLoss(nn.Module):
    """
    Hybrid Loss function combining MSE, Cross Entropy, and Pearson Loss.
    L_Total = L_MSE + lambda_ce * L_CE + lambda_pearson * (1 - rho)
    """

    def __init__(self, config=CFG):
        super(HybridLoss, self).__init__()
        self.config = config

        # Initialize sub-loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.pearson_loss = PearsonLoss()

        # Load weights from config
        self.mse_weight = config.loss_config.get("mse_weight", 1.0)
        self.ce_weight = config.loss_config.get("ce_weight", 0.2)
        self.pearson_weight = config.loss_config.get("pearson_weight", 0.5)
        self.ce_bins = config.loss_config.get("ce_bins", 10)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Dictionary containing 'score' (regression output) and
                    optionally 'logits' (classification output).
                    Or a single tensor (assumed to be scores).
            targets: Ground truth similarity scores (0.0 to 1.0).
        """
        # Handle input format
        if isinstance(inputs, dict):
            scores = inputs.get("score")
            logits = inputs.get("logits")
        else:
            scores = inputs
            logits = None

        # Ensure correct shapes for regression
        scores = scores.view(-1)
        targets = targets.view(-1)

        # 1. MSE Loss
        loss_mse = self.mse_loss(scores, targets)

        # 2. Pearson Loss
        loss_pearson = self.pearson_loss(scores, targets)

        # 3. Cross Entropy Loss (Auxiliary)
        loss_ce = torch.tensor(0.0, device=scores.device)
        if logits is not None and self.ce_weight > 0:
            # Discretize continuous targets into bins for classification
            # e.g., if bins=10, 0.25 -> 2.25 -> round to 2
            # 0.0 -> 0, 1.0 -> 9
            class_targets = torch.round(targets * (self.ce_bins - 1)).long()

            # Clamp to ensure indices are within valid range [0, bins-1]
            class_targets = torch.clamp(class_targets, 0, self.ce_bins - 1)

            loss_ce = self.ce_loss(logits, class_targets)

        # Combine losses
        total_loss = (
            (self.mse_weight * loss_mse)
            + (self.pearson_weight * loss_pearson)
            + (self.ce_weight * loss_ce)
        )

        return total_loss
