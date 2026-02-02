import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PearsonLoss(nn.Module):
    """
    Optimization objective that directly maximizes the Pearson Correlation Coefficient.
    Loss = 1 - Pearson_Correlation(predictions, targets)
    """

    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, ) or (batch_size, 1)
            targets: Tensor of shape (batch_size, ) or (batch_size, 1)
        """
        # Flatten tensors to 1D
        x = predictions.view(-1)
        y = targets.view(-1)

        # Calculate means
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Calculate Pearson Correlation
        # rho = sum(vx * vy) / (sqrt(sum(vx^2)) * sqrt(sum(vy^2)))
        # We add a small epsilon to the denominator to avoid division by zero
        cost = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8
        )

        # Return Loss (1 - correlation)
        # We clamp the correlation between -1 and 1 to ensure stability
        return 1.0 - torch.clamp(cost, min=-1.0, max=1.0)


class HybridLoss(nn.Module):
    """
    Composite loss function combining:
    1. MSE Loss (Regression precision)
    2. CrossEntropy Loss (Classification boundary sharpness)
    3. Pearson Loss (Metric optimization)
    """

    def __init__(self, cfg: Config):
        super(HybridLoss, self).__init__()
        self.cfg = cfg

        # Initialize component loss functions
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.pearson = PearsonLoss()

        # Weights
        self.w_mse = cfg.loss_mse_weight
        self.w_ce = cfg.loss_ce_weight
        self.w_pearson = cfg.loss_pearson_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary containing:
                - 'logits': Regression output (batch_size, 1)
                - 'class_logits': Classification output (batch_size, 5)
            targets: Ground truth scores (batch_size, )
        """
        # Extract model outputs
        reg_logits = outputs["logits"].view(-1)  # Flatten to (batch_size,)
        class_logits = outputs["class_logits"]  # (batch_size, 5)

        # 1. MSE Loss (Regression)
        loss_mse = self.mse(reg_logits, targets)

        # 2. Pearson Loss (Regression)
        loss_pearson = self.pearson(reg_logits, targets)

        # 3. Cross Entropy Loss (Classification)
        # Map float scores [0.0, 0.25, 0.5, 0.75, 1.0] to indices [0, 1, 2, 3, 4]
        # We multiply by 4 and round to handle potential float precision issues
        target_indices = (targets * 4).round().long()
        loss_ce = self.ce(class_logits, target_indices)

        # Weighted Sum
        total_loss = (
            (self.w_mse * loss_mse)
            + (self.w_ce * loss_ce)
            + (self.w_pearson * loss_pearson)
        )

        return {
            "loss": total_loss,
            "mse": loss_mse,
            "ce": loss_ce,
            "pearson": loss_pearson,
        }
