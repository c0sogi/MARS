import torch
import torch.nn as nn
from library.config import CFG


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson Correlation Loss.
    Loss = 1 - Pearson Correlation Coefficient.
    """

    def __init__(self):
        super(PearsonLoss, self).__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted scores, shape (batch_size, 1) or (batch_size,)
            targets (torch.Tensor): Ground truth scores, shape (batch_size, 1) or (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten tensors to ensure 1D vectors
        x = preds.view(-1)
        y = targets.view(-1)

        # Calculate means
        x_mean = x - torch.mean(x)
        y_mean = y - torch.mean(y)

        # Calculate numerator (covariance * N)
        numerator = torch.sum(x_mean * y_mean)

        # Calculate denominator (std_x * std_y * N)
        denominator = torch.sqrt(torch.sum(x_mean**2)) * torch.sqrt(
            torch.sum(y_mean**2)
        )

        # Calculate Pearson Correlation (rho)
        # Add a small epsilon to denominator to prevent division by zero
        rho = numerator / (denominator + 1e-8)

        # Return loss (1 - rho)
        # We clamp rho to [-1, 1] for numerical stability before subtraction
        return 1.0 - torch.clamp(rho, min=-1.0, max=1.0)


class HybridPearsonLoss(nn.Module):
    """
    Hybrid Loss combining MSE, Cross Entropy, and Pearson Loss.

    L_total = L_MSE + lambda_ce * L_CE + lambda_pearson * L_Pearson

    - MSE ensures geometric closeness to the target score.
    - Cross Entropy (on auxiliary classification head) sharpens decision boundaries
      between the discrete score levels (0, 0.25, 0.5, 0.75, 1.0).
    - Pearson Loss directly optimizes the competition metric.
    """

    def __init__(self, cfg=CFG):
        super(HybridPearsonLoss, self).__init__()
        self.cfg = cfg

        # Loss components
        self.mse_loss_fn = nn.MSELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.pearson_loss_fn = PearsonLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'score': Regression predictions (batch_size, 1)
                - 'logits': Classification logits (batch_size, 5)
            targets (torch.Tensor): Ground truth scores (batch_size,)

        Returns:
            dict: Dictionary containing 'loss' (total) and individual components.
        """
        # Extract predictions
        pred_score = outputs["score"].view(-1)
        pred_logits = outputs["logits"]

        # Ensure targets are on the correct device and shape
        targets = targets.to(pred_score.device).view(-1)

        # 1. MSE Loss (Regression)
        mse_loss = self.mse_loss_fn(pred_score, targets)

        # 2. Pearson Loss (Regression)
        pearson_loss = self.pearson_loss_fn(pred_score, targets)

        # 3. Cross Entropy Loss (Classification)
        # Convert continuous scores (0.0, 0.25, ..., 1.0) to class indices (0, 1, ..., 4)
        # Formula: index = round(score * 4)
        target_classes = (targets * 4).round().long()
        ce_loss = self.ce_loss_fn(pred_logits, target_classes)

        # Weighted Sum
        total_loss = (
            mse_loss
            + self.cfg.lambda_ce * ce_loss
            + self.cfg.lambda_pearson * pearson_loss
        )

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "ce": ce_loss,
            "pearson": pearson_loss,
        }
