import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import cfg


class PearsonCorrelationLoss(nn.Module):
    """
    A differentiable implementation of the Pearson Correlation Coefficient loss.
    Loss = 1 - Pearson_Correlation(preds, labels)
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        """
        Args:
            preds (torch.Tensor): Predicted scores of shape (N, 1) or (N,).
            labels (torch.Tensor): Ground truth scores of shape (N, 1) or (N,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        x = preds.view(-1)
        y = labels.view(-1)

        # Center the vectors
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Compute Pearson correlation (cosine similarity of centered vectors)
        # Add epsilon to denominator to prevent division by zero
        denominator = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8
        correlation = torch.sum(vx * vy) / denominator

        # We want to maximize correlation, so minimize (1 - correlation)
        return 1 - correlation


class CompositeLoss(nn.Module):
    """
    Composite loss function combining MSE, Cross-Entropy, and Pearson Correlation.
    Weights for each component are defined in the global configuration.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.pearson = PearsonCorrelationLoss()

        # Load weights from config
        self.w_mse = cfg.mse_weight
        self.w_ce = cfg.ce_weight
        self.w_pearson = cfg.pearson_weight

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'logits': Regression scores (N, 1)
                - 'class_logits': Classification logits (N, num_classes)
            batch (dict): Dictionary containing targets:
                - 'labels': Regression targets (N,)
                - 'class_labels': Classification targets (N,)

        Returns:
            dict: Dictionary containing 'loss' (total weighted loss) and individual components.
        """
        # Extract predictions and targets
        # Flatten logits to match labels shape (N,)
        reg_logits = outputs["logits"].view(-1)
        reg_labels = batch["labels"].view(-1)

        class_logits = outputs["class_logits"]
        class_labels = batch["class_labels"]

        # 1. Mean Squared Error Loss (Regression)
        loss_mse = self.mse(reg_logits, reg_labels)

        # 2. Pearson Correlation Loss (Regression)
        loss_pearson = self.pearson(reg_logits, reg_labels)

        # 3. Cross Entropy Loss (Classification)
        loss_ce = self.ce(class_logits, class_labels)

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
