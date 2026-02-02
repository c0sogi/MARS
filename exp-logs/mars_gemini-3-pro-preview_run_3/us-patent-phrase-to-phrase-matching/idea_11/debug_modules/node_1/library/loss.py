import torch
import torch.nn as nn
from library.config import CFG


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson Correlation Loss.
    Loss = 1 - Pearson_Correlation(predictions, targets)
    """

    def __init__(self):
        super(PearsonLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Predicted scores (Batch,).
            targets (torch.Tensor): Ground truth scores (Batch,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure inputs are flattened and float
        x = predictions.view(-1).float()
        y = targets.view(-1).float()

        # Calculate means
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Calculate covariance and variances
        # Note: We don't need to divide by N-1 for correlation if we do it consistently
        # because it cancels out in the numerator and denominator.
        cost = torch.sum(vx * vy)

        # Add epsilon to denominator for stability
        denom = (
            torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + self.epsilon
        )

        pearson_score = cost / denom

        # We want to maximize correlation, so minimize (1 - correlation)
        return 1.0 - pearson_score


class HybridLoss(nn.Module):
    """
    Combines MSE, CrossEntropy, and Pearson Loss based on configuration weights.
    L_Total = w_mse * MSE + w_ce * CE + w_pearson * (1 - Pearson)
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.pearson = PearsonLoss()

        # Load weights from config
        self.weights = CFG.loss_weights

        # Ensure weights exist
        self.w_mse = self.weights.get("mse", 1.0)
        self.w_ce = self.weights.get("ce", 0.5)
        self.w_pearson = self.weights.get("pearson", 1.0)

    def forward(self, outputs, targets, targets_cls):
        """
        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'logits': Regression predictions (Batch,)
                - 'logits_cls': Classification logits (Batch, Num_Classes)
            targets (torch.Tensor): Regression targets (Batch,)
            targets_cls (torch.Tensor): Classification targets (Batch,)

        Returns:
            loss (torch.Tensor): The weighted total loss.
            loss_dict (dict): Dictionary containing individual loss components for logging.
        """
        logits_reg = outputs["logits"]
        logits_cls = outputs["logits_cls"]

        # 1. MSE Loss (Regression)
        # Ensure shapes match (Batch,) vs (Batch,)
        loss_mse = self.mse(logits_reg.view(-1), targets.view(-1))

        # 2. Cross Entropy Loss (Classification)
        # logits_cls: (Batch, Num_Classes), targets_cls: (Batch,)
        loss_ce = self.ce(logits_cls, targets_cls)

        # 3. Pearson Loss (Regression)
        loss_pearson = self.pearson(logits_reg.view(-1), targets.view(-1))

        # Weighted Sum
        total_loss = (
            (self.w_mse * loss_mse)
            + (self.w_ce * loss_ce)
            + (self.w_pearson * loss_pearson)
        )

        loss_dict = {
            "loss": total_loss.item(),
            "mse": loss_mse.item(),
            "ce": loss_ce.item(),
            "pearson": loss_pearson.item(),
        }

        return total_loss, loss_dict
