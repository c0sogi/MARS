import torch
import torch.nn as nn
from library.config import Config


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson Correlation Loss.
    Loss = 1 - Pearson_Correlation(predictions, targets)
    """

    def __init__(self):
        super(PearsonLoss, self).__init__()
        self.epsilon = 1e-8

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted scores of shape [batch_size]
            targets (torch.Tensor): Ground truth scores of shape [batch_size]
        """
        # Avoid calculation for small batches (e.g., last batch in validation)
        if preds.size(0) < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # Center the data (subtract mean)
        preds_mean = preds - torch.mean(preds)
        targets_mean = targets - torch.mean(targets)

        # Compute numerator (covariance)
        numerator = torch.sum(preds_mean * targets_mean)

        # Compute denominator (product of standard deviations)
        denominator = torch.sqrt(torch.sum(preds_mean**2) * torch.sum(targets_mean**2))

        # Add epsilon to prevent division by zero
        denominator = torch.max(
            denominator, torch.tensor(self.epsilon, device=preds.device)
        )

        # Calculate Pearson Correlation
        pearson_corr = numerator / denominator

        # Return Loss (1 - correlation)
        # We want to maximize correlation, so we minimize (1 - correlation)
        return 1.0 - pearson_corr


class HybridLoss(nn.Module):
    """
    Composite loss function combining MSE, CrossEntropy, and Pearson Correlation.
    L_Total = w1 * MSE + w2 * CE + w3 * (1 - Pearson)
    """

    def __init__(self):
        super(HybridLoss, self).__init__()

        # Initialize Loss Components
        self.mse_loss_fn = nn.MSELoss()
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.pearson_loss_fn = PearsonLoss()

        # Load weights from Config
        self.w_mse = Config.loss_mse_weight
        self.w_ce = Config.loss_ce_weight
        self.w_pearson = Config.loss_pearson_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs (tuple): (regression_logits, classification_logits)
                - regression_logits: [batch_size]
                - classification_logits: [batch_size, num_aux_classes]
            targets (torch.Tensor): Ground truth scores [batch_size] (floats 0.0-1.0)

        Returns:
            loss (torch.Tensor): The weighted total loss.
            metrics (dict): Dictionary containing individual loss components for logging.
        """
        logits, aux_logits = outputs

        # 1. MSE Loss (Regression Head)
        loss_mse = self.mse_loss_fn(logits, targets)

        # 2. Cross Entropy Loss (Auxiliary Classification Head)
        # Convert continuous targets (0.0, 0.25, 0.5, 0.75, 1.0) to indices (0, 1, 2, 3, 4)
        # We multiply by 4 and round to nearest integer.
        target_indices = (targets * 4.0).round().long()
        loss_ce = self.ce_loss_fn(aux_logits, target_indices)

        # 3. Pearson Loss (Regression Head)
        loss_pearson = self.pearson_loss_fn(logits, targets)

        # 4. Weighted Sum
        total_loss = (
            (self.w_mse * loss_mse)
            + (self.w_ce * loss_ce)
            + (self.w_pearson * loss_pearson)
        )

        # Prepare metrics dictionary
        metrics = {
            "loss_mse": loss_mse.item(),
            "loss_ce": loss_ce.item(),
            "loss_pearson": loss_pearson.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, metrics
