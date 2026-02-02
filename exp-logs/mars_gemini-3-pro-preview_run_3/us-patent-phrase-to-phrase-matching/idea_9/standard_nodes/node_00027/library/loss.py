import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PearsonLoss(nn.Module):
    """
    Differentiable Pearson Correlation Loss.
    Loss = 1 - Pearson Correlation Coefficient.
    """

    def __init__(self, eps=1e-6):
        super(PearsonLoss, self).__init__()
        self.eps = eps

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Predicted scores, shape (batch_size, ) or (batch_size, 1)
            targets (torch.Tensor): Ground truth scores, shape (batch_size, ) or (batch_size, 1)
        """
        # Flatten tensors
        x = predictions.view(-1)
        y = targets.view(-1)

        # Mean centering
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        # Covariance
        cost = torch.sum(vx * vy)

        # Variances
        var_x = torch.sum(vx**2)
        var_y = torch.sum(vy**2)

        # Pearson Correlation Coefficient
        # Add epsilon for numerical stability
        denom = torch.sqrt(var_x * var_y) + self.eps
        pearson_score = cost / denom

        # We want to maximize correlation, so minimize (1 - r)
        return 1.0 - pearson_score


class HybridLoss(nn.Module):
    """
    Hybrid Loss function combining MSE, CrossEntropy (Auxiliary), and Pearson Loss.

    Expects model output to be a dictionary containing:
        - 'logits': Regression output (batch_size, 1)
        - 'class_logits': Classification logits (batch_size, num_bins)
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.mse = nn.MSELoss()
        # Although the config key is 'bce', we use CrossEntropyLoss for the 5-class auxiliary task
        # as implied by num_classification_bins=5.
        self.ce = nn.CrossEntropyLoss()
        self.pearson = PearsonLoss()

        self.weights = Config.loss_weights
        self.num_bins = Config.num_classification_bins

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing model predictions.
                - 'logits': Regression scores (batch_size, 1)
                - 'class_logits': Auxiliary classification logits (batch_size, num_bins)
            targets (torch.Tensor): Ground truth similarity scores (0.0 to 1.0).

        Returns:
            loss (torch.Tensor): Weighted sum of losses.
            metrics (dict): Dictionary of individual loss components for logging.
        """
        # 1. Extract predictions
        regression_logits = outputs["logits"].view(-1)
        class_logits = outputs["class_logits"]

        # 2. Prepare targets
        # Regression targets: same as input (0.0 to 1.0)
        reg_targets = targets.view(-1)

        # Classification targets: Map 0.0-1.0 to integer indices 0-4
        # 0.0 -> 0, 0.25 -> 1, 0.5 -> 2, 0.75 -> 3, 1.0 -> 4
        # Formula: round(score * 4)
        class_targets = (targets * (self.num_bins - 1)).long().view(-1)

        # 3. Compute individual losses
        loss_mse = self.mse(regression_logits, reg_targets)
        loss_pearson = self.pearson(regression_logits, reg_targets)
        loss_ce = self.ce(class_logits, class_targets)

        # 4. Weighted Sum
        # Note: Config uses 'bce' key for the classification weight
        total_loss = (
            self.weights["mse"] * loss_mse
            + self.weights["bce"] * loss_ce
            + self.weights["pearson"] * loss_pearson
        )

        # 5. Return loss and components dictionary
        metrics = {
            "loss_total": total_loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_ce": loss_ce.item(),
            "loss_pearson": loss_pearson.item(),
        }

        return total_loss, metrics
