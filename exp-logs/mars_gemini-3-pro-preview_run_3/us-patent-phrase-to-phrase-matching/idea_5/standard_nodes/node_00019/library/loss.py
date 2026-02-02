import torch
import torch.nn as nn
from library.config import Config


class HybridLoss(nn.Module):
    """
    Implements a hybrid objective function combining Mean Squared Error (MSE)
    for regression and Cross Entropy (CE) for classification.

    This loss function is designed to optimize both the geometric closeness of the
    prediction (MSE) and the decision boundary sharpness (CE) for the discrete
    score buckets.

    Formula: Loss = MSE(y_pred, y_true) + lambda * CE(y_class_pred, y_class_true)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.hybrid_lambda = Config.hybrid_lambda
        self.use_hybrid = Config.use_hybrid_loss

    def forward(self, outputs, targets, bin_targets=None):
        """
        Computes the loss.

        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'logits': Regression scores [batch_size, 1]
                - 'class_logits': Classification scores [batch_size, num_classes]
            targets (torch.Tensor): Ground truth regression scores [batch_size].
            bin_targets (torch.Tensor, optional): Ground truth classification bins [batch_size].
                                                  Required if Config.use_hybrid_loss is True.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        # Extract regression logits and flatten to match targets shape [batch_size]
        logits = outputs["logits"].view(-1)

        # Compute Regression Loss (MSE)
        loss = self.mse(logits, targets)

        # If hybrid loss is enabled and bin_targets are provided, add classification loss
        if self.use_hybrid and bin_targets is not None:
            class_logits = outputs["class_logits"]

            # Compute Classification Loss (Cross Entropy)
            loss_ce = self.ce(class_logits, bin_targets)

            # Combine losses
            loss = loss + (self.hybrid_lambda * loss_ce)

        return loss
