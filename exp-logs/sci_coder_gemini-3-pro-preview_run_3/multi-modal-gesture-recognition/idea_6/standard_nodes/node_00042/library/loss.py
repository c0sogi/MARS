import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class SmoothingLoss(nn.Module):
    """
    Computes the Truncated Mean Squared Error (MSE) loss on the log-probabilities
    of adjacent frames. This loss penalizes rapid fluctuations in class predictions,
    enforcing temporal smoothness in the output.
    """

    def __init__(self, threshold=4.0):
        """
        Args:
            threshold (float): The truncation threshold for the log-probability differences.
                               Differences larger than this magnitude are clamped to avoid
                               exploding gradients at sharp boundaries. Default is 4.0.
        """
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (logits) of shape (Batch, Classes, Time).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log probabilities
        # Shape: (Batch, Classes, Time)
        log_probs = F.log_softmax(logits, dim=1)

        # Calculate temporal difference: log_p(t) - log_p(t-1)
        # We slice the tensor to align t and t-1
        # Shape: (Batch, Classes, Time-1)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Truncate the differences (clamp) to limit the penalty for valid sharp transitions
        diff = torch.clamp(diff, min=-self.threshold, max=self.threshold)

        # Compute MSE (Mean Squared Error) of the differences towards zero
        loss = torch.mean(diff**2)

        return loss


class CascadedLoss(nn.Module):
    """
    Aggregates the loss functions for the Three-Stage Hybrid Network.

    Components:
    1. Weighted Cross-Entropy Loss for Stage 1 (Bi-GRU).
    2. Weighted Cross-Entropy Loss for Stage 2 (Refinement TCN 1).
    3. Weighted Cross-Entropy Loss for Stage 3 (Refinement TCN 2).
    4. Smoothing Loss for Stage 2.
    5. Smoothing Loss for Stage 3.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Retrieve class weights from config and move to the appropriate device
        # Weights handle class imbalance (e.g., background class weight = 0.2)
        weights = config.CLASS_WEIGHTS.to(config.get_device())

        # Initialize Weighted Cross-Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        # Initialize Smoothing Loss
        self.smoothing_loss = SmoothingLoss()

        # Retrieve smoothing coefficient from config
        self.smoothing_lambda = config.SMOOTHING_LAMBDA

    def forward(self, predictions, targets):
        """
        Computes the total loss for the cascaded architecture.

        Args:
            predictions (list or tuple): A list containing three tensors [p1, p2, p3].
                - p1: Output logits from Stage 1 (Bi-GRU), shape (Batch, Classes, Time).
                - p2: Output logits from Stage 2 (TCN 1), shape (Batch, Classes, Time).
                - p3: Output logits from Stage 3 (TCN 2), shape (Batch, Classes, Time).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Time).

        Returns:
            torch.Tensor: The aggregated scalar loss.
        """
        # Unpack predictions
        p1, p2, p3 = predictions

        # 1. Compute Classification Losses (Weighted Cross-Entropy) for all stages
        loss_ce_1 = self.ce_loss(p1, targets)
        loss_ce_2 = self.ce_loss(p2, targets)
        loss_ce_3 = self.ce_loss(p3, targets)

        # 2. Compute Smoothing Losses for refinement stages only
        loss_smooth_2 = self.smoothing_loss(p2)
        loss_smooth_3 = self.smoothing_loss(p3)

        # 3. Aggregate Total Loss
        # Formula: L_total = sum(L_CE) + lambda * sum(L_Smooth)
        total_loss = (loss_ce_1 + loss_ce_2 + loss_ce_3) + self.smoothing_lambda * (
            loss_smooth_2 + loss_smooth_3
        )

        return total_loss
