import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SmoothingLoss(nn.Module):
    """
    Computes the Truncated Mean Squared Error (MSE) loss on log-probabilities
    of adjacent frames to enforce temporal smoothness.
    """

    def __init__(self, threshold=16.0):
        """
        Args:
            threshold (float): The maximum squared difference value allowed.
                               Differences larger than this (e.g., at true boundaries)
                               are clipped to prevent excessive penalization.
        """
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits: Tensor of shape (Batch, Classes, Time) containing raw scores.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log-probabilities: log(P_t)
        log_probs = F.log_softmax(logits, dim=1)

        # Calculate temporal difference: log(P_t) - log(P_{t-1})
        # Slicing: [:, :, 1:] is t=1..T, [:, :, :-1] is t=0..T-1
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared Error
        mse = diff**2

        # Truncate (clamp) the error to allow for sharp transitions at gesture boundaries
        truncated_mse = torch.clamp(mse, max=self.threshold)

        # Return mean over batch, classes, and time
        return torch.mean(truncated_mse)


class CascadedLoss(nn.Module):
    """
    Aggregates losses from the three stages of the RSK-ARN model.
    Stage 1: Weighted Cross Entropy
    Stage 2: Weighted Cross Entropy + Smoothing Loss
    Stage 3: Weighted Cross Entropy + Smoothing Loss
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Initialize Class Weights for CrossEntropy
        # Weights are defined in Config (e.g., 0.2 for background, 1.0 for gestures)
        weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(
            Config.DEVICE
        )

        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.smoothing_loss = SmoothingLoss(threshold=16.0)
        self.lambda_smoothing = Config.LAMBDA_SMOOTHING

    def forward(self, stage1_logits, stage2_logits, stage3_logits, targets):
        """
        Args:
            stage1_logits: (Batch, Classes, Time) - Output from Bi-GRU
            stage2_logits: (Batch, Classes, Time) - Output from 1st Refinement
            stage3_logits: (Batch, Classes, Time) - Output from 2nd Refinement
            targets: (Batch, Time) - Ground truth class indices

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # 1. Cross Entropy Losses for all stages
        # PyTorch CrossEntropyLoss handles (N, C, T) input and (N, T) target
        loss_ce_s1 = self.ce_loss(stage1_logits, targets)
        loss_ce_s2 = self.ce_loss(stage2_logits, targets)
        loss_ce_s3 = self.ce_loss(stage3_logits, targets)

        # 2. Smoothing Losses for refinement stages only
        loss_smooth_s2 = self.smoothing_loss(stage2_logits)
        loss_smooth_s3 = self.smoothing_loss(stage3_logits)

        # 3. Aggregate Total Loss
        # L_total = L_s1 + (L_s2 + lambda * Smooth_s2) + (L_s3 + lambda * Smooth_s3)
        total_loss = (
            loss_ce_s1
            + (loss_ce_s2 + self.lambda_smoothing * loss_smooth_s2)
            + (loss_ce_s3 + self.lambda_smoothing * loss_smooth_s3)
        )

        # Create metrics dictionary for logging
        metrics = {
            "loss_ce_s1": loss_ce_s1.item(),
            "loss_ce_s2": loss_ce_s2.item(),
            "loss_ce_s3": loss_ce_s3.item(),
            "loss_smooth_s2": loss_smooth_s2.item(),
            "loss_smooth_s3": loss_smooth_s3.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
