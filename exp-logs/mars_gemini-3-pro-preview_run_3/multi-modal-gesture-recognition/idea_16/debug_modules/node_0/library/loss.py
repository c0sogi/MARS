import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TruncatedMSE(nn.Module):
    """
    Computes the Truncated Mean Squared Error (TMSE) loss for temporal smoothing.
    Applied to the log-probabilities of adjacent frames to encourage smooth transitions
    while allowing for sharp changes at true boundaries (via truncation).

    Formula: mean(clamp((log_prob[t] - log_prob[t-1])^2, max=threshold))
    """

    def __init__(self, threshold=4.0):
        super(TruncatedMSE, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits: (Batch, Time, NumClasses) tensor of raw output scores.
        Returns:
            loss: Scalar tensor representing the smoothing loss.
        """
        # Convert logits to log-probabilities
        # Shape: (Batch, Time, NumClasses)
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate difference between adjacent frames: t and t-1
        # diff shape: (Batch, Time-1, NumClasses)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        mse = diff.pow(2)

        # Truncate gradients for large errors (likely boundaries)
        truncated_mse = torch.clamp(mse, min=0, max=self.threshold)

        # Average over all dimensions
        loss = torch.mean(truncated_mse)

        return loss


class CascadedLoss(nn.Module):
    """
    Aggregates losses from all stages of the RD-KRN model.

    Structure:
    - Stage 1 (Encoder): Weighted Cross-Entropy only.
    - Stage 2 (Refinement 1): Weighted Cross-Entropy + Weighted Truncated MSE.
    - Stage 3 (Refinement 2): Weighted Cross-Entropy + Weighted Truncated MSE.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Initialize Weighted Cross Entropy Loss
        # Weights are defined in Config (background class dampened)
        # We ensure weights are float32
        weights = Config.CLASS_WEIGHTS.float()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        # Initialize Smoothing Loss
        self.mse_loss = TruncatedMSE()
        self.mse_weight = Config.MSE_LOSS_WEIGHT

    def forward(self, predictions, targets):
        """
        Args:
            predictions: List of tensors [l1, l2, l3] from the model.
                         Each tensor has shape (Batch, Time, NumClasses).
            targets: Tensor of shape (Batch, Time) containing ground truth labels.

        Returns:
            total_loss: The aggregated scalar loss for backpropagation.
            loss_dict: Dictionary containing breakdown of loss components.
        """
        total_loss = 0.0
        loss_dict = {}

        # Flatten targets once for CrossEntropy
        # targets: (Batch, Time) -> (Batch * Time)
        targets_flat = targets.view(-1)

        # Iterate through each stage's output
        for i, logits in enumerate(predictions):
            stage_idx = i + 1

            # --- 1. Cross Entropy Loss ---
            # Reshape logits to (Batch * Time, NumClasses)
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)

            ce = self.ce_loss(logits_flat, targets_flat)

            total_loss += ce
            loss_dict[f"loss_ce_stage{stage_idx}"] = ce.item()

            # --- 2. Smoothing Loss (Refinement Stages Only) ---
            # Stage 1 (index 0) is the Encoder, which typically doesn't use smoothing
            # to allow it to be sensitive to features. Refinement stages (index > 0)
            # smooth the output.
            if i > 0:
                mse = self.mse_loss(logits)
                weighted_mse = self.mse_weight * mse

                total_loss += weighted_mse
                loss_dict[f"loss_mse_stage{stage_idx}"] = mse.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
