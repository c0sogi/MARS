import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_CLASSES,
    BACKGROUND_CLASS_ID,
    LOSS_STAGE_WEIGHTS,
    BG_CLASS_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    TRUNCATION_THRESHOLD,
    DEVICE,
)


class LogSpaceTruncatedMSE(nn.Module):
    """
    Computes a Truncated Mean Squared Error on the log-probabilities of adjacent frames.
    This enforces temporal smoothness in predictions while allowing for sharp transitions
    (controlled by the truncation threshold).
    """

    def __init__(self, threshold=TRUNCATION_THRESHOLD):
        """
        Args:
            threshold (float): The maximum penalty per frame-pair. Errors exceeding
                               this value are clamped.
        """
        super(LogSpaceTruncatedMSE, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Raw output scores of shape (Batch, Time, Classes).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log-probabilities: (Batch, Time, Classes)
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate difference between frame t and t-1: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference summed over classes: (Batch, Time-1)
        mse_per_frame = diff.pow(2).sum(dim=2)

        # Truncate the error to avoid over-penalizing valid sharp transitions
        truncated_mse = torch.clamp(mse_per_frame, max=self.threshold)

        # Return mean over the batch and time dimensions
        return truncated_mse.mean()


class CascadedSmoothLoss(nn.Module):
    """
    Implements the cascaded loss function for the LG-KRN architecture.
    Combines Weighted Cross-Entropy for all stages with Log-Space Smoothing
    for the refinement stages.
    """

    def __init__(self):
        super(CascadedSmoothLoss, self).__init__()

        # Initialize Class Weights
        # Down-weight the background class to focus learning on active gestures
        class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
        class_weights[BACKGROUND_CLASS_ID] = BG_CLASS_WEIGHT

        # Base Classification Loss
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Smoothing Loss
        self.smoothing_criterion = LogSpaceTruncatedMSE(threshold=TRUNCATION_THRESHOLD)

        # Hyperparameters
        self.stage_weights = LOSS_STAGE_WEIGHTS
        self.smoothing_weight = SMOOTHING_LOSS_WEIGHT

    def forward(self, logits_list, targets):
        """
        Calculates the total weighted loss across all stages.

        Args:
            logits_list (list[torch.Tensor]): List of logits from [Stage1, Stage2, Stage3].
                                              Each tensor has shape (Batch, Time, Classes).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Time).

        Returns:
            torch.Tensor: Total scalar loss.
        """
        total_loss = 0.0

        # Flatten targets for CrossEntropy: (Batch * Time)
        targets_flat = targets.view(-1)

        for i, logits in enumerate(logits_list):
            stage_weight = self.stage_weights[i]
            batch_size, time_steps, num_classes = logits.shape

            # 1. Classification Loss (Cross Entropy)
            # Reshape logits to (Batch * Time, Classes)
            ce_loss = self.ce_criterion(logits.reshape(-1, num_classes), targets_flat)

            stage_loss = ce_loss

            # 2. Smoothing Loss (Refinement Stages Only)
            # Stage 1 (Index 0) is the initial prediction and relies purely on GRU dynamics.
            # Stages 2 & 3 (Indices 1 & 2) are TCN refinement stages where explicit smoothing is applied.
            if i > 0:
                smooth_loss = self.smoothing_criterion(logits)
                stage_loss = stage_loss + (self.smoothing_weight * smooth_loss)

            # Accumulate weighted stage loss
            total_loss += stage_weight * stage_loss

        return total_loss
