import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class LogSpaceSmoothingLoss(nn.Module):
    """
    Computes a truncated MSE loss on the difference between log-probabilities
    of adjacent frames to enforce temporal smoothness.
    """

    def __init__(self, threshold=config.SMOOTHING_THRESHOLD):
        """
        Args:
            threshold (float): The maximum allowed difference in log-space before clamping.
                               Prevents gradients from exploding due to sharp transitions.
        """
        super(LogSpaceSmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Tensor of shape (Batch, Time, NumClasses).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(logits, dim=2)

        # Calculate temporal difference: log_p[t] - log_p[t-1]
        # Slice to get t=1..T and t=0..T-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Truncate gradients via clamping (Truncated MSE)
        # This allows sharp transitions (large diffs) to exist without penalizing them quadratically forever
        diff = torch.clamp(diff, min=-self.threshold, max=self.threshold)

        # Mean Squared Error on the clamped differences
        loss = torch.mean(diff**2)

        return loss


class CascadedLoss(nn.Module):
    """
    Aggregates losses from all three stages of the ANG-KN model.
    Combines Weighted Cross-Entropy with Log-Space Smoothing for refinement stages.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()

        # Register class weights as a buffer so it moves to device with the module
        # config.CLASS_WEIGHTS is already a tensor in the config file
        self.register_buffer("class_weights", config.CLASS_WEIGHTS)

        # Weighted Cross Entropy Loss
        # We use the buffer 'self.class_weights' which will be on the correct device
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # Temporal Smoothing Loss
        self.smooth_loss = LogSpaceSmoothingLoss(threshold=config.SMOOTHING_THRESHOLD)

        # Weights for Deep Supervision
        self.stage_weights = config.LOSS_WEIGHTS
        self.smooth_weight = config.SMOOTHING_LOSS_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing logits for 'stage1', 'stage2', 'stage3'.
                            Each value is a tensor of shape (Batch, Time, NumClasses).
            targets (torch.Tensor): Ground truth labels of shape (Batch, Time).

        Returns:
            tuple: (total_loss, loss_dict)
                   total_loss is a scalar Tensor for backprop.
                   loss_dict is a dictionary of float values for logging.
        """
        total_loss = 0.0
        loss_dict = {}

        for stage_name, logits in outputs.items():
            # --- Cross Entropy Loss ---
            # PyTorch CrossEntropyLoss expects (Batch, NumClasses, Time) for multi-dimensional inputs
            # Input logits are (Batch, Time, NumClasses), so we permute.
            ce_input = logits.permute(0, 2, 1)

            stage_ce = self.ce_loss(ce_input, targets)

            # Apply stage weight
            weight = self.stage_weights.get(stage_name, 1.0)
            weighted_ce = stage_ce * weight

            # Accumulate
            total_loss += weighted_ce
            loss_dict[f"{stage_name}_ce"] = stage_ce.item()

            # --- Smoothing Loss (Refinement Stages Only) ---
            if stage_name in ["stage2", "stage3"]:
                stage_smooth = self.smooth_loss(logits)
                weighted_smooth = stage_smooth * self.smooth_weight

                total_loss += weighted_smooth
                loss_dict[f"{stage_name}_smooth"] = stage_smooth.item()

        return total_loss, loss_dict
