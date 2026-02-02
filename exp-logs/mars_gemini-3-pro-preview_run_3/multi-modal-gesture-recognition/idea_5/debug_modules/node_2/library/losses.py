import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SmoothingLoss(nn.Module):
    """
    Computes the Truncated Mean Squared Error (TMSE) between log-probabilities
    of adjacent frames. This loss encourages temporal smoothness in the predictions,
    penalizing high-frequency jitter.

    Formula:
        L_smooth = mean( clamp( (log(p_t) - log(p_{t-1}))^2, max=threshold ) )
    """

    def __init__(self, threshold=16.0):
        """
        Args:
            threshold (float): The maximum value for the squared difference before clamping.
                               This prevents outliers (e.g., true boundaries) from dominating the loss.
                               Default is 16.0, commonly used in MS-TCN literature.
        """
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits (torch.Tensor): Tensor of shape (Batch, Classes, Time) or (Batch, Time, Classes).
                                   Raw output scores (before softmax).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure logits are in (Batch, Classes, Time) for consistency
        if (
            logits.dim() == 3
            and logits.size(1) != Config.NUM_CLASSES
            and logits.size(2) == Config.NUM_CLASSES
        ):
            # Permute from (B, T, C) to (B, C, T)
            logits = logits.permute(0, 2, 1)

        # Compute log probabilities: log(softmax(x))
        # dim=1 corresponds to Classes in (B, C, T)
        log_probs = F.log_softmax(logits, dim=1)

        # Compute difference between adjacent frames along the Time dimension (dim=2)
        # diff[t] = log_p[t] - log_p[t-1]
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

        # Squared difference
        sq_diff = diff.pow(2)

        # Truncate (clamp) the squared error to avoid penalizing sharp transitions too heavily
        # This allows the model to change state when necessary (true gesture boundaries)
        clamped_diff = torch.clamp(sq_diff, min=0, max=self.threshold)

        # Return the mean over all batches, classes, and time steps
        return torch.mean(clamped_diff)


class CombinedLoss(nn.Module):
    """
    Aggregates losses from the multi-stage architecture.

    Components:
    1. Weighted Cross-Entropy Loss for Stage 1 (Dual-Stream Encoder).
    2. Weighted Cross-Entropy Loss for Stage 2 (Refinement Module).
    3. Smoothing Loss for Stage 2 (Refinement Module) to enforce continuity.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        # Initialize Class Weights to handle imbalance (Background dominance)
        # Shape: (NUM_CLASSES,)
        weights = torch.ones(Config.NUM_CLASSES)
        weights[Config.BACKGROUND_CLASS_ID] = Config.BG_WEIGHT

        # Register as buffer so it moves to device automatically with the module
        self.register_buffer("class_weights", weights)

        # Weighted Cross Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        # Smoothing Loss
        self.smooth_loss = SmoothingLoss()
        self.smooth_weight = Config.SMOOTHING_LOSS_WEIGHT

    def forward(self, stage1_logits, stage2_logits, targets):
        """
        Args:
            stage1_logits (torch.Tensor): Output from Dual-Stream Encoder. Shape (B, C, T).
            stage2_logits (torch.Tensor): Output from Refinement Module. Shape (B, C, T).
            targets (torch.Tensor): Ground truth labels. Shape (B, T).

        Returns:
            tuple: (total_loss, loss_ce_stage1, loss_ce_stage2, loss_smooth)
        """
        # Cross Entropy expects (B, C, T) logits and (B, T) targets
        loss_ce1 = self.ce_loss(stage1_logits, targets)
        loss_ce2 = self.ce_loss(stage2_logits, targets)

        # Smoothing loss is only applied to the final output (Stage 2)
        # to ensure the final predictions are smooth.
        loss_smooth = self.smooth_loss(stage2_logits)

        # Combine losses
        # L = CE1 + CE2 + lambda * Smooth
        total_loss = loss_ce1 + loss_ce2 + (self.smooth_weight * loss_smooth)

        return total_loss, loss_ce1, loss_ce2, loss_smooth
