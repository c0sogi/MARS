import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    NUM_CLASSES,
    LOSS_WEIGHT_CLS_BG,
    LOSS_WEIGHT_CLS_FG,
    TMSE_WEIGHT,
)


class TruncatedMSE(nn.Module):
    """
    Computes the Truncated Mean Squared Error (T-MSE) on Softmax probabilities
    to enforce temporal smoothness.

    As per specific task instructions:
    - Applies to Softmax probabilities (not Log-Softmax).
    - Does not clamp the loss (threshold is effectively infinite/None).
    """

    def __init__(self, threshold=None):
        super(TruncatedMSE, self).__init__()
        self.threshold = threshold

    def forward(self, logits, mask):
        """
        Args:
            logits: (B, C, T) Class logits.
            mask: (B, T) Boolean/Float mask indicating valid frames.
        Returns:
            Scalar loss.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Calculate squared difference between adjacent frames: (P_t - P_{t-1})^2
        # Shape: (B, C, T-1)
        diff = (probs[:, :, 1:] - probs[:, :, :-1]).pow(2)

        # Sum over classes -> (B, T-1)
        mse = diff.sum(dim=1)

        # Apply truncation if a threshold is provided (default None = Unclamped)
        if self.threshold is not None:
            mse = torch.clamp(mse, min=0, max=self.threshold)

        # Masking
        # We need to mask the transitions. If frame t is padding, transition t-1 -> t is invalid.
        # mask[:, 1:] corresponds to frames t=1..T-1.
        valid_mask = mask[:, 1:].float()

        # Compute mean over valid transitions
        # Add epsilon to denominator to prevent division by zero
        loss = (mse * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss


class CombinedLoss(nn.Module):
    """
    Aggregates Classification, Boundary, and Smoothing losses across multiple stages
    with Deep Supervision.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        # 1. Classification Loss (Weighted Cross Entropy)
        # Weights: 0.1 for Background (index 0), 1.0 for Gestures (indices 1-20)
        weights = torch.ones(NUM_CLASSES)
        weights[0] = LOSS_WEIGHT_CLS_BG
        weights[1:] = LOSS_WEIGHT_CLS_FG
        self.register_buffer("class_weights", weights)

        self.ce_loss = nn.CrossEntropyLoss(weight=weights, reduction="none")

        # 2. Boundary Loss (Binary Cross Entropy)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # 3. Smoothing Loss (T-MSE)
        self.tmse_loss = TruncatedMSE(threshold=None)  # Unclamped as per instructions
        self.tmse_weight = TMSE_WEIGHT

    def forward(self, predictions, targets, boundaries, mask):
        """
        Args:
            predictions: List of tensors, one per stage.
                         Each tensor shape: (B, NUM_CLASSES + 1, T)
            targets: (B, T) LongTensor of class labels.
            boundaries: (B, T) FloatTensor of boundary targets (0 or 1).
            mask: (B, T) Bool/Float tensor of valid frames.
        """
        total_loss = 0.0
        metrics = {}

        # Ensure mask is float for calculations
        mask_float = mask.float()
        num_valid = mask_float.sum() + 1e-8

        # Handle case where predictions is a single tensor instead of a list
        if not isinstance(predictions, list):
            predictions = [predictions]

        for stage_idx, pred in enumerate(predictions):
            # Split prediction into Class and Boundary heads
            # Input shape: (B, NUM_CLASSES + 1, T)
            # Class logits: (B, NUM_CLASSES, T)
            cls_logits = pred[:, :NUM_CLASSES, :]
            # Boundary logits: (B, 1, T) -> Squeeze to (B, T)
            bnd_logits = pred[:, NUM_CLASSES:, :].squeeze(1)

            # --- Classification Loss ---
            ce = self.ce_loss(cls_logits, targets)  # Output: (B, T)
            loss_cls = (ce * mask_float).sum() / num_valid

            # --- Boundary Loss ---
            bce = self.bce_loss(bnd_logits, boundaries)  # Output: (B, T)
            loss_bnd = (bce * mask_float).sum() / num_valid

            # --- Smoothing Loss ---
            loss_smooth = self.tmse_loss(cls_logits, mask)

            # --- Aggregate ---
            # Sum losses for this stage
            stage_loss = loss_cls + loss_bnd + (self.tmse_weight * loss_smooth)
            total_loss += stage_loss

            # Record metrics for debugging/monitoring
            metrics[f"s{stage_idx}_cls"] = loss_cls.item()
            metrics[f"s{stage_idx}_bnd"] = loss_bnd.item()
            metrics[f"s{stage_idx}_sm"] = loss_smooth.item()

        return total_loss, metrics
