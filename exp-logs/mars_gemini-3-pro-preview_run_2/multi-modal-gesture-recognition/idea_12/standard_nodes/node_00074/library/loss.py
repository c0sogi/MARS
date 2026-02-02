import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CascadedSmoothLoss(nn.Module):
    """
    Implements the loss function for MD-CRCN.
    Components:
    1. Weighted Cross-Entropy Loss (Classification)
    2. Unconditional Truncated Mean Squared Error (TMSE) for Smoothing

    Applies Deep Supervision by summing losses across all 3 stages.
    """

    def __init__(self, device="cpu"):
        super(CascadedSmoothLoss, self).__init__()

        # Load weights from Config
        self.cls_weight = Config.LOSS_WEIGHT_CLS
        self.smooth_weight = Config.LOSS_WEIGHT_SMOOTH

        # Initialize Classification Loss
        # We use reduction='none' to apply the mask manually
        class_weights = Config.get_class_weights(device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def compute_masked_cls_loss(self, logits, targets, mask):
        """
        Computes masked Cross Entropy Loss.
        logits: (B, T, C)
        targets: (B, T)
        mask: (B, T)
        """
        # Flatten
        B, T, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute element-wise loss
        loss = self.ce_loss(logits_flat, targets_flat)

        # Apply mask
        loss = loss * mask_flat

        # Average over valid frames
        total_valid = mask_flat.sum()
        if total_valid > 0:
            return loss.sum() / total_valid
        else:
            return loss.sum() * 0.0

    def compute_unconditional_smoothing_loss(self, probs, mask):
        """
        Computes Unconditional Probability-Space Smoothing Loss (TMSE).
        L_smooth = Mean( ||P_t - P_{t-1}||^2 )

        probs: (B, T, C) - Softmax probabilities
        mask: (B, T)
        """
        # Calculate temporal difference: P_t - P_{t-1}
        diff = probs[:, 1:, :] - probs[:, :-1, :]  # (B, T-1, C)

        # Squared Euclidean Norm per frame
        diff_sq = torch.sum(diff**2, dim=2)  # (B, T-1)

        # Masking
        mask_sliced = mask[:, 1:]  # (B, T-1)

        # Compute masked loss
        loss = diff_sq * mask_sliced

        total_valid = mask_sliced.sum()
        if total_valid > 0:
            return loss.sum() / total_valid
        else:
            return loss.sum() * 0.0

    def forward(self, model_outputs, targets_cls, mask):
        """
        Args:
            model_outputs (dict): Dictionary containing outputs for 'stage1', 'stage2', 'stage3'.
            targets_cls (Tensor): (B, T) Long tensor of class labels.
            mask (Tensor): (B, T) Float tensor (1 for valid, 0 for padding).

        Returns:
            total_loss (Tensor): Scalar loss for backprop.
            metrics (dict): Dictionary of loss components for logging.
        """
        total_loss = 0.0
        metrics = {}

        stages = ["stage1", "stage2", "stage3"]

        for stage in stages:
            if stage not in model_outputs:
                continue

            outputs = model_outputs[stage]
            cls_logits = outputs["cls_logits"]  # (B, T, C)
            cls_probs = outputs["cls_probs"]  # (B, T, C)

            # 1. Classification Loss
            l_cls = self.compute_masked_cls_loss(cls_logits, targets_cls, mask)

            # 2. Smoothing Loss
            l_smooth = self.compute_unconditional_smoothing_loss(cls_probs, mask)

            # Weighted Sum for this stage
            stage_loss = (self.cls_weight * l_cls) + (self.smooth_weight * l_smooth)

            total_loss += stage_loss

            # Log metrics
            metrics[f"{stage}_loss"] = stage_loss.item()
            metrics[f"{stage}_cls"] = l_cls.item()
            metrics[f"{stage}_smooth"] = l_smooth.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
