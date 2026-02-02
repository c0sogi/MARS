import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BoundaryAwareLoss(nn.Module):
    """
    Implements the composite loss function for the BA-MD-CRCN model.
    Components:
    1. Weighted Cross-Entropy Loss (Classification)
    2. Weighted Binary Cross-Entropy Loss (Boundary Detection)
    3. Adaptive Probability-Space Smoothing (Smoothness)

    Applies Deep Supervision by summing losses across all 3 stages.
    """

    def __init__(self, device="cpu"):
        super(BoundaryAwareLoss, self).__init__()

        # Load weights from Config
        self.cls_weight = Config.LOSS_WEIGHT_CLS
        self.bnd_weight = Config.LOSS_WEIGHT_BND
        self.smooth_weight = Config.LOSS_WEIGHT_SMOOTH

        # Initialize Classification Loss
        # We use reduction='none' to apply the mask manually
        class_weights = Config.get_class_weights(device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

        # Initialize Boundary Loss
        # We use reduction='none' to apply the mask manually
        pos_weight = Config.get_boundary_pos_weight(device=device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def compute_masked_cls_loss(self, logits, targets, mask):
        """
        Computes masked Cross Entropy Loss.
        logits: (B, T, C) - Raw scores (before Softmax) - but CrossEntropyLoss expects (B, C, T) or (B, T, C) depending on implementation.
                PyTorch CrossEntropyLoss with (B, C, T) is standard for 1D, but here we likely have (B, T, C).
                We will flatten to (B*T, C) for simplicity.
        targets: (B, T) - Class indices
        mask: (B, T) - Valid frames
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

    def compute_masked_bnd_loss(self, logits, targets, mask):
        """
        Computes masked BCE Loss for boundaries.
        logits: (B, T, 1)
        targets: (B, T) or (B, T, 1)
        mask: (B, T)
        """
        # Ensure shapes match
        if logits.dim() == 3 and logits.shape[2] == 1:
            logits = logits.squeeze(2)  # (B, T)

        # Flatten
        logits_flat = logits.reshape(-1)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss = self.bce_loss(logits_flat, targets_flat)
        loss = loss * mask_flat

        total_valid = mask_flat.sum()
        if total_valid > 0:
            return loss.sum() / total_valid
        else:
            return loss.sum() * 0.0

    def compute_adaptive_smoothing_loss(self, probs, bnd_targets, mask):
        """
        Computes Adaptive Probability-Space Smoothing Loss.
        L_smooth = Mean( (1 - Y_bnd) * ||P_t - P_{t-1}||^2 )

        probs: (B, T, C) - Softmax probabilities
        bnd_targets: (B, T) - Ground truth boundary (1.0 for transition, 0.0 otherwise)
        mask: (B, T)
        """
        # Calculate temporal difference: P_t - P_{t-1}
        # We slice from 1 to T
        diff = probs[:, 1:, :] - probs[:, :-1, :]  # (B, T-1, C)

        # Squared Euclidean Norm per frame
        # Sum over classes (dim 2)
        diff_sq = torch.sum(diff**2, dim=2)  # (B, T-1)

        # Align targets and mask to T-1 (dropping the first frame for the diff)
        # Note: diff[t] corresponds to change between t and t+1 in original 0..T indexing?
        # No, diff[:, i] is probs[:, i+1] - probs[:, i].
        # So it corresponds to the transition arriving at frame i+1.
        # We should use bnd_targets[:, 1:] which indicates if frame i+1 is a boundary.

        bnd_targets_sliced = bnd_targets[:, 1:]  # (B, T-1)
        mask_sliced = mask[:, 1:]  # (B, T-1)

        # Adaptive Weighting: (1 - Y_bnd)
        # If Y_bnd is 1 (transition), weight is 0 -> No smoothing penalty.
        # If Y_bnd is 0 (continuity), weight is 1 -> Enforce smoothness.
        adaptive_weight = 1.0 - bnd_targets_sliced

        # Compute weighted loss
        loss = diff_sq * adaptive_weight * mask_sliced

        total_valid = mask_sliced.sum()
        if total_valid > 0:
            return loss.sum() / total_valid
        else:
            return loss.sum() * 0.0

    def forward(self, model_outputs, targets_cls, targets_bnd, mask):
        """
        Args:
            model_outputs (dict): Dictionary containing outputs for 'stage1', 'stage2', 'stage3'.
                                  Each value is a dict with 'cls_logits', 'bnd_logits', 'cls_probs'.
            targets_cls (Tensor): (B, T) Long tensor of class labels.
            targets_bnd (Tensor): (B, T) Float tensor of boundary labels (0 or 1).
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
            bnd_logits = outputs["bnd_logits"]  # (B, T, 1)
            cls_probs = outputs["cls_probs"]  # (B, T, C)

            # 1. Classification Loss
            l_cls = self.compute_masked_cls_loss(cls_logits, targets_cls, mask)

            # 2. Boundary Loss
            l_bnd = self.compute_masked_bnd_loss(bnd_logits, targets_bnd, mask)

            # 3. Smoothing Loss
            l_smooth = self.compute_adaptive_smoothing_loss(
                cls_probs, targets_bnd, mask
            )

            # Weighted Sum for this stage
            stage_loss = (
                (self.cls_weight * l_cls)
                + (self.bnd_weight * l_bnd)
                + (self.smooth_weight * l_smooth)
            )

            total_loss += stage_loss

            # Log metrics
            metrics[f"{stage}_loss"] = stage_loss.item()
            metrics[f"{stage}_cls"] = l_cls.item()
            metrics[f"{stage}_bnd"] = l_bnd.item()
            metrics[f"{stage}_smooth"] = l_smooth.item()

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
