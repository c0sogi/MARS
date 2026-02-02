import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for Probability-Space Smoothing.
    Penalizes rapid changes in frame-wise probabilities to encourage temporal smoothness.

    Logic:
        Loss = mean( clamp(|P_t - P_{t-1}| - threshold, min=0)^2 )
    """

    def __init__(self, threshold=0.15):
        super(TMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, probs, mask):
        """
        Args:
            probs (torch.Tensor): Probability distributions (Batch, Time, Classes).
            mask (torch.Tensor): Sequence mask (Batch, Time).
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Calculate temporal difference: |P_t - P_{t-1}|
        # Shape: (Batch, Time-1, Classes)
        diff = torch.abs(probs[:, 1:, :] - probs[:, :-1, :])

        # Apply truncation threshold
        # Only penalize differences larger than threshold
        loss = F.relu(diff - self.threshold).pow(2)

        # Adjust mask for transitions (valid only if both t and t-1 are valid)
        mask_transitions = mask[:, 1:] * mask[:, :-1]

        # Expand mask for classes: (Batch, Time-1, 1)
        mask_expanded = mask_transitions.unsqueeze(2)

        # Sum loss over valid transitions
        masked_loss = torch.sum(loss * mask_expanded)

        # Normalize by total number of valid elements (Frames * Classes)
        total_elements = torch.sum(mask_expanded) * probs.shape[2]

        if total_elements > 0:
            return masked_loss / total_elements
        return torch.tensor(0.0, device=probs.device)


class BoundaryLoss(nn.Module):
    """
    Weighted Binary Cross Entropy for Boundary Detection.
    Uses a positive weight to handle class imbalance (boundaries are sparse).
    """

    def __init__(self, pos_weight=15.0):
        super(BoundaryLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, probs, targets, mask):
        """
        Args:
            probs (torch.Tensor): Boundary probabilities (Batch, Time, 1).
            targets (torch.Tensor): Ground truth boundaries (Batch, Time).
            mask (torch.Tensor): Sequence mask (Batch, Time).
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Clamp probabilities to avoid log(0)
        probs_clamped = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        # Squeeze channel dim: (Batch, Time)
        probs_flat = probs_clamped.squeeze(2)

        # Weighted BCE Formula: -[pos_weight * y * log(p) + (1-y) * log(1-p)]
        loss = -(
            self.pos_weight * targets * torch.log(probs_flat)
            + (1.0 - targets) * torch.log(1.0 - probs_flat)
        )

        # Apply mask
        loss = loss * mask

        # Normalize by valid frames
        valid_frames = torch.sum(mask)
        if valid_frames > 0:
            return torch.sum(loss) / valid_frames
        return torch.tensor(0.0, device=probs.device)


class TotalLoss(nn.Module):
    """
    Multi-Stage Deep Supervision Loss for SG-CRCN.
    Aggregates Classification, Boundary, and Smoothing losses across all stages.
    """

    def __init__(self):
        super(TotalLoss, self).__init__()

        # Initialize Class Weights for CrossEntropy
        # Background (0) has lower weight, Gestures (1-20) have higher weight
        self.register_buffer(
            "class_weights",
            torch.tensor(Config.CLASS_WEIGHTS_INIT, dtype=torch.float32),
        )

        # Sub-losses
        self.tmse = TMSELoss(threshold=0.15)
        self.bnd_loss = BoundaryLoss(pos_weight=15.0)

        # Loss Component Weights
        self.w_cls = Config.W_CLS
        self.w_bnd = Config.W_BND
        self.w_smooth = Config.W_SMOOTH

    def forward(self, predictions, batch_targets):
        """
        Args:
            predictions (dict): Output dictionary from SG_CRCN model.
            batch_targets (dict): Batch dictionary containing 'labels', 'boundaries', 'mask'.
        Returns:
            tuple: (total_loss, metrics_dict)
        """
        targets_cls = batch_targets["labels"]  # (B, T)
        targets_bnd = batch_targets["boundaries"]  # (B, T)
        mask = batch_targets["mask"]  # (B, T)

        total_loss = 0.0
        metrics = {}

        # Iterate through all 3 stages
        for stage in [1, 2, 3]:
            # Get stage predictions
            pred_cls = predictions[f"stage{stage}_cls"]  # (B, T, C)
            pred_bnd = predictions[f"stage{stage}_bnd"]  # (B, T, 1)

            # 1. Classification Loss (Weighted NLL)
            # Clamp probs for stability
            pred_cls_clamped = torch.clamp(pred_cls, min=1e-7, max=1.0)
            log_probs = torch.log(pred_cls_clamped)

            # Transpose for NLLLoss: (B, C, T)
            log_probs_t = log_probs.transpose(1, 2)

            # Compute raw loss without reduction to apply mask
            cls_loss_raw = F.nll_loss(
                log_probs_t, targets_cls, weight=self.class_weights, reduction="none"
            )

            # Mask and reduce
            cls_loss = torch.sum(cls_loss_raw * mask) / (torch.sum(mask) + 1e-8)

            # 2. Boundary Loss
            bnd_loss_val = self.bnd_loss(pred_bnd, targets_bnd, mask)

            # 3. Smoothing Loss (T-MSE)
            smooth_loss_val = self.tmse(pred_cls, mask)

            # Aggregate Stage Loss
            stage_loss = (
                (self.w_cls * cls_loss)
                + (self.w_bnd * bnd_loss_val)
                + (self.w_smooth * smooth_loss_val)
            )

            total_loss += stage_loss

            # Record metrics
            metrics[f"s{stage}_cls"] = cls_loss.item()
            metrics[f"s{stage}_bnd"] = bnd_loss_val.item()
            metrics[f"s{stage}_smooth"] = smooth_loss_val.item()

        return total_loss, metrics
