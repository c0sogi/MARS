import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CLASS_WEIGHTS, TMSE_THRESHOLD, LOSS_WEIGHTS


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss that ignores padded frames based on a binary mask.
    Applies class balancing weights.
    """

    def __init__(self, weight=None):
        super(MaskedCrossEntropyLoss, self).__init__()
        # weight should be a tensor of class weights.
        # We register it as a buffer so it moves to device automatically.
        self.register_buffer("weight", weight)

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, T, C)
            targets: (B, T) - Integer class indices
            mask: (B, T) - Binary mask (1 for valid, 0 for padding)
        """
        # Flatten temporal and batch dimensions
        logits_flat = logits.view(-1, logits.size(-1))  # (B*T, C)
        targets_flat = targets.view(-1)  # (B*T)
        mask_flat = mask.view(-1)  # (B*T)

        # Calculate CE loss without reduction
        loss = F.cross_entropy(
            logits_flat, targets_flat, weight=self.weight, reduction="none"
        )

        # Apply mask
        masked_loss = loss * mask_flat

        # Normalize by number of valid tokens
        # Add epsilon to avoid division by zero
        return masked_loss.sum() / (mask_flat.sum() + 1e-8)


class MaskedBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Logits that ignores padded frames.
    Used for boundary detection.
    """

    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, T, 1)
            targets: (B, T, 1) or (B, T)
            mask: (B, T)
        """
        # Ensure targets match logits shape
        if targets.dim() == 2:
            targets = targets.unsqueeze(2)

        # Flatten
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        mask_flat = mask.view(-1)

        loss = self.bce(logits_flat, targets_flat)
        masked_loss = loss * mask_flat

        return masked_loss.sum() / (mask_flat.sum() + 1e-8)


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error Loss for temporal smoothing.
    Penalizes small changes in probability distributions between adjacent frames,
    but truncates the loss for large changes (boundaries).
    """

    def __init__(self, threshold=TMSE_THRESHOLD):
        super(TMSELoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, probs, mask):
        """
        Args:
            probs: (B, T, C) - Softmax probabilities
            mask: (B, T)
        """
        # Calculate diff between t and t-1
        # We slice from 1: to T and 0: to T-1
        current_probs = probs[:, 1:, :]
        prev_probs = probs[:, :-1, :]

        # Corresponding mask (valid if both t and t-1 are valid)
        # Usually mask is 1,1,1,0,0.
        # mask[:, 1:] checks t. mask[:, :-1] checks t-1.
        valid_transitions = mask[:, 1:] * mask[:, :-1]

        # Squared Error per element
        # (B, T-1, C)
        squared_diff = (current_probs - prev_probs) ** 2

        # Sum over classes to get total squared variation per frame transition
        # (B, T-1)
        frame_mse = squared_diff.sum(dim=2)

        # Truncate: min(mse, threshold^2)
        # We square the threshold because we are comparing to squared differences
        threshold_sq = self.threshold**2
        truncated_mse = torch.clamp(frame_mse, max=threshold_sq)

        # Apply mask
        masked_loss = truncated_mse * valid_transitions

        return masked_loss.sum() / (valid_transitions.sum() + 1e-8)


class DeepSupervisionLoss(nn.Module):
    """
    Aggregates losses from all stages of the MCAG-CN model.
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()
        self.cls_criterion = MaskedCrossEntropyLoss(weight=CLASS_WEIGHTS)
        self.bnd_criterion = MaskedBCELoss()
        self.smooth_criterion = TMSELoss(threshold=TMSE_THRESHOLD)

        self.weights = LOSS_WEIGHTS

    def forward(self, model_outputs, targets_cls, targets_bnd, mask):
        """
        Args:
            model_outputs: List of dicts from MCAGCN forward pass.
                           Each dict contains 'cls_logits', 'bnd_logits', 'cls_probs', 'bnd_probs'.
            targets_cls: (B, T) - Ground truth class indices
            targets_bnd: (B, T) - Ground truth boundary (0 or 1)
            mask: (B, T) - Valid frame mask
        """
        total_loss = 0.0
        metrics = {}

        for i, stage_out in enumerate(model_outputs):
            # Classification Loss
            l_cls = self.cls_criterion(stage_out["cls_logits"], targets_cls, mask)

            # Boundary Loss
            l_bnd = self.bnd_criterion(stage_out["bnd_logits"], targets_bnd, mask)

            # Smoothing Loss (applied to probabilities)
            l_smooth = self.smooth_criterion(stage_out["cls_probs"], mask)

            # Weighted Sum
            stage_loss = (
                self.weights["cls"] * l_cls
                + self.weights["bnd"] * l_bnd
                + self.weights["smooth"] * l_smooth
            )

            total_loss += stage_loss

            # Store metrics for the final stage (or all stages if needed for debug)
            # We prefix with stage index
            metrics[f"s{i+1}_loss"] = stage_loss.item()
            metrics[f"s{i+1}_cls"] = l_cls.item()
            metrics[f"s{i+1}_bnd"] = l_bnd.item()
            metrics[f"s{i+1}_smooth"] = l_smooth.item()

        return total_loss, metrics
