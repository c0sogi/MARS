import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    CLASS_WEIGHTS,
    BOUNDARY_LOSS_WEIGHT,
    TMSE_LOSS_WEIGHT,
    NUM_CLASSES,
)


class MaskedWeightedCrossEntropy(nn.Module):
    """
    Weighted Cross Entropy Loss that ignores masked (padding) tokens.
    Expects probabilities as input (since the model applies Softmax).
    """

    def __init__(self, weights=None):
        super(MaskedWeightedCrossEntropy, self).__init__()
        self.weights = weights

    def forward(self, probs, targets, mask):
        """
        Args:
            probs: (B, T, C) Softmax probabilities
            targets: (B, T) Class indices
            mask: (B, T) Binary mask (1 for valid, 0 for padding)
        """
        # Clamp probabilities for numerical stability in log
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        # Flatten dimensions
        B, T, C = probs.shape
        probs_flat = probs.view(-1, C)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Compute Log Probabilities
        log_probs = torch.log(probs_flat)

        # Ensure weights are on the correct device
        if self.weights is not None:
            if self.weights.device != probs.device:
                self.weights = self.weights.to(probs.device)

            # NLL Loss with weights (reduction='none' to apply mask manually)
            loss = F.nll_loss(
                log_probs, targets_flat, weight=self.weights, reduction="none"
            )
        else:
            loss = F.nll_loss(log_probs, targets_flat, reduction="none")

        # Apply Mask
        loss = loss * mask_flat

        # Normalize by number of valid tokens
        valid_count = mask_flat.sum()
        if valid_count > 0:
            return loss.sum() / valid_count
        else:
            return loss.sum() * 0.0


class MaskedBinaryCrossEntropy(nn.Module):
    """
    Binary Cross Entropy Loss that ignores masked (padding) tokens.
    Expects probabilities as input (since the model applies Sigmoid).
    """

    def forward(self, probs, targets, mask):
        """
        Args:
            probs: (B, T, 1) or (B, T) Sigmoid probabilities
            targets: (B, T) Binary targets
            mask: (B, T) Binary mask
        """
        if probs.dim() == 3:
            probs = probs.squeeze(-1)

        # Clamp probabilities
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

        loss = F.binary_cross_entropy(probs, targets, reduction="none")

        # Apply Mask
        loss = loss * mask

        valid_count = mask.sum()
        if valid_count > 0:
            return loss.sum() / valid_count
        else:
            return loss.sum() * 0.0


class MaskedTMSE(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for probability smoothing.
    Penalizes rapid changes in probability distributions between adjacent frames.
    """

    def forward(self, probs, mask):
        """
        Args:
            probs: (B, T, C) Softmax probabilities
            mask: (B, T) Binary mask
        """
        # Calculate MSE between frame t and t-1
        # P_t: probs[:, 1:, :]
        # P_{t-1}: probs[:, :-1, :]
        diff = probs[:, 1:, :] - probs[:, :-1, :]
        mse = torch.sum(diff**2, dim=2)  # Sum over classes -> (B, T-1)

        # Mask for transitions
        # A transition is valid only if both t and t-1 are valid
        mask_t = mask[:, 1:] * mask[:, :-1]

        loss = mse * mask_t

        valid_count = mask_t.sum()
        if valid_count > 0:
            return loss.sum() / valid_count
        else:
            return loss.sum() * 0.0


class TotalLoss(nn.Module):
    """
    Aggregates losses from all stages of the MSE-GCN model.
    L_total = Sum_stages(L_cls + w_bnd * L_bnd + w_smooth * L_smooth)
    """

    def __init__(self):
        super(TotalLoss, self).__init__()
        self.cls_criterion = MaskedWeightedCrossEntropy(weights=CLASS_WEIGHTS)
        self.bnd_criterion = MaskedBinaryCrossEntropy()
        self.smooth_criterion = MaskedTMSE()

    def forward(self, stage_outputs, cls_targets, bnd_targets, mask):
        """
        Args:
            stage_outputs: List of dicts [{'cls': (B,T,C), 'bnd': (B,T,1)}, ...]
            cls_targets: (B, T) LongTensor of class indices
            bnd_targets: (B, T) FloatTensor of boundary labels
            mask: (B, T) FloatTensor mask
        """
        total_loss = 0.0
        metrics = {}

        for i, stage_out in enumerate(stage_outputs):
            cls_prob = stage_out["cls"]
            bnd_prob = stage_out["bnd"]

            # 1. Classification Loss
            l_cls = self.cls_criterion(cls_prob, cls_targets, mask)

            # 2. Boundary Loss
            l_bnd = self.bnd_criterion(bnd_prob, bnd_targets, mask)

            # 3. Smoothing Loss
            l_smooth = self.smooth_criterion(cls_prob, mask)

            # Weighted Sum for this stage
            stage_loss = (
                l_cls + (BOUNDARY_LOSS_WEIGHT * l_bnd) + (TMSE_LOSS_WEIGHT * l_smooth)
            )
            total_loss += stage_loss

            # Record metrics
            metrics[f"s{i+1}_cls"] = l_cls.item()
            metrics[f"s{i+1}_bnd"] = l_bnd.item()
            metrics[f"s{i+1}_smooth"] = l_smooth.item()

        return total_loss, metrics
