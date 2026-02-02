import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WeightedCELoss(nn.Module):
    """
    Cross Entropy Loss with class weights and sequence masking.
    Weights: 0.1 for Background, 1.0 for Gestures.
    """

    def __init__(self):
        super(WeightedCELoss, self).__init__()
        # Load weights from config
        self.weights = Config.CLASS_WEIGHTS.clone()

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, C, T)
            targets: (B, T)
            mask: (B, T)
        """
        # Ensure weights are on the correct device
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)

        # CrossEntropyLoss expects (B, C, T) logits and (B, T) targets
        criterion = nn.CrossEntropyLoss(weight=self.weights, reduction="none")

        loss = criterion(logits, targets)  # (B, T)

        # Apply mask: only valid frames contribute
        masked_loss = (loss * mask).sum()
        total_valid = mask.sum() + 1e-8

        return masked_loss / total_valid


class BoundaryLoss(nn.Module):
    """
    Binary Cross Entropy Loss for boundary detection with sequence masking.
    Uses sharp targets (0 or 1).
    """

    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):
        """
        Args:
            logits: (B, 1, T)
            targets: (B, T) - FloatTensor
            mask: (B, T)
        """
        # Squeeze channel dim: (B, 1, T) -> (B, T)
        logits = logits.squeeze(1)

        loss = self.criterion(logits, targets)  # (B, T)

        masked_loss = (loss * mask).sum()
        total_valid = mask.sum() + 1e-8

        return masked_loss / total_valid


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error (T-MSE) for smoothing.
    Applied to Softmax probabilities (not Log-Softmax).
    Explicitly UNCLAMPED as per instructions.
    """

    def __init__(self):
        super(TMSELoss, self).__init__()

    def forward(self, logits, mask):
        """
        Args:
            logits: (B, C, T)
            mask: (B, T)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, T)

        # Compute temporal difference: P(t) - P(t-1)
        # Slice to get adjacent pairs
        # probs[:, :, 1:] is t=1..T
        # probs[:, :, :-1] is t=0..T-1
        diff = probs[:, :, 1:] - probs[:, :, :-1]  # (B, C, T-1)

        # Squared Error
        loss = diff.pow(2).sum(dim=1)  # Sum over classes -> (B, T-1)

        # Adjust mask for T-1 length
        # A transition is valid if both t and t-1 are valid
        mask_t = mask[:, 1:]
        mask_t_minus_1 = mask[:, :-1]
        valid_transitions = mask_t * mask_t_minus_1  # (B, T-1)

        masked_loss = (loss * valid_transitions).sum()
        total_transitions = valid_transitions.sum() + 1e-8

        return masked_loss / total_transitions


def compute_total_loss(model_outputs, cls_targets, bnd_targets, mask):
    """
    Computes the aggregated loss for Deep Supervision across all stages.

    Args:
        model_outputs: Dict with keys 'stage1', 'stage2', 'stage3'.
                       Values are tuples (cls_logits, bnd_logits).
        cls_targets: (B, T) LongTensor
        bnd_targets: (B, T) FloatTensor
        mask: (B, T) Bool/Float Tensor

    Returns:
        total_loss: Scalar Tensor
        metrics: Dict of float values for logging
    """
    # Instantiate loss functions
    ce_loss_fn = WeightedCELoss()
    bnd_loss_fn = BoundaryLoss()
    tmse_loss_fn = TMSELoss()

    total_loss = 0.0
    metrics = {}

    # Iterate over available stages
    # Typically: stage1 (Encoder), stage2 (Refine 1), stage3 (Refine 2)
    for stage_name, (cls_logits, bnd_logits) in model_outputs.items():

        # 1. Classification Loss
        l_cls = ce_loss_fn(cls_logits, cls_targets, mask)

        # 2. Boundary Loss
        l_bnd = bnd_loss_fn(bnd_logits, bnd_targets, mask)

        # 3. Smoothing Loss (only on class logits)
        l_smooth = tmse_loss_fn(cls_logits, mask)

        # Weighted Sum
        stage_loss = (
            Config.LOSS_WEIGHT_CLS * l_cls
            + Config.LOSS_WEIGHT_BND * l_bnd
            + Config.LOSS_WEIGHT_SMOOTH * l_smooth
        )

        total_loss += stage_loss

        # Log metrics for this stage
        metrics[f"{stage_name}_loss"] = stage_loss.item()
        metrics[f"{stage_name}_cls"] = l_cls.item()
        metrics[f"{stage_name}_bnd"] = l_bnd.item()
        metrics[f"{stage_name}_smooth"] = l_smooth.item()

    return total_loss, metrics
