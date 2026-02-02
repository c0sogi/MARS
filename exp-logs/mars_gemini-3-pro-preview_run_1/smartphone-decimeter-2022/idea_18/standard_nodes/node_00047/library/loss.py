import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DeepSupervisionLoss(nn.Module):
    """
    Calculates the weighted sum of Mean Absolute Error (MAE) losses for deep supervision.
    Handles dynamic downsampling of ground truth targets and masks to match auxiliary head resolutions.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (list): List of weights for [Final, Aux1, Aux2, ...].
                            Defaults to Config.DEEP_SUPERVISION_WEIGHTS.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.weights = (
            weights if weights is not None else Config.DEEP_SUPERVISION_WEIGHTS
        )
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets, mask=None):
        """
        Args:
            preds: Tuple/List of model outputs (final_head, aux_head1, aux_head2, ...).
                   Each tensor has shape (B, C, T).
            targets: Ground truth tensor of shape (B, C, T).
            mask: Validity mask tensor of shape (B, T). 1.0 for valid, 0.0 for padding.

        Returns:
            total_loss: Weighted sum of losses.
        """
        # Ensure preds is iterable (handle case where model returns single tensor in eval)
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        total_loss = 0.0

        # Ensure mask has channel dimension for broadcasting: (B, T) -> (B, 1, T)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)

        for i, pred in enumerate(preds):
            # Get weight for this head, default to 0 if not specified
            weight = self.weights[i] if i < len(self.weights) else 0.0

            if weight == 0.0:
                continue

            # Match target resolution to prediction resolution
            if pred.shape[2] != targets.shape[2]:
                # Downsample targets using Average Pooling as per requirements
                # Targets are (B, C, T), pool operates on last dim (Time)
                curr_target = F.adaptive_avg_pool1d(targets, output_size=pred.shape[2])

                # Downsample mask using Nearest Neighbor to maintain binary nature (valid/invalid)
                if mask is not None:
                    curr_mask = F.interpolate(mask, size=pred.shape[2], mode="nearest")
                else:
                    curr_mask = None
            else:
                curr_target = targets
                curr_mask = mask

            # Calculate element-wise L1 loss
            loss = self.l1(pred, curr_target)

            # Apply mask if present
            if curr_mask is not None:
                # curr_mask is (B, 1, T_scaled), loss is (B, C, T_scaled)
                masked_loss = loss * curr_mask

                # Calculate mean over valid elements
                # Number of valid elements = sum(mask) * channels
                num_valid = curr_mask.sum() * loss.shape[1]

                if num_valid > 0:
                    term_loss = masked_loss.sum() / num_valid
                else:
                    # Handle case with no valid elements (e.g. all padding)
                    term_loss = 0.0 * masked_loss.sum()
            else:
                term_loss = loss.mean()

            total_loss += weight * term_loss

        return total_loss
