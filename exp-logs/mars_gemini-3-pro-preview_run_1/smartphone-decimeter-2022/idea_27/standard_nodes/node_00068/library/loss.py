import torch
import torch.nn as nn


class DeepSupervisionMAELoss(nn.Module):
    """
    Calculates Mean Absolute Error (MAE) with Deep Supervision.
    Computes a weighted sum of the loss on the final output and auxiliary outputs.
    Targets for auxiliary outputs are decimated (subsampled) to match temporal resolution.
    """

    def __init__(self, final_weight=1.0, aux_weight=0.3):
        super(DeepSupervisionMAELoss, self).__init__()
        self.final_weight = final_weight
        self.aux_weight = aux_weight
        self.mae = nn.L1Loss(reduction="none")

    def decimate_target(self, tensor, stride):
        """
        Subsamples the tensor to match the temporal resolution of auxiliary heads.
        Args:
            tensor: (Batch, Time, ...)
            stride: Integer stride factor
        Returns:
            Subsampled tensor
        """
        return tensor[:, ::stride]

    def masked_mae(self, pred, target, mask):
        """
        Computes MAE ignoring masked (padded) values.
        Handles length mismatch by truncating target/mask to prediction length.

        Args:
            pred: (Batch, Time, Channels)
            target: (Batch, Time_Target, Channels)
            mask: (Batch, Time_Target)
        """
        # Align lengths: Prediction might be slightly shorter due to pooling math
        seq_len = pred.size(1)

        # Ensure we don't index out of bounds if pred is somehow longer
        if seq_len > target.size(1):
            pred = pred[:, : target.size(1)]
            seq_len = target.size(1)

        # Truncate target and mask to match prediction length
        target = target[:, :seq_len]
        mask = mask[:, :seq_len]

        # Compute element-wise L1 loss
        loss = self.mae(pred, target)  # (B, T, C)

        # Expand mask for broadcasting: (B, T) -> (B, T, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # Apply mask
        loss = loss * mask_expanded

        # Compute mean over valid elements
        # Count total valid scalar elements (Time * Channels)
        num_valid = mask_expanded.sum() * loss.size(2)

        if num_valid == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return loss.sum() / num_valid

    def forward(self, outputs, targets, mask):
        """
        Args:
            outputs: Tuple (final_pred, [aux_pred1, aux_pred2, ...])
            targets: (Batch, Time, Channels)
            mask: (Batch, Time) - True for valid data, False for padding
        """
        final_pred, aux_preds = outputs

        # 1. Calculate loss for final high-resolution output
        total_loss = self.final_weight * self.masked_mae(final_pred, targets, mask)

        # 2. Calculate loss for auxiliary heads
        for pred in aux_preds:
            # Determine stride based on resolution reduction
            # Stride = Original_Length / Feature_Length
            if pred.size(1) == 0:
                continue

            stride = targets.size(1) // pred.size(1)

            # Safety check for stride
            stride = max(1, stride)

            # Decimate target and mask to match auxiliary resolution
            target_dec = self.decimate_target(targets, stride)
            mask_dec = self.decimate_target(mask, stride)

            # Calculate weighted aux loss
            aux_loss = self.masked_mae(pred, target_dec, mask_dec)
            total_loss += self.aux_weight * aux_loss

        return total_loss
