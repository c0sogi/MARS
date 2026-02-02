import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiScaleMAELoss(nn.Module):
    """
    Computes the weighted sum of Mean Absolute Errors (MAE) for multi-scale predictions.

    The model outputs a list of predictions [P_0, P_1, ..., P_n] corresponding to different
    temporal resolutions (scales). The dataset provides a corresponding list of ground truth
    targets [T_0, T_1, ..., T_n].

    T_0 is the original resolution. T_i is derived from T_{i-1} via average pooling.
    The mask must also be downsampled to match the resolution of each scale.
    """

    def __init__(self):
        super(MultiScaleMAELoss, self).__init__()
        self.weights = Config.LOSS_WEIGHTS
        # We use reduction='none' so we can apply the mask before averaging
        self.criterion = nn.L1Loss(reduction="none")
        self.epsilon = 1e-8

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions (list[torch.Tensor]): List of predicted tensors (B, C, L_i).
            targets (list[torch.Tensor]): List of target tensors (B, C, L_i).
            mask (torch.Tensor): Validity mask tensor for the original resolution (B, L_0).

        Returns:
            torch.Tensor: Scalar weighted loss.
        """
        total_loss = 0.0

        # Ensure mask is (B, 1, L) for broadcasting with (B, C, L)
        if mask.dim() == 2:
            current_mask = mask.unsqueeze(1)
        else:
            current_mask = mask

        # Iterate over scales
        # predictions[0] and targets[0] are the highest resolution (original scale)
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            weight = self.weights[i] if i < len(self.weights) else 1.0

            # Downsample mask for auxiliary heads if necessary
            # Assuming targets were downsampled by factor of 2 at each step in dataset.py
            if i > 0:
                # Use avg_pool1d to match the target downsampling logic.
                # This results in a fractional mask value (0.0 to 1.0) indicating
                # how much of the pooled window was valid.
                current_mask = F.avg_pool1d(current_mask, kernel_size=2, stride=2)

            # Ensure dimensions match (handle potential rounding errors in pooling if any)
            if current_mask.shape[-1] != pred.shape[-1]:
                # In case of odd lengths or padding mismatches, interpolate mask to match
                current_mask = F.interpolate(
                    current_mask, size=pred.shape[-1], mode="nearest"
                )

            # Compute element-wise absolute error
            loss_elementwise = self.criterion(pred, target)

            # Apply mask
            masked_loss = loss_elementwise * current_mask

            # Compute mean over valid elements
            # Sum over all dimensions and divide by sum of mask * channels
            # mask sum needs to be multiplied by C because mask is broadcasted over C channels
            num_valid = current_mask.sum() * pred.shape[1]

            if num_valid > 0:
                scale_loss = masked_loss.sum() / (num_valid + self.epsilon)
            else:
                scale_loss = 0.0

            total_loss += weight * scale_loss

        return total_loss
