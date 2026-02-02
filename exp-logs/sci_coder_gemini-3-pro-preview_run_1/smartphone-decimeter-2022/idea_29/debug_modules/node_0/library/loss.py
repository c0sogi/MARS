import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiResolutionMAELoss(nn.Module):
    """
    Computes the weighted Mean Absolute Error for multi-resolution outputs.
    Handles decimation of ground truth targets for auxiliary low-resolution heads.
    """

    def __init__(self):
        super().__init__()
        self.weights = Config.LOSS_WEIGHTS
        self.factors = Config.RESOLUTION_FACTORS
        self.max_factor = max(self.factors)

    def forward(self, preds, targets, mask):
        """
        Args:
            preds (list[torch.Tensor]): List of model outputs from different streams.
                                        Stream 0 is [B, 2, T].
                                        Stream i>0 is [B, 2, T_padded // factor].
            targets (torch.Tensor): Ground truth targets [B, 2, T].
            mask (torch.Tensor): Boolean mask [B, T], True for valid time steps.

        Returns:
            torch.Tensor: Scalar weighted loss.
        """
        total_loss = 0.0

        # Original sequence length
        T = targets.shape[-1]

        # Calculate padding length to match model's internal padding logic
        # The model pads input to be divisible by max_factor (16)
        pad_len = (self.max_factor - (T % self.max_factor)) % self.max_factor

        # Prepare padded versions of targets and mask for auxiliary heads
        # We pad targets with 0 and mask with False (invalid)
        if pad_len > 0:
            targets_padded = F.pad(targets, (0, pad_len), value=0.0)
            mask_padded = F.pad(mask, (0, pad_len), value=False)
        else:
            targets_padded = targets
            mask_padded = mask

        for i, pred in enumerate(preds):
            weight = self.weights[i]
            factor = self.factors[i]

            if i == 0:
                # High-Res Stream: Model output is explicitly cropped to original length T
                curr_target = targets
                curr_mask = mask
            else:
                # Aux Streams: Model output corresponds to T_padded // factor
                # We decimate (subsample) the padded targets and mask
                curr_target = targets_padded[..., ::factor]
                curr_mask = mask_padded[..., ::factor]

            # Safety check: Ensure shapes match along time dimension
            if pred.shape[-1] != curr_target.shape[-1]:
                # Crop to the minimum length if there's a mismatch (e.g. edge cases in padding math)
                min_len = min(pred.shape[-1], curr_target.shape[-1])
                pred = pred[..., :min_len]
                curr_target = curr_target[..., :min_len]
                curr_mask = curr_mask[..., :min_len]

            # Compute Absolute Error: [B, 2, T_i]
            abs_diff = torch.abs(pred - curr_target)

            # Apply Mask: [B, T_i] -> [B, 1, T_i]
            # Broadcast mask over the coordinate dimension (North/East)
            mask_expanded = curr_mask.unsqueeze(1)

            # Sum errors only at valid positions
            masked_sum = (abs_diff * mask_expanded).sum()

            # Count valid elements (multiply by 2 for both coordinates)
            valid_count = mask_expanded.sum() * 2

            # Compute mean loss for this stream
            if valid_count > 0:
                stream_loss = masked_sum / valid_count
            else:
                # Fallback if batch has no valid data (unlikely with proper batching)
                stream_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

            # Add weighted loss
            total_loss += weight * stream_loss

        return total_loss
