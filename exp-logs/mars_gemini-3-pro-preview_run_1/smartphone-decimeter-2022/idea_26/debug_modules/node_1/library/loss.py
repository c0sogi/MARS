import torch
import torch.nn as nn
from library.config import Config


class DecimatedMAELoss(nn.Module):
    """
    Custom loss function for Phase-Aware Attention ResUNet.
    Calculates Mean Absolute Error (MAE) for the final output and weighted MAEs
    for decimated auxiliary outputs (Deep Supervision).
    """

    def __init__(self, config=None):
        super(DecimatedMAELoss, self).__init__()
        self.config = config if config else Config()
        self.weights = self.config.LOSS_WEIGHTS
        # self.weights format: [final_weight, aux1_weight, aux2_weight, ...]
        # aux1 corresponds to the highest resolution auxiliary head (e.g., L/2)

    def forward(self, outputs, targets, mask):
        """
        Args:
            outputs: Tensor (final) or Tuple (final, [aux_L8, aux_L4, aux_L2])
            targets: Ground truth tensor [Batch, Channels, Length]
            mask: Boolean mask [Batch, Length] indicating valid time steps

        Returns:
            total_loss: Weighted sum of losses
            metrics: Dictionary of individual loss components
        """
        if isinstance(outputs, (list, tuple)):
            final_output, aux_outputs = outputs
        else:
            final_output = outputs
            aux_outputs = []

        metrics = {}
        total_loss = 0.0

        # 1. Final Output Loss (Full Resolution)
        # Weight index 0
        final_weight = self.weights[0]
        final_loss = self._compute_masked_mae(final_output, targets, mask)
        total_loss += final_weight * final_loss
        metrics["loss_final"] = final_loss.item()

        # 2. Auxiliary Losses (Decimated Resolutions)
        # aux_outputs from model are ordered: [Deepest (L/8), ..., Shallowest (L/2)]
        # We want to map weights to resolutions:
        # weights[1] -> L/2 (Shallowest aux)
        # weights[2] -> L/4
        # ...

        if aux_outputs:
            # Reverse aux_outputs to go from High Res (L/2) to Low Res (L/8)
            # reversed_aux: [L/2, L/4, L/8]
            reversed_aux = aux_outputs[::-1]

            for i, pred in enumerate(reversed_aux):
                weight_idx = i + 1

                # Check if we have a weight defined for this aux head
                if weight_idx >= len(self.weights):
                    break

                weight = self.weights[weight_idx]
                if weight == 0:
                    continue

                # Calculate decimation factor (stride)
                # target length L, pred length L' -> stride = L // L'
                target_len = targets.shape[-1]
                pred_len = pred.shape[-1]

                if pred_len == 0:
                    continue

                stride = target_len // pred_len

                # Decimate target and mask
                # Take every s-th element
                target_dec = targets[..., ::stride]
                mask_dec = mask[..., ::stride]

                # Ensure lengths match exactly (handle potential rounding/padding edges)
                # Slice to the length of prediction
                target_dec = target_dec[..., :pred_len]
                mask_dec = mask_dec[..., :pred_len]

                # Compute Loss
                aux_loss = self._compute_masked_mae(pred, target_dec, mask_dec)
                total_loss += weight * aux_loss

                metrics[f"loss_aux_{stride}x"] = aux_loss.item()

        return total_loss, metrics

    def _compute_masked_mae(self, pred, target, mask):
        """
        Computes Mean Absolute Error ignoring masked (padded) values.

        Args:
            pred: [Batch, Channels, Length]
            target: [Batch, Channels, Length]
            mask: [Batch, Length]
        """
        # Expand mask to match channel dimension: [Batch, 1, Length] -> [Batch, Channels, Length]
        mask_expanded = mask.unsqueeze(1).expand_as(pred)

        # Absolute Error
        diff = torch.abs(pred - target)

        # Apply Mask
        masked_diff = diff * mask_expanded

        # Compute Mean
        # Count valid elements (sum of mask)
        valid_elements = mask_expanded.sum()

        if valid_elements > 0:
            return masked_diff.sum() / valid_elements
        else:
            # Return 0 loss with grad if no valid elements
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
