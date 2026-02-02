import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CascadedDeepSupervisionLoss(nn.Module):
    """
    Loss function for Cascaded 1D ResUNet with Scaled Deep Supervision.
    Calculates MAE for the final output and auxiliary outputs.
    Ground truth targets are downsampled to match the resolution of auxiliary heads.
    """

    def __init__(self):
        super().__init__()
        self.aux_weight = Config.AUX_LOSS_WEIGHT
        # Use reduction='none' to manually handle masking
        self.criterion = nn.L1Loss(reduction="none")

    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: Tuple (final_output, [aux_pred1, aux_pred2, ...])
                         final_output shape: (B, C, L)
                         aux_preds shapes: (B, C, L_scaled)
            targets: Ground truth tensor of shape (B, C, L)
            mask: Validity mask of shape (B, L) indicating valid time steps (1.0) vs padding (0.0)

        Returns:
            total_loss: Scalar tensor
            metrics: Dictionary of loss components
        """
        final_pred, aux_preds = predictions

        # Ensure mask has channel dimension for broadcasting: (B, L) -> (B, 1, L)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)

        # --- 1. Final Output Loss ---
        # Calculate element-wise MAE
        loss_final_elementwise = self.criterion(final_pred, targets)

        # Apply mask
        masked_loss_final = loss_final_elementwise * mask

        # Normalize by total valid elements (Time steps * Channels)
        # mask.sum() gives total valid time steps across batch
        valid_elements_final = mask.sum() * final_pred.size(1) + 1e-8
        term_final = masked_loss_final.sum() / valid_elements_final

        # --- 2. Auxiliary Losses (Scaled Deep Supervision) ---
        loss_aux_total = 0.0

        # Iterate over auxiliary outputs (e.g., from Stage 1 decoder layers)
        for aux_pred in aux_preds:
            # Determine the temporal resolution of the auxiliary output
            target_size = aux_pred.size(2)

            # Downsample targets and mask to match auxiliary resolution
            # 'area' interpolation is equivalent to Adaptive Average Pooling,
            # which is appropriate for aggregating ground truth over a larger receptive field.
            target_scaled = F.interpolate(targets, size=target_size, mode="area")
            mask_scaled = F.interpolate(mask, size=target_size, mode="area")

            # Calculate loss against scaled targets
            loss_aux_elementwise = self.criterion(aux_pred, target_scaled)

            # Apply scaled mask (which now contains values between 0 and 1 representing validity fraction)
            masked_loss_aux = loss_aux_elementwise * mask_scaled

            # Normalize
            valid_elements_aux = mask_scaled.sum() * aux_pred.size(1) + 1e-8
            term_aux = masked_loss_aux.sum() / valid_elements_aux

            loss_aux_total += term_aux

        # --- 3. Combine Losses ---
        total_loss = term_final + (self.aux_weight * loss_aux_total)

        metrics = {
            "loss": total_loss.item(),
            "loss_final": term_final.item(),
            "loss_aux": (
                loss_aux_total.item()
                if isinstance(loss_aux_total, torch.Tensor)
                else loss_aux_total
            ),
        }

        return total_loss, metrics
