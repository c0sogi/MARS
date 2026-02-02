import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss between predictions and targets,
    strictly applying a binary mask to ignore unscored sequence positions.

    This aligns with the training objective defined in the strategy:
    "Masked Mean Squared Error (MSE): Calculate loss only for the first 68 positions."
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the masked MSE loss.

        Args:
            preds: Predictions tensor of shape (Batch, SeqLen, NumTargets).
            targets: Ground truth tensor of shape (Batch, SeqLen, NumTargets).
            mask: Binary mask tensor of shape (Batch, SeqLen).
                  Values should be 1.0 for scored positions and 0.0 for unscored positions.

        Returns:
            A scalar tensor representing the mean squared error over all valid positions.
        """
        # 1. Expand mask dimensions for broadcasting
        # Input mask: (Batch, SeqLen)
        # Target shape: (Batch, SeqLen, NumTargets)
        # We need mask to be (Batch, SeqLen, 1) to broadcast across the target channels
        mask_expanded = mask.unsqueeze(-1)

        # 2. Compute element-wise squared errors
        # (Batch, SeqLen, NumTargets)
        squared_errors = (preds - targets) ** 2

        # 3. Apply the mask
        # Zero out errors at positions where mask is 0.0
        masked_errors = squared_errors * mask_expanded

        # 4. Compute the mean loss
        # We sum the errors and divide by the total number of valid elements.
        # Total valid elements = (Sum of mask weights) * (Number of target channels)

        # Sum of mask weights (Batch * SeqLen valid positions)
        valid_position_count = mask_expanded.sum()

        # Total number of scalar predictions considered (Batch * SeqLen_valid * NumTargets)
        num_valid_elements = valid_position_count * preds.shape[-1]

        # Safety check to avoid division by zero (though unlikely with proper batching)
        if num_valid_elements == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        loss = masked_errors.sum() / num_valid_elements

        return loss
