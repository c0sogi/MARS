import torch
import torch.nn as nn
from library.config import Config


class DeepSupervisionMAELoss(nn.Module):
    """
    Custom loss function for Deep Supervision with Masked Mean Absolute Error.
    Calculates MAE between predicted and ground truth offsets, respecting sequence padding.
    Combines main output loss with auxiliary output losses using defined weights.
    """

    def __init__(self):
        super().__init__()
        self.weights = Config.LOSS_WEIGHTS
        # Use 'none' reduction to apply mask manually
        self.mae_loss = nn.L1Loss(reduction="none")

    def calculate_masked_mae(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes MAE ignoring padded values.

        Args:
            pred: Prediction tensor of shape (Batch, Seq_Len, Channels)
            target: Target tensor of shape (Batch, Seq_Len, Channels)
            mask: Boolean mask tensor of shape (Batch, Seq_Len) where True indicates valid data

        Returns:
            Scalar loss value
        """
        # Calculate element-wise absolute error
        # pred and target are (B, L, C)
        abs_err = self.mae_loss(pred, target)

        # Expand mask to match channel dimension: (B, L) -> (B, L, 1) -> (B, L, C)
        mask_expanded = mask.unsqueeze(-1).expand_as(abs_err)

        # Apply mask (cast boolean mask to float)
        masked_err = abs_err * mask_expanded.float()

        # Calculate mean over valid elements
        # Sum of mask gives number of valid time steps. Multiply by Channels for total valid elements.
        num_valid_elements = mask_expanded.sum()

        # Avoid division by zero
        if num_valid_elements == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return masked_err.sum() / num_valid_elements

    def forward(self, preds, targets, mask):
        """
        Args:
            preds: Model output. Can be a single Tensor (B, C, L) or a tuple of Tensors.
            targets: Ground truth targets (B, L, C).
            mask: Padding mask (B, L).

        Returns:
            Weighted loss scalar.
        """
        # Case 1: Deep Supervision (Training) - Tuple of outputs
        if isinstance(preds, (tuple, list)):
            total_loss = 0.0

            # Iterate over outputs (Main, Aux1, Aux2, ...)
            for i, pred in enumerate(preds):
                if i < len(self.weights):
                    weight = self.weights[i]

                    # Model output is (B, C, L), permute to (B, L, C) to match targets
                    pred_permuted = pred.permute(0, 2, 1)

                    # Ensure dimensions match (handles potential slight interpolation mismatches if any)
                    if pred_permuted.size(1) != targets.size(1):
                        # This should theoretically be handled by the model's interpolation,
                        # but as a safety measure for loss calculation:
                        pred_permuted = torch.nn.functional.interpolate(
                            pred,
                            size=targets.size(1),
                            mode="linear",
                            align_corners=False,
                        ).permute(0, 2, 1)

                    loss = self.calculate_masked_mae(pred_permuted, targets, mask)
                    total_loss += weight * loss

            return total_loss

        # Case 2: Single Output (Validation/Inference)
        else:
            # Model output is (B, C, L), permute to (B, L, C)
            pred_permuted = preds.permute(0, 2, 1)
            return self.calculate_masked_mae(pred_permuted, targets, mask)
