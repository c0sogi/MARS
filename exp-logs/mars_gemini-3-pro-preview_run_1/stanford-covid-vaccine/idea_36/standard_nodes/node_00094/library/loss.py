import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyAwareMSELoss(nn.Module):
    """
    Implements the Uncertainty-Aware Multi-Task Mean Squared Error Loss.

    This loss function calculates the MSE for both the primary degradation predictions
    and the experimental error predictions. It applies a mask to ensure that only
    the valid scored positions (first 68 bases) contribute to the loss.

    Formula:
        Loss = MSE(y, y_pred) + lambda * MSE(sigma, sigma_pred)
    """

    def __init__(self, lambda_uncertainty: float = 1.0):
        """
        Args:
            lambda_uncertainty (float): Weighting factor for the uncertainty/error loss term.
                                        Defaults to 1.0.
        """
        super().__init__()
        self.lambda_uncertainty = lambda_uncertainty

    def forward(self, pred_val, pred_err, target_val, target_err, mask):
        """
        Calculates the weighted multi-task MSE loss.

        Args:
            pred_val (torch.Tensor): Predicted degradation values. Shape [Batch, SeqLen, 3].
            pred_err (torch.Tensor): Predicted error/uncertainty values. Shape [Batch, SeqLen, 3].
            target_val (torch.Tensor): Ground truth degradation values. Shape [Batch, SeqLen, 3].
            target_err (torch.Tensor): Ground truth error values. Shape [Batch, SeqLen, 3].
            mask (torch.Tensor): Binary mask indicating scored positions. Shape [Batch, SeqLen].

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # Expand mask to match channel dimensions: [Batch, SeqLen] -> [Batch, SeqLen, 3]
        # We unsqueeze the last dimension and expand to matching size
        mask_expanded = mask.unsqueeze(-1).expand_as(pred_val)

        # Calculate the number of valid elements in the mask (sum over all dimensions)
        # Add a small epsilon to prevent division by zero in case of empty masks (unlikely)
        count = mask_expanded.sum() + 1e-8

        # --- 1. Value Loss (Primary Targets) ---
        # Calculate squared difference
        diff_val = pred_val - target_val
        # Apply mask: invalid positions become 0
        masked_diff_val = diff_val * mask_expanded
        # Sum of squares divided by count = MSE
        mse_val = torch.sum(masked_diff_val**2) / count

        # --- 2. Uncertainty Loss (Error Targets) ---
        # Calculate squared difference
        diff_err = pred_err - target_err
        # Apply mask
        masked_diff_err = diff_err * mask_expanded
        # Sum of squares divided by count = MSE
        mse_err = torch.sum(masked_diff_err**2) / count

        # --- Total Loss ---
        loss = mse_val + self.lambda_uncertainty * mse_err

        return loss
