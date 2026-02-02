import torch
import torch.nn as nn
from library.config import Config


class DeepSupervisionMAELoss(nn.Module):
    """
    Calculates Mean Absolute Error (MAE) with support for Deep Supervision.
    Handles masking to ignore padded time steps in the sequence.
    """

    def __init__(self):
        super().__init__()
        self.weights = Config.DEEP_SUPERVISION_WEIGHTS

    def _masked_mae(self, pred, target, mask):
        """
        Computes MAE only for valid time steps defined by the mask.

        Args:
            pred (torch.Tensor): Predictions of shape (Batch, Channels, Time).
            target (torch.Tensor): Ground truth of shape (Batch, Channels, Time).
            mask (torch.Tensor): Valid time step mask of shape (Batch, Time).
                                 1.0 for valid, 0.0 for padding.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Expand mask to match channel dimension: (B, T) -> (B, 1, T)
        # This allows broadcasting against (B, C, T)
        mask_expanded = mask.unsqueeze(1)

        # Calculate absolute error
        abs_err = torch.abs(pred - target)

        # Zero out errors corresponding to padded positions
        masked_err = abs_err * mask_expanded

        # Calculate the number of valid elements (considering channels)
        # mask_expanded has 1s and 0s. Expanding it to (B, C, T) counts total valid entries.
        valid_elements = mask_expanded.expand_as(pred).sum()

        # Compute mean over valid elements
        # Add epsilon to avoid division by zero if a batch is entirely padding (unlikely but safe)
        loss = masked_err.sum() / (valid_elements + 1e-8)

        return loss

    def forward(self, outputs, targets, mask):
        """
        Args:
            outputs: Model output. Can be a list [final, aux1, aux2...] or a single tensor.
            targets: Ground truth tensor (Batch, Channels, Time).
            mask: Mask tensor (Batch, Time).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        # Case 1: Deep Supervision (Training)
        # Model returns a list of tensors: [final_output, aux_output_1, aux_output_2, ...]
        if isinstance(outputs, list):
            final_pred = outputs[0]
            aux_preds = outputs[1:]

            # Calculate loss for the final output
            loss_final = self._masked_mae(final_pred, targets, mask)

            # Calculate loss for auxiliary outputs
            loss_aux_total = 0.0
            for aux_pred in aux_preds:
                loss_aux_total += self._masked_mae(aux_pred, targets, mask)

            # Combine losses using weights from Config
            # Total Loss = w_final * L_final + w_aux * (L_aux1 + L_aux2 + ...)
            total_loss = (self.weights["final"] * loss_final) + (
                self.weights["aux"] * loss_aux_total
            )

            return total_loss

        # Case 2: Single Output (Validation / Inference)
        # Model returns a single tensor
        else:
            return self._masked_mae(outputs, targets, mask)
