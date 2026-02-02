import torch
import torch.nn as nn
from library.config import SCORED_COLS_INDICES, PRED_LEN


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This loss function focuses only on the scored columns specified in the competition
    (reactivity, deg_Mg_pH10, deg_Mg_50C) and the valid sequence positions (first 68 bases).

    It supports a combined loss calculation for architectures with auxiliary outputs (e.g., recycling):
    Loss = MCRMSE(Y_final) + 0.5 * MCRMSE(Y_aux)
    """

    def __init__(self):
        super().__init__()
        self.scored_indices = SCORED_COLS_INDICES
        self.pred_len = PRED_LEN
        self.aux_weight = 0.5

    def compute_single(self, pred, target, mask=None):
        """
        Computes MCRMSE for a single prediction tensor.

        Args:
            pred: Prediction tensor [B, 5, L]
            target: Ground truth tensor [B, 5, L]
            mask: Optional mask tensor [B, L]

        Returns:
            torch.Tensor: Scalar loss value.
        """
        device = pred.device

        # Select only the scored columns (indices 0, 1, 3)
        scored_indices_tensor = torch.tensor(self.scored_indices, device=device)
        pred_scored = torch.index_select(pred, 1, scored_indices_tensor)
        target_scored = torch.index_select(target, 1, scored_indices_tensor)

        # Compute Squared Error
        mse = (pred_scored - target_scored) ** 2

        # Construct the validity mask
        # We only score the first PRED_LEN (68) positions.
        B, C, L = pred_scored.shape

        # Create a positional mask (1 for indices 0-67, 0 otherwise)
        pos_mask = torch.zeros((B, L), device=device)
        pos_mask[:, : self.pred_len] = 1.0

        # Combine with input mask (e.g., padding mask) if provided
        if mask is not None:
            final_mask = mask * pos_mask
        else:
            final_mask = pos_mask

        # Expand mask to match channel dimensions [B, C, L]
        final_mask_exp = final_mask.unsqueeze(1).expand_as(mse)

        # Apply mask to MSE
        mse = mse * final_mask_exp

        # Sum errors and counts per column (sum over Batch and Length)
        sum_sq_err = mse.sum(dim=(0, 2))
        counts = final_mask_exp.sum(dim=(0, 2))

        # Avoid division by zero
        counts = torch.clamp(counts, min=1e-6)

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(sum_sq_err / counts)

        # Return the mean of the column RMSEs
        return rmse_per_col.mean()

    def forward(self, preds, target, mask=None):
        """
        Computes the loss.

        Args:
            preds: either a single Tensor [B, 5, L] or a tuple/list of Tensors
                   ([B, 5, L], [B, 5, L]) representing (final_pred, aux_pred).
            target: Tensor [B, 5, L]
            mask: Tensor [B, L] (optional)

        Returns:
            torch.Tensor: The calculated loss.
        """
        if isinstance(preds, (list, tuple)):
            # Combined Loss: Final + 0.5 * Aux
            # Assuming preds[0] is the Final output and preds[1] is the Auxiliary (Pass 1) output
            y_final = preds[0]
            y_aux = preds[1]

            loss_final = self.compute_single(y_final, target, mask)
            loss_aux = self.compute_single(y_aux, target, mask)

            return loss_final + self.aux_weight * loss_aux
        else:
            # Single output case
            return self.compute_single(preds, target, mask)


class GlobalMCRMSE:
    """
    Accumulates errors and counts over the entire validation set to compute
    the correct global MCRMSE.

    This avoids the bias introduced by averaging RMSEs calculated on small batches.
    """

    def __init__(self):
        self.scored_indices = SCORED_COLS_INDICES
        self.pred_len = PRED_LEN
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # Stores sum of squared errors for each scored column
        self.sum_sq_errors = None
        # Stores total valid counts for each scored column
        self.total_counts = None

    def update(self, pred, target, mask=None):
        """
        Updates the accumulators with a new batch of predictions.

        Args:
            pred: Prediction tensor [B, 5, L]
            target: Ground truth tensor [B, 5, L]
            mask: Optional mask tensor [B, L]
        """
        device = pred.device
        scored_indices_tensor = torch.tensor(self.scored_indices, device=device)

        # Select scored columns
        pred_scored = torch.index_select(pred, 1, scored_indices_tensor)
        target_scored = torch.index_select(target, 1, scored_indices_tensor)

        # Calculate squared errors
        sq_errors = (pred_scored - target_scored) ** 2

        # Masking logic
        B, C, L = pred_scored.shape
        pos_mask = torch.zeros((B, L), device=device)
        pos_mask[:, : self.pred_len] = 1.0

        if mask is not None:
            final_mask = mask * pos_mask
        else:
            final_mask = pos_mask

        final_mask_exp = final_mask.unsqueeze(1).expand_as(sq_errors)

        # Apply mask
        sq_errors = sq_errors * final_mask_exp

        # Sum over batch and length dimensions to get totals for this batch
        batch_sum_sq = sq_errors.sum(dim=(0, 2))  # Shape: [num_scored_cols]
        batch_counts = final_mask_exp.sum(dim=(0, 2))  # Shape: [num_scored_cols]

        # Initialize state if this is the first update
        if self.sum_sq_errors is None:
            self.sum_sq_errors = torch.zeros_like(batch_sum_sq).detach()
            self.total_counts = torch.zeros_like(batch_counts).detach()

        self.sum_sq_errors += batch_sum_sq.detach()
        self.total_counts += batch_counts.detach()

    def compute(self):
        """
        Computes the final global MCRMSE based on accumulated statistics.

        Returns:
            float: The global MCRMSE value.
        """
        if self.sum_sq_errors is None or self.total_counts is None:
            return 0.0

        # Avoid division by zero
        counts = torch.clamp(self.total_counts, min=1e-6)

        # RMSE per column = sqrt(Total SSE / Total Count)
        rmse_per_col = torch.sqrt(self.sum_sq_errors / counts)

        # Metric is the mean of the column RMSEs
        global_mcrmse = rmse_per_col.mean().item()

        return global_mcrmse
