import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function specifically handles:
    1. Selecting only the scored columns defined in Config.SCORED_INDICES.
    2. Masking invalid sequence positions (e.g., padding or unscored tails).
    3. Slicing the sequence to the maximum scored length for efficiency.
    """

    def __init__(self):
        super().__init__()
        # Load configuration
        self.scored_indices = torch.tensor(Config.SCORED_INDICES, dtype=torch.long)
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, preds, targets, mask=None):
        """
        Computes the MCRMSE loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Channels, Length).
                                  Expected Channels=5.
            targets (torch.Tensor): Ground truth targets of shape (Batch, Length, Channels).
                                    Expected Channels=5.
            mask (torch.Tensor, optional): Validity mask of shape (Batch, Length).
                                           1.0 for valid positions, 0.0 otherwise.

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # Align targets to (Batch, Channels, Length) to match predictions
        # targets: (N, L, C) -> (N, C, L)
        if targets.shape[1] == Config.SEQ_LEN and targets.shape[2] == 5:
            targets = targets.permute(0, 2, 1)

        # Move indices to the correct device
        if self.scored_indices.device != preds.device:
            self.scored_indices = self.scored_indices.to(preds.device)

        # Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # preds: (N, 5, L) -> (N, 3, L)
        preds_scored = torch.index_select(preds, 1, self.scored_indices)
        targets_scored = torch.index_select(targets, 1, self.scored_indices)

        # Optimization: Slice tensors to the maximum scored length defined in Config.
        # This avoids computing loss on the unscored tail (indices 68-106).
        # We take the minimum of SEQ_SCORED and the actual length to be safe.
        eff_len = min(self.seq_scored, preds_scored.shape[2])
        preds_scored = preds_scored[:, :, :eff_len]
        targets_scored = targets_scored[:, :, :eff_len]

        # Compute squared differences
        squared_diff = (preds_scored - targets_scored) ** 2

        if mask is not None:
            # Slice mask to match the effective length and expand dimensions
            # mask: (N, L) -> (N, L_eff) -> (N, 1, L_eff)
            mask_sliced = mask[:, :eff_len].unsqueeze(1)

            # Apply mask to squared differences
            squared_diff = squared_diff * mask_sliced

            # Calculate the number of valid elements per column
            # Since the mask is shared across channels, the count is the sum of the mask
            # summed over Batch (dim 0) and Length (dim 2).
            # Result is a scalar representing total valid positions per channel.
            total_valid = torch.sum(mask_sliced)

            # Compute Sum of Squared Errors (SSE) per column
            # Sum over Batch (dim 0) and Length (dim 2) -> Shape (Num_Scored_Cols,)
            col_sse = torch.sum(squared_diff, dim=(0, 2))

            # Compute MSE per column
            # Add epsilon to avoid division by zero
            col_mse = col_sse / (total_valid + 1e-8)
        else:
            # If no mask is provided, assume all sliced positions are valid
            # Mean over Batch (0) and Length (2)
            col_mse = torch.mean(squared_diff, dim=(0, 2))

        # Compute RMSE per column
        col_rmse = torch.sqrt(col_mse)

        # Compute MCRMSE (Mean of column RMSEs)
        loss = torch.mean(col_rmse)

        return loss
