import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    Implements the 'Masked Optimization' strategy by specifically masking out
    auxiliary targets (deg_pH10, deg_50C) and calculating the loss only on
    the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()

        # Retrieve column definitions from Config
        target_cols = Config.TARGET_COLS
        scored_cols = Config.SCORED_COLS

        # Determine indices of the columns that contribute to the score
        # Expected indices for standard setup: [0, 1, 3]
        self.scored_indices = [
            i for i, col in enumerate(target_cols) if col in scored_cols
        ]

        # Register indices as a buffer so they are saved with the state_dict
        # and moved to the correct device automatically.
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, preds, targets):
        """
        Calculates the MCRMSE loss on the masked columns.

        Args:
            preds (torch.Tensor): Predictions. Shape (Batch, SeqLen, Channels) or (Batch, Channels).
            targets (torch.Tensor): Ground truth. Shape (Batch, SeqLen, Channels) or (Batch, Channels).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # 1. Sequence Length Alignment
        # If predictions cover the full sequence (107) but targets are only for the scored portion (68),
        # slice the predictions to match the targets.
        if preds.ndim == 3 and targets.ndim == 3:
            if preds.shape[1] > targets.shape[1]:
                preds = preds[:, : targets.shape[1], :]

        # 2. Column Masking
        # Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # using the pre-calculated indices.
        preds_masked = torch.index_select(preds, -1, self.scored_indices_tensor)
        targets_masked = torch.index_select(targets, -1, self.scored_indices_tensor)

        # 3. Compute MSE per column
        # We average over the Batch and Sequence dimensions, keeping the Channel dimension.
        if preds_masked.ndim == 3:
            # Shape: (Batch, SeqLen, Channels) -> Reduce dim 0 and 1
            mse_per_col = torch.mean((preds_masked - targets_masked) ** 2, dim=(0, 1))
        elif preds_masked.ndim == 2:
            # Shape: (Batch, Channels) -> Reduce dim 0
            mse_per_col = torch.mean((preds_masked - targets_masked) ** 2, dim=0)
        else:
            raise ValueError(
                f"Unsupported input dimension for loss calculation: {preds_masked.ndim}"
            )

        # 4. Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Average RMSEs to get final MCRMSE
        loss = torch.mean(rmse_per_col)

        return loss
