import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function specifically handles the requirements of the RNA degradation task:
    1. It masks the input to consider only the first `Config.PRED_LEN` (68) sequence positions.
    2. It filters the targets to consider only the columns specified in `Config.SCORED_COLS`.
       (reactivity, deg_Mg_pH10, deg_Mg_50C), ignoring auxiliary targets.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()

        # Determine the indices of the columns that should be scored
        # Config.TARGET_COLS contains all available targets in the dataset
        # Config.SCORED_COLS contains the subset used for the metric
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

        # Register indices as a buffer so they are moved to the correct device (GPU/CPU)
        # alongside the model parameters.
        self.register_buffer(
            "scored_idxs", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, preds, targets):
        """
        Compute the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice to the scoring length (first 68 positions)
        # We ensure we don't exceed the actual tensor dimensions if they are shorter for some reason
        score_len = min(Config.PRED_LEN, preds.shape[1])

        p_sliced = preds[:, :score_len, :]
        t_sliced = targets[:, :score_len, :]

        # 2. Select only the scored columns
        # Using index_select is robust for non-contiguous indices
        p_scored = torch.index_select(p_sliced, 2, self.scored_idxs)
        t_scored = torch.index_select(t_sliced, 2, self.scored_idxs)

        # 3. Compute MSE per column
        # We average over the batch (dim 0) and sequence (dim 1) dimensions
        # Result shape: (Num_Scored_Cols,)
        mse_per_col = torch.mean((p_scored - t_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Average RMSEs across columns to get MCRMSE
        loss = torch.mean(rmse_per_col)

        return loss
