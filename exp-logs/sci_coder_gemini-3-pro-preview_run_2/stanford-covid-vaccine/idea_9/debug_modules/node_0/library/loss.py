import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Implements the Masked Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function:
    1. Slices the input to the scored sequence length (first 68 bases).
    2. Selects only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computes RMSE for each selected column.
    4. Returns the average of these RMSEs.

    This strategy prevents negative transfer from the auxiliary targets (deg_pH10, deg_50C)
    which are not part of the competition scoring metric.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()

        # Get column definitions from Config
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS

        # Determine indices of the columns that contribute to the score
        # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Indices should be [0, 1, 3]
        indices = [self.target_cols.index(col) for col in self.scored_cols]

        # Register as a buffer so it is saved with the model state and moves to device
        self.register_buffer("scored_indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice to the scored sequence length
        # Inputs might be length 107, but we only evaluate on the first 68
        preds_sliced = preds[:, : Config.SCORED_SEQ_LENGTH, :]
        targets_sliced = targets[:, : Config.SCORED_SEQ_LENGTH, :]

        # 2. Select only the scored columns
        # Use index_select to extract the specific columns we care about
        preds_filtered = torch.index_select(preds_sliced, 2, self.scored_indices)
        targets_filtered = torch.index_select(targets_sliced, 2, self.scored_indices)

        # 3. Compute MSE per column
        # We average over Batch (dim 0) and Sequence (dim 1)
        # Result shape: (Num_Scored_Cols,) i.e., (3,)
        mse = torch.mean((preds_filtered - targets_filtered) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Compute Mean of RMSEs (MCRMSE)
        loss = torch.mean(rmse)

        return loss
