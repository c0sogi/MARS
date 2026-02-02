import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function computes the RMSE for each specific target column and then
    averages them. It is designed to handle the specific requirements of the
    RNA degradation task:
    1. Slicing: The model predicts for the full sequence length (e.g., 107), but
       ground truth is only available for the first `seq_scored` positions (e.g., 68).
    2. Column Filtering: While the dataset contains 5 target columns, only a subset
       (reactivity, deg_Mg_pH10, deg_Mg_50C) contributes to the competition score.
       This loss masks out the auxiliary targets during optimization.
    """

    def __init__(self):
        super().__init__()
        # Identify the indices of the columns that are actually scored
        # Config.ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Config.SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            Config.ALL_TARGETS.index(t) for t in Config.SCORED_TARGETS
        ]

        # Register indices as a buffer so they are moved to the correct device automatically
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, inputs, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Num_Targets).
                                   e.g., (Batch, 107, 5)
            targets (torch.Tensor): Ground truth values of shape (Batch, Pred_Len, Num_Targets).
                                    e.g., (Batch, 68, 5)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Align Sequence Lengths
        # The model outputs predictions for the entire sequence (107), but targets
        # only exist for the first `PRED_LEN` (68) positions.
        # We slice the inputs to match the targets.
        preds_sliced = inputs[:, : Config.PRED_LEN, :]

        # 2. Select Scored Columns
        # We filter both predictions and targets to keep only the columns used for scoring.
        # Using index_select or simple slicing with the stored indices.
        preds_scored = torch.index_select(preds_sliced, 2, self.scored_indices_tensor)
        targets_scored = torch.index_select(targets, 2, self.scored_indices_tensor)

        # 3. Compute MSE per Column
        # Calculate squared differences
        squared_diff = (preds_scored - targets_scored) ** 2

        # Average over Batch (dim 0) and Sequence Length (dim 1)
        # Result is a vector of MSE values, one per scored column.
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # 4. Compute RMSE per Column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Average RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss
