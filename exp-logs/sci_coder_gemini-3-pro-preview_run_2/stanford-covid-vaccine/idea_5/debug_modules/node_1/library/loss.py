import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This loss function is specific to the RNA degradation task. It:
    1. Masks the input to only consider the first `seq_scored` positions (typically 68).
    2. Selects only the specific target columns used for scoring (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computes the RMSE for each column independently.
    4. Returns the average of these RMSE values.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

        # Determine the indices of the columns that contribute to the loss
        # Config.ALL_TARGET_COLS contains the order of outputs from the model
        # Config.SCORED_TARGET_COLS contains the subset we want to optimize on
        self.scored_indices = [
            Config.ALL_TARGET_COLS.index(col) for col in Config.SCORED_TARGET_COLS
        ]

        # Register as buffer so it moves to device with the module but isn't a parameter
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice to the scored sequence length
        # The competition only scores the first 68 bases.
        # Inputs/Targets shape becomes: (Batch, 68, Num_Targets)
        inputs_scored = inputs[:, : self.seq_scored, :]
        targets_scored = targets[:, : self.seq_scored, :]

        # 2. Select only the scored columns
        # We index the last dimension (channels/targets)
        # Inputs/Targets shape becomes: (Batch, 68, 3)
        inputs_selected = torch.index_select(
            inputs_scored, dim=2, index=self.scored_indices_tensor
        )
        targets_selected = torch.index_select(
            targets_scored, dim=2, index=self.scored_indices_tensor
        )

        # 3. Compute MSE per column
        # We calculate the squared difference
        squared_diff = (inputs_selected - targets_selected) ** 2

        # We average over Batch (dim 0) and Sequence (dim 1) to get MSE per column
        # Result shape: (3,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Compute Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
