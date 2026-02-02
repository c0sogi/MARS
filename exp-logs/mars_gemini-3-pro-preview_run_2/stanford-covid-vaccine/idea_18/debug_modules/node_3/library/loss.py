import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss,
    specifically masking out auxiliary targets that are not scored in the competition.

    The loss is calculated as:
    MCRMSE = (1/M) * Sum_{k=1}^M (RMSE_k)
    where M is the number of scored columns.

    It automatically handles slicing of predictions if the model outputs the full
    sequence length (107) while targets are provided only for the scored length (68).
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.scored_indices = Config.SCORED_TARGET_INDICES
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions from the model.
                                  Shape: (Batch, Seq_Len_Pred, 5)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Len_Target, 5)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Align Sequence Lengths
        # If predictions cover the full sequence (e.g., 107) but targets are only for
        # the scored portion (e.g., 68), slice the predictions.
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # 2. Select Scored Columns
        # We only compute loss on reactivity, deg_Mg_pH10, and deg_Mg_50C
        # as defined in Config.SCORED_TARGET_INDICES.
        preds_scored = preds[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # 3. Compute Columnwise RMSE
        # We compute MSE for each column individually, take the sqrt, then average.
        # Note: nn.MSELoss with reduction='mean' computes the mean over all elements.
        # To get columnwise RMSE, we need to compute MSE per column first.

        # Calculate squared errors: (Batch, Seq, Columns)
        squared_errors = (preds_scored - targets_scored) ** 2

        # Mean over batch and sequence dimensions (dim 0 and 1), keeping columns (dim 2)
        mse_per_column = torch.mean(squared_errors, dim=(0, 1))

        # RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 4. Average RMSEs to get MCRMSE
        loss = torch.mean(rmse_per_column)

        return loss
