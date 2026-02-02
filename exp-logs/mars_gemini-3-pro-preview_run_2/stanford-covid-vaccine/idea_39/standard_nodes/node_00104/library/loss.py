import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function is designed for the RNA degradation prediction task.
    It computes the RMSE specifically for the scored columns:
    'reactivity', 'deg_Mg_pH10', and 'deg_Mg_50C'.

    The metric is defined as the average of the RMSE values of these columns.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.config = Config()

        # Identify the indices of the columns that are used for the evaluation metric.
        # target_cols: ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        # scored_cols: ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        # Resulting indices: [0, 1, 3]
        self.scored_indices = [
            self.config.target_cols.index(col) for col in self.config.scored_cols
        ]

        # Register indices as a buffer to ensure they move to the correct device
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Channels).
                                  Seq_Len is typically 107.
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Scored, Channels).
                                    Seq_Scored is typically 68.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Ensure predictions are sliced to match the length of the ground truth targets.
        # The model predicts the full sequence (107), but we only have ground truth
        # for the first 'seq_scored' positions (68).
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # Select only the specific columns required for the metric
        # Using index_select or simple slicing with the stored indices
        preds_scored = preds[:, :, self.scored_indices_tensor]
        targets_scored = targets[:, :, self.scored_indices_tensor]

        # Calculate Mean Squared Error (MSE) for each column.
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (num_scored_cols,)
        mse_per_col = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column.
        rmse_per_col = torch.sqrt(mse_per_col)

        # The final loss is the mean of the column-wise RMSEs.
        loss = torch.mean(rmse_per_col)

        return loss
