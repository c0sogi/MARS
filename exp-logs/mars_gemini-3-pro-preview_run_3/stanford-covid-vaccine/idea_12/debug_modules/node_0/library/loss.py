import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function calculates the Root Mean Squared Error (RMSE) for each
    target column independently and then returns the average of these RMSEs.
    It automatically handles the slicing of predictions to match the target
    sequence length (e.g., slicing 107 predictions to the first 68 scored positions).
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values from the model.
                                   Shape: (Batch, SeqLen_Input, NumColumns).
                                   Typically (Batch, 107, 5).
            targets (torch.Tensor): Ground truth target values.
                                    Shape: (Batch, SeqLen_Target, NumColumns).
                                    Typically (Batch, 68, 5).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # The model predicts for the full sequence (107), but targets exist
        # only for the first 'seq_scored' positions (68).
        # We slice the inputs to align with the targets.
        seq_len_target = targets.shape[1]
        if inputs.shape[1] > seq_len_target:
            inputs = inputs[:, :seq_len_target, :]

        # Compute squared differences: (Batch, SeqLen, NumColumns)
        squared_diff = (inputs - targets) ** 2

        # Compute Mean Squared Error (MSE) per column.
        # Average over batch (dim 0) and sequence (dim 1) dimensions.
        # Shape: (NumColumns,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Compute Root Mean Squared Error (RMSE) per column.
        rmse_per_column = torch.sqrt(mse_per_column)

        # Compute the mean of the column-wise RMSEs.
        # This treats all 5 columns equally (unweighted).
        loss = torch.mean(rmse_per_column)

        return loss
