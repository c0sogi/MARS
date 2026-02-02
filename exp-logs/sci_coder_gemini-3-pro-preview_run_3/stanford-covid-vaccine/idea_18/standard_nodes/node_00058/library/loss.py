import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function calculates the RMSE for each target column independently
    and then averages them. It is the specific metric used for the RNA Degradation
    competition.

    Formula:
    MCRMSE = (1/Nt) * sum_{j=1}^{Nt} sqrt( (1/n) * sum_{i=1}^{n} (y_{ij} - y_hat_{ij})^2 )

    Where:
    - Nt is the number of target columns (5).
    - n is the total number of scored positions (Batch_Size * Seq_Scored).
    - j iterates over columns.
    - i iterates over flattened samples/positions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions from the model.
                Shape: (Batch_Size, Seq_Len_Model, Num_Targets)
                e.g., (B, 107, 5)
            targets (torch.Tensor): Ground truth values.
                Shape: (Batch_Size, Seq_Len_Target, Num_Targets)
                e.g., (B, 68, 5)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # The model may output predictions for the full sequence length (e.g., 107),
        # but targets are only provided for the scored length (e.g., 68).
        # We slice the inputs to match the targets.
        if inputs.shape[1] > targets.shape[1]:
            inputs = inputs[:, : targets.shape[1], :]

        # Ensure shapes match after slicing
        if inputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}"
            )

        # Calculate Squared Error: (y - y_hat)^2
        squared_diff = (inputs - targets) ** 2

        # Calculate MSE per column.
        # We average over dimension 0 (Batch) and dimension 1 (Sequence Length).
        # This aggregates all predictions for a specific target type.
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate RMSE per column: sqrt(MSE)
        # Adding a small epsilon to prevent NaN gradients if MSE is exactly 0
        rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

        # Calculate the Mean of the column-wise RMSEs
        loss = torch.mean(rmse_per_column)

        return loss
