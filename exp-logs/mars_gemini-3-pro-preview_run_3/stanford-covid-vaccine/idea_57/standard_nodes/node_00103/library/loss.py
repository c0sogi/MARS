import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function computes the average Root Mean Squared Error (RMSE) across
    multiple target columns. It is designed for the RNA degradation prediction task
    where the objective is to minimize the error across 5 distinct reactivity/degradation
    conditions simultaneously (Multi-Task Learning).

    Formula:
        MCRMSE = (1 / N_t) * Sum_{j=1}^{N_t} sqrt( (1 / n) * Sum_{i=1}^{n} (y_{ij} - y_hat_{ij})^2 )
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values. Expected shape is (Batch, SeqLen, NumTargets).
                                   Typically (B, 68, 5) after slicing to scored positions.
            targets (torch.Tensor): Ground truth values. Expected shape is (Batch, SeqLen, NumTargets).
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Verify shapes match
        if inputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch in MCRMSELoss: inputs {inputs.shape} vs targets {targets.shape}. "
                "Ensure predictions are sliced to match the target sequence length (e.g., 68)."
            )

        # 1. Compute Squared Error: (y - y_hat)^2
        squared_diff = (inputs - targets) ** 2

        # 2. Compute Mean Squared Error (MSE) per column
        # We average over the batch (dim 0) and sequence (dim 1) dimensions,
        # preserving the target column dimension (dim 2).
        # Dynamic reduction based on input dimensions to handle (B, L, C) or flattened (N, C)
        dims_to_reduce = tuple(range(inputs.dim() - 1))
        mse_per_column = torch.mean(squared_diff, dim=dims_to_reduce)

        # 3. Compute Root Mean Squared Error (RMSE) per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 4. Compute Mean of RMSEs across all columns (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss
