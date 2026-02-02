import torch
import torch.nn as nn


class AnchoredMCRMSELoss(nn.Module):
    """
    Anchored Mean Columnwise Root Mean Squared Error Loss.

    This loss calculates the MCRMSE over the full sequence length (0-107).
    It is designed to work with the 'Boundary Anchoring' strategy where the
    unscored tail positions (68-107) in the targets are filled with a neutral
    baseline (0.0). By calculating loss over these positions, the model is
    penalized for drifting away from zero in the tail, stabilizing the
    Bidirectional RNN's backward pass.
    """

    def __init__(self):
        super(AnchoredMCRMSELoss, self).__init__()

    def forward(self, pred, target):
        """
        Calculates the loss.

        Args:
            pred (torch.Tensor): Predicted values of shape (N, L, 5).
            target (torch.Tensor): Ground truth values of shape (N, L, 5).
                                   Tail positions should be 0.0.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate Squared Error: (N, L, 5)
        squared_error = (pred - target) ** 2

        # Calculate MSE per column by averaging over Batch (dim 0) and Sequence (dim 1)
        # Result shape: (5,)
        mse_per_col = torch.mean(squared_error, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Calculate Mean of RMSEs across the 5 columns
        loss = torch.mean(rmse_per_col)

        return loss
