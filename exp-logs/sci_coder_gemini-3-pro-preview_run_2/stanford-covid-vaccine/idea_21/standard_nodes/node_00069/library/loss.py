import torch
import torch.nn as nn


class MaskedMCRMSELoss(nn.Module):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) for specific
    target columns, ignoring others.

    Target columns structure:
    0: reactivity (Scored)
    1: deg_Mg_pH10 (Scored)
    2: deg_pH10 (Ignored)
    3: deg_Mg_50C (Scored)
    4: deg_50C (Ignored)

    Logical Mask: [1, 1, 0, 1, 0]
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        # Indices corresponding to the mask [1, 1, 0, 1, 0]
        self.scored_indices = [0, 1, 3]

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Length, 5)
            targets (torch.Tensor): Ground truth of shape (Batch, Length, 5)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Select only the scored columns
        # Using list indexing works on tensors regardless of device
        preds_scored = preds[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # 2. Compute MSE per column
        # Average over Batch (dim 0) and Sequence Length (dim 1)
        # Result shape: (3,) -> one MSE value per scored column
        mse_per_column = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # 3. Compute RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 4. Compute Mean of RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss
