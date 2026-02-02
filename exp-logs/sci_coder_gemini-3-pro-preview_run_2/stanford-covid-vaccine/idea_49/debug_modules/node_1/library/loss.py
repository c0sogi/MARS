import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Custom loss function for RNA degradation prediction.
    Calculates Mean Columnwise Root Mean Squared Error (MCRMSE)
    on specific scored target columns and masked sequence positions.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.scored_indices = Config.TARGET_INDICES
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, preds, targets, mask=None):
        """
        Computes the MCRMSE loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Length, Num_Targets).
            targets (torch.Tensor): Ground truth targets of shape (Batch, Length, Num_Targets).
            mask (torch.Tensor, optional): Binary mask of shape (Batch, Length) indicating
                                           valid positions to score. If None, a mask is created
                                           automatically based on Config.SEQ_SCORED.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # If mask is not provided, generate it based on SEQ_SCORED (usually 68)
        if mask is None:
            device = preds.device
            B, L, _ = preds.shape
            mask = torch.zeros((B, L), device=device)
            mask[:, : self.seq_scored] = 1.0

        # Convert mask to boolean for indexing (1.0 = valid, 0.0 = invalid)
        mask_bool = mask > 0.5

        column_losses = []

        # Iterate over the specific columns that contribute to the score
        # Typically: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
        for idx in self.scored_indices:
            p = preds[:, :, idx]
            t = targets[:, :, idx]

            # Calculate squared errors
            diff_sq = (p - t) ** 2

            # Select only the valid positions defined by the mask
            # This flattens the selected elements into a 1D tensor
            valid_diff_sq = diff_sq[mask_bool]

            # Compute MSE and then RMSE
            if valid_diff_sq.numel() > 0:
                mse = torch.mean(valid_diff_sq)
                # Add epsilon for numerical stability in sqrt
                rmse = torch.sqrt(mse + 1e-8)
            else:
                rmse = torch.tensor(0.0, device=preds.device)

            column_losses.append(rmse)

        # MCRMSE is the mean of the RMSEs of the scored columns
        if not column_losses:
            return torch.tensor(0.0, device=preds.device)

        loss = torch.mean(torch.stack(column_losses))

        return loss
