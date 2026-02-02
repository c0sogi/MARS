import torch
import torch.nn as nn
from library.config import Config


class AnchoredMCRMSELoss(nn.Module):
    """
    Implements the Anchored MCRMSE Loss for the ADF-RN strategy.

    This loss calculates the Mean Columnwise Root Mean Squared Error over the
    entire sequence length (0 to 107). By including the unmeasured tail positions
    (where ground truth is set to 0.0), it anchors the bidirectional RNN's hidden
    states, preventing drift and noise in the valid scored region.

    The final loss is a weighted sum of the losses from the two iterative passes.
    """

    def __init__(self):
        super(AnchoredMCRMSELoss, self).__init__()
        self.pass1_weight = Config.PASS1_LOSS_WEIGHT
        self.pass2_weight = Config.PASS2_LOSS_WEIGHT

    def forward(self, preds_pass1, preds_pass2, targets):
        """
        Computes the combined anchored loss.

        Args:
            preds_pass1 (torch.Tensor): Predictions from the first pass (no feedback).
                                        Shape: (Batch, Seq_Len, Num_Targets)
            preds_pass2 (torch.Tensor): Predictions from the second pass (with feedback).
                                        Shape: (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth targets.
                                    Shape: (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: The weighted scalar loss.
        """
        # Compute MCRMSE for Pass 1
        loss_pass1 = self.mcrmse(preds_pass1, targets)

        # Compute MCRMSE for Pass 2
        loss_pass2 = self.mcrmse(preds_pass2, targets)

        # Weighted sum
        total_loss = (self.pass1_weight * loss_pass1) + (self.pass2_weight * loss_pass2)

        return total_loss

    def mcrmse(self, preds, targets):
        """
        Calculates the Mean Columnwise Root Mean Squared Error over the full sequence.

        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Scalar MCRMSE value.
        """
        # Calculate Squared Error: (y - y_hat)^2
        squared_errors = (preds - targets) ** 2

        # Average over the batch and sequence dimensions (N elements)
        # Shape: (Batch, Seq_Len, Num_Targets) -> (Num_Targets,)
        mse_per_column = torch.mean(squared_errors, dim=(0, 1))

        # RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Mean of RMSEs across columns (MCRMSE)
        mcrmse_val = torch.mean(rmse_per_column)

        return mcrmse_val
