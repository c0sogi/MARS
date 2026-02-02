import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss is used for the Multi-Task Learning objective, optimizing
    across all 5 target columns:
    ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    It addresses the requirement to slice predictions to the scored sequence length
    before comparison with targets, and provides a differentiable approximation
    of the competition metric suitable for backpropagation.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.PRED_LEN

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MCRMSE Loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Num_Targets).
                                  Typically (B, 107, 5).
            targets (torch.Tensor): Ground truth targets of shape (Batch, Seq_Scored, Num_Targets).
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: A scalar tensor representing the mean of the RMSEs across all columns.
        """
        # 1. Slice predictions to match the scored sequence length (Config.PRED_LEN = 68)
        # The model outputs predictions for the full sequence (107), but targets exist only for the first 68.
        preds_sliced = preds[:, : self.seq_scored, :]

        # Verify shape compatibility
        if preds_sliced.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch in MCRMSELoss: "
                f"Preds sliced {preds_sliced.shape} vs Targets {targets.shape}"
            )

        # 2. Compute Squared Errors
        # Shape: (Batch, Seq_Scored, Num_Targets)
        squared_diff = (preds_sliced - targets) ** 2

        # 3. Compute Mean Squared Error (MSE) per column
        # Average over Batch (dim 0) and Sequence (dim 1) dimensions
        # Shape: (Num_Targets,) i.e., (5,)
        mse_per_col = torch.mean(squared_diff, dim=(0, 1))

        # 4. Compute Root Mean Squared Error (RMSE) per column
        # Add a small epsilon for numerical stability during backprop (derivative of sqrt(0) is undefined)
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # 5. Compute the Mean of RMSEs across all columns
        # Scalar output
        loss = torch.mean(rmse_per_col)

        return loss
