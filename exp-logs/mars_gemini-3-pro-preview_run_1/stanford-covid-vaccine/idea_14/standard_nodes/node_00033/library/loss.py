import torch
import torch.nn as nn
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Implements a Masked Mean Squared Error (MSE) loss function.

    This loss is designed specifically for the RNA degradation task where:
    1. Only the first `Config.PRED_LEN` (68) positions of the sequence are scored.
    2. The model outputs predictions for the full sequence length (107).
    3. The metric is strictly MSE (L2), aligning with the MCRMSE evaluation.

    The forward pass slices the input predictions and targets to the valid
    scored region before computing the loss.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pred_len = Config.PRED_LEN

    def forward(self, preds, targets):
        """
        Computes the MSE loss on the valid scored positions.

        Args:
            preds (torch.Tensor): Predictions from the model.
                                  Shape: (Batch, Seq_Len, Channels)
                                  Typically (B, 107, 3).
            targets (torch.Tensor): Ground truth targets.
                                    Shape: (Batch, Seq_Len, Channels) or (Batch, Pred_Len, Channels).
                                    Typically (B, 107, 3) or (B, 68, 3).

        Returns:
            torch.Tensor: The scalar MSE loss calculated over the first `pred_len` positions.
        """
        # Slice predictions to the valid scored length (e.g., first 68 positions)
        # preds: [B, 107, 3] -> [B, 68, 3]
        preds_sliced = preds[:, : self.pred_len, :]

        # Slice targets to the valid scored length
        # This handles cases where targets are padded to full seq_length
        if targets.shape[1] > self.pred_len:
            targets_sliced = targets[:, : self.pred_len, :]
        else:
            targets_sliced = targets

        # Verify shapes align (sanity check)
        if preds_sliced.shape != targets_sliced.shape:
            raise ValueError(
                f"Shape mismatch in MaskedMSELoss: "
                f"preds sliced {preds_sliced.shape} vs targets sliced {targets_sliced.shape}"
            )

        # Compute standard MSE on the valid region
        return self.mse(preds_sliced, targets_sliced)
