import torch
import torch.nn as nn
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss restricted to the first `seq_scored` positions.

    This loss function ensures that the model is optimized only on the valid experimental
    data (first 68 bases) and ignores the unscored positions (bases 69-107) which
    lack ground truth labels. This aligns the optimization objective with the
    evaluation metric and prevents the model from learning noise from padded/unscored regions.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.pred_len = Config.PRED_LEN

    def forward(self, preds, targets):
        """
        Computes the masked MSE loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Channels).
                                  Typically (Batch, 107, 3).
            targets (torch.Tensor): Ground truth values. Can be shape (Batch, Seq_Len, Channels)
                                    or (Batch, Scored_Len, Channels).

        Returns:
            torch.Tensor: Scalar MSE loss computed over the first `pred_len` positions.
        """
        # Slice predictions to the scored length (e.g., first 68 positions)
        # preds shape: [Batch, 107, 3] -> [Batch, 68, 3]
        preds_sliced = preds[:, : self.pred_len, :]

        # Slice targets to the scored length.
        # This handles cases where targets might be padded to the full sequence length (107)
        # or provided as just the scored length (68).
        targets_sliced = targets[:, : self.pred_len, :]

        # Compute standard MSE on the valid slice
        return self.mse(preds_sliced, targets_sliced)
