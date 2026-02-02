import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Implements the objective function for the RNA Degradation Prediction.
    Calculates MSE only on the scored positions (first 68 bases).
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.pred_len = Config.PRED_LEN

    def forward(self, pred, y_true):
        """
        Args:
            pred (torch.Tensor): Predictions (Batch, Seq_Len, Channels).
            y_true (torch.Tensor): Targets (Batch, Pred_Len, Channels).
        """
        # Ensure inputs are float
        y_true = y_true.float()

        # Slice Prediction to match scoring length
        pred_scored = pred[:, : self.pred_len, :]

        # Compute MSE
        loss = F.mse_loss(pred_scored, y_true)

        return loss
