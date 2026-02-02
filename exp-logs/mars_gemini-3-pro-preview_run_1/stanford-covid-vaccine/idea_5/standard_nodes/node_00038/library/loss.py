import torch
import torch.nn as nn
from library.model import weighted_masked_mse_loss


class SignalWeightedMSELoss(nn.Module):
    """
    A PyTorch Loss module that implements the Signal-Weighted Masked Mean Squared Error.

    This class wraps the functional implementation provided in library.model.
    It computes the MSE between predictions and targets, but only for the specific
    scored positions (masked), and weights the error for each sample based on its
    signal-to-noise ratio (SN_filter/signal_to_noise).
    """

    def __init__(self):
        """
        Initializes the SignalWeightedMSELoss module.
        """
        super(SignalWeightedMSELoss, self).__init__()

    def forward(self, preds, targets, masks, weights):
        """
        Forward pass to compute the loss.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 5).
            masks (torch.Tensor): Mask tensor of shape (Batch, Seq_Len), where 1.0 indicates
                                  a scored position and 0.0 indicates an unscored position.
            weights (torch.Tensor): Weight tensor of shape (Batch,), representing the
                                    signal-to-noise ratio or importance of each sample.

        Returns:
            torch.Tensor: A scalar tensor representing the computed loss.
        """
        return weighted_masked_mse_loss(preds, targets, masks, weights)
