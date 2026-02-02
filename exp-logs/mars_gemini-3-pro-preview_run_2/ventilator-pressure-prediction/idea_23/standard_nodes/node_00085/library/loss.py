import torch
import torch.nn as nn
from library.config import Config


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss for Ventilator Pressure Prediction.

    This loss function calculates the Mean Absolute Error (MAE) but applies different
    weights to the inspiratory and expiratory phases of the breath.

    - Inspiratory Phase (u_out = 0): Assigned a weight of 1.0 (Config.W_INSPIRATORY).
      This is the primary metric for the competition.
    - Expiratory Phase (u_out = 1): Assigned a reduced weight of 0.1 (Config.W_EXPIRATORY).
      This maintains gradient flow and temporal continuity for the Recurrent Neural Network
      hidden states without dominating the loss landscape with irrelevant data.
    """

    def __init__(self):
        super().__init__()
        self.w_insp = Config.W_INSPIRATORY
        self.w_exp = Config.W_EXPIRATORY

        # Determine the index of 'u_out' in the input feature tensor.
        # The SegregatedScaler constructs the input by concatenating:
        # [Continuous Features, Binary Features].
        # 'u_out' is located in the binary features list.
        try:
            self.u_out_idx = len(
                Config.CONTINUOUS_FEATURES
            ) + Config.BINARY_FEATURES.index("u_out")
        except ValueError:
            raise ValueError("'u_out' not found in Config.BINARY_FEATURES")

    def forward(self, preds, targets, inputs):
        """
        Calculates the weighted mean absolute error.

        Args:
            preds (torch.Tensor): Model predictions of shape (batch_size, seq_len).
            targets (torch.Tensor): Ground truth pressures of shape (batch_size, seq_len).
            inputs (torch.Tensor): Input features of shape (batch_size, seq_len, input_dim).
                                   Required to extract the 'u_out' control signal for weighting.

        Returns:
            torch.Tensor: The scalar weighted mean absolute error.
        """
        # Extract the u_out feature (0 for inspiration, 1 for expiration)
        # inputs shape: (batch, seq, features)
        u_out = inputs[:, :, self.u_out_idx]

        # Calculate element-wise L1 loss (Absolute Error)
        absolute_error = torch.abs(preds - targets)

        # Create weight mask:
        # (1 - u_out) is 1 during inspiration, 0 during expiration.
        # u_out is 0 during inspiration, 1 during expiration.
        weights = (1.0 - u_out) * self.w_insp + u_out * self.w_exp

        # Apply weights to the error
        weighted_error = absolute_error * weights

        # Return the mean of the weighted errors
        return weighted_error.mean()
