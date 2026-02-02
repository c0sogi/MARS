import torch
import torch.nn as nn
from library.config import INSPIRATORY_WEIGHT, EXPIRATORY_WEIGHT


class WeightedL1Loss(nn.Module):
    """
    Custom Weighted L1 Loss for Ventilator Pressure Prediction.

    Applies different weights to the inspiratory and expiratory phases of the breath.
    - Inspiratory (u_out=0): High weight (e.g., 1.0) to optimize for the competition metric.
    - Expiratory (u_out=1): Low weight (e.g., 0.1) to provide auxiliary supervision
      and maintain hidden state stability without dominating the gradient.
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.w_in = INSPIRATORY_WEIGHT
        self.w_out = EXPIRATORY_WEIGHT
        # Use reduction='none' to apply element-wise weighting before averaging
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds (torch.Tensor): Model predictions. Shape (Batch, Seq_Len) or (Batch, Seq_Len, 1).
            targets (torch.Tensor): Ground truth pressure. Shape (Batch, Seq_Len).
            u_out (torch.Tensor): Expiratory valve status (0 or 1). Shape (Batch, Seq_Len).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure predictions and targets have matching shapes (Batch, Seq_Len)
        if preds.dim() == 3:
            preds = preds.squeeze(-1)

        # Calculate element-wise L1 loss
        raw_loss = self.l1(preds, targets)

        # Create weight mask based on u_out
        # u_out == 0 -> Inspiratory -> weight = w_in
        # u_out == 1 -> Expiratory  -> weight = w_out
        # Since u_out is binary (0.0 or 1.0), we can use arithmetic:
        weights = (1.0 - u_out) * self.w_in + u_out * self.w_out

        # Apply weights
        weighted_loss = raw_loss * weights

        # Return the mean loss over the batch
        return weighted_loss.mean()


def compute_mae(preds, targets, u_out):
    """
    Calculates the Mean Absolute Error (MAE) strictly for the inspiratory phase.
    This aligns with the competition leaderboard metric.

    Args:
        preds (torch.Tensor or np.ndarray): Model predictions.
        targets (torch.Tensor or np.ndarray): Ground truth pressure.
        u_out (torch.Tensor or np.ndarray): Expiratory valve status.

    Returns:
        float: MAE for the inspiratory phase.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    if not isinstance(u_out, torch.Tensor):
        u_out = torch.tensor(u_out)

    # Ensure tensors are on CPU for metric calculation to avoid sync overhead if just logging
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    u_out = u_out.detach().cpu()

    # Align shapes
    if preds.dim() == 3:
        preds = preds.squeeze(-1)

    # Flatten for boolean masking
    preds = preds.flatten()
    targets = targets.flatten()
    u_out = u_out.flatten()

    # Create mask for inspiratory phase (u_out == 0)
    # Using a small epsilon for float comparison safety, though 0.0 is usually exact
    inspiratory_mask = u_out < 0.5

    # If no inspiratory phase data exists (edge case), return 0.0
    if inspiratory_mask.sum() == 0:
        return 0.0

    # Filter data
    preds_in = preds[inspiratory_mask]
    targets_in = targets[inspiratory_mask]

    # Calculate MAE
    mae = torch.abs(preds_in - targets_in).mean()

    return mae.item()
