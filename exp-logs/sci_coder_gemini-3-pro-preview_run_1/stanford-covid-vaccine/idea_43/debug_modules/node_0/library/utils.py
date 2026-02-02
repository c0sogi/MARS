import torch
import torch.nn as nn
import numpy as np
from library.config import Config, set_seed


def mcrmse_metric(preds, targets, scored_len=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each target column independently over all
    samples and positions, and then averages these RMSE values. This avoids the
    'Mean of Sqrts' artifact where RMSE is averaged per sample.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (B, Seq_Len, C) or (B, C).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (B, Seq_Len, C) or (B, C).
        scored_len (int, optional): Number of positions to score along the sequence dimension.
                                    Defaults to Config.SCORED_LEN (68).

    Returns:
        float: The MCRMSE score.
    """
    if scored_len is None:
        scored_len = Config.SCORED_LEN

    # Convert numpy arrays to torch tensors
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Detach and move to CPU to prevent memory accumulation during validation
    preds = preds.detach().cpu().float()
    targets = targets.detach().cpu().float()

    # Handle 3D inputs: (Batch, Sequence, Channels)
    if preds.dim() == 3:
        # Slice predictions and targets to the scored length
        if preds.shape[1] > scored_len:
            preds = preds[:, :scored_len, :]
        if targets.shape[1] > scored_len:
            targets = targets[:, :scored_len, :]

        # Flatten batch and sequence dimensions to treat all positions as samples
        # Shape becomes (N_total_positions, Channels)
        preds = preds.reshape(-1, preds.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

    # Calculate Mean Squared Error for each column
    mse = torch.mean((preds - targets) ** 2, dim=0)

    # Calculate Root Mean Squared Error for each column
    rmse = torch.sqrt(mse)

    # Calculate the mean of the column RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse.item()


class MCRMSELoss(nn.Module):
    """
    MCRMSE Loss function implemented as a PyTorch Module.

    Calculates the Mean Columnwise Root Mean Squared Error.
    Useful if one wishes to optimize this metric directly, though MSE is
    generally preferred for stability.
    """

    def __init__(self, scored_len=None):
        super().__init__()
        self.scored_len = scored_len if scored_len is not None else Config.SCORED_LEN

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (B, L, C).
            targets (torch.Tensor): Ground truth of shape (B, L, C).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice to scored length
        if preds.shape[1] > self.scored_len:
            preds = preds[:, : self.scored_len, :]
        if targets.shape[1] > self.scored_len:
            targets = targets[:, : self.scored_len, :]

        # Flatten batch and sequence dimensions
        preds_flat = preds.reshape(-1, preds.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])

        # Calculate MSE per column
        mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)

        # Calculate RMSE per column (adding epsilon for gradient stability)
        rmse = torch.sqrt(mse + 1e-8)

        # Average the RMSEs
        loss = torch.mean(rmse)

        return loss
