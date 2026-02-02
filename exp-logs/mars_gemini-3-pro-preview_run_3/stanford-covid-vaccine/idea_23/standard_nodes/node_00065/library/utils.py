import torch
import torch.nn as nn
import numpy as np
from library.config import seed_everything


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility by calling the library configuration's seeder.

    Args:
        seed (int): The seed value to set.
    """
    seed_everything(seed)


class MCRMSELoss(nn.Module):
    """
    Unweighted Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss calculates the RMSE for each target column separately and then
    takes the mean across columns. It aggregates over both batch and sequence
    dimensions to treat every position as a sample for the columnwise statistic.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Channels).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Calculate MSE for each column.
        # We average over dim=0 (Batch) and dim=1 (Sequence) to get MSE per channel.
        colwise_mse = torch.mean((inputs - targets) ** 2, dim=(0, 1))

        # Calculate RMSE for each column
        colwise_rmse = torch.sqrt(colwise_mse)

        # Return the mean of the columnwise RMSEs
        return torch.mean(colwise_rmse)


def compute_global_mcrmse(preds, targets, target_indices=None):
    """
    Computes the global MCRMSE metric for validation.

    This function explicitly concatenates all predictions and targets before
    calculation to avoid the bias introduced by averaging batch-level metrics.
    It supports filtering specific columns (e.g., for competition scoring).

    Args:
        preds (np.ndarray or list of np.ndarray): Predicted values. Can be a single
            array or a list of batch arrays. Shape: (N, Seq_Len, Channels).
        targets (np.ndarray or list of np.ndarray): Ground truth values. Can be a
            single array or a list of batch arrays. Shape: (N, Seq_Len, Channels).
        target_indices (list[int], optional): List of column indices to include
            in the calculation. If None, all columns are used.

    Returns:
        float: The global MCRMSE score.
    """
    # Concatenate list of batches if necessary
    if isinstance(preds, list):
        preds = np.concatenate(preds, axis=0)
    if isinstance(targets, list):
        targets = np.concatenate(targets, axis=0)

    # Filter for specific scored columns if indices are provided
    if target_indices is not None:
        preds = preds[..., target_indices]
        targets = targets[..., target_indices]

    # Flatten batch and sequence dimensions to treat all positions as samples
    # Shape becomes (N_total_points, N_channels)
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])

    # Calculate MSE per column across the entire dataset
    mse_per_col = np.mean((preds_flat - targets_flat) ** 2, axis=0)

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Return the mean of the columnwise RMSEs
    return np.mean(rmse_per_col)
