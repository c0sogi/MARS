import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricCalculator:
    """
    Handles metric calculations for the RNA degradation task.
    Implements MCRMSE (Mean Columnwise Root Mean Squared Error).
    """

    def __init__(self, config: Config):
        self.config = config
        self.target_cols = config.TARGET_COLS
        self.scored_cols = config.SCORED_COLS

        # Identify indices of the columns that are actually scored in the competition
        self.scored_indices = [self.target_cols.index(col) for col in self.scored_cols]

    def compute_train_loss(self, y_pred, y_true):
        """
        Computes MCRMSE on all 5 columns for training purposes.

        Args:
            y_pred (torch.Tensor): Predictions of shape (Batch, Seq_Len_Pred, 5).
            y_true (torch.Tensor): Ground truth of shape (Batch, Seq_Len_True, 5).
                                   Seq_Len_True is typically 68.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice predictions to match the length of ground truth (seq_scored=68)
        seq_scored = y_true.shape[1]
        y_pred_sliced = y_pred[:, :seq_scored, :]

        # Calculate MCRMSE over all columns
        return self._mcrmse(y_pred_sliced, y_true)

    def compute_val_metric(self, y_pred, y_true):
        """
        Computes MCRMSE on only the 3 scored columns for validation/scoring.

        Args:
            y_pred (torch.Tensor): Predictions of shape (Batch, Seq_Len_Pred, 5).
            y_true (torch.Tensor): Ground truth of shape (Batch, Seq_Len_True, 5).

        Returns:
            float: The MCRMSE score.
        """
        # Slice predictions to match the length of ground truth (seq_scored=68)
        seq_scored = y_true.shape[1]
        y_pred_sliced = y_pred[:, :seq_scored, :]

        # Calculate MCRMSE only for the scored columns
        metric_tensor = self._mcrmse(y_pred_sliced, y_true, indices=self.scored_indices)
        return metric_tensor.item()

    def _mcrmse(self, pred, true, indices=None):
        """
        Internal function to calculate MCRMSE.

        Args:
            pred (torch.Tensor): Sliced predictions (Batch, Seq, Channels).
            true (torch.Tensor): Ground truth (Batch, Seq, Channels).
            indices (list, optional): List of channel indices to include.

        Returns:
            torch.Tensor: Scalar MCRMSE.
        """
        # Flatten batch and sequence dimensions: (N, Channels)
        # N = Batch * Seq
        pred_flat = pred.reshape(-1, pred.shape[-1])
        true_flat = true.reshape(-1, true.shape[-1])

        # Calculate MSE for each column
        mse = torch.mean((pred_flat - true_flat) ** 2, dim=0)

        # Calculate RMSE for each column
        rmse = torch.sqrt(mse)

        # Filter specific columns if requested
        if indices is not None:
            rmse = rmse[indices]

        # Return the mean of the column RMSEs
        return torch.mean(rmse)
