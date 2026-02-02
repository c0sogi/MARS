import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for CuDNN backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricTracker:
    """
    Computes MCRMSE (Mean Columnwise Root Mean Squared Error) by accumulating
    squared errors and counts globally across batches. This avoids the bias
    introduced by averaging RMSEs calculated on small batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self._col_sse = None
        self._col_count = None

    def update(self, y_pred, y_true):
        """
        Updates the running statistics with a new batch of data.

        Args:
            y_pred (torch.Tensor or np.ndarray): Predictions.
                Expected shape: (Batch, Seq_Scored, Channels) or (N, Channels).
            y_true (torch.Tensor or np.ndarray): Ground truth.
                Expected shape: (Batch, Seq_Scored, Channels) or (N, Channels).

        Note:
            Inputs should already be filtered to the scored sequence positions
            (e.g., first 68) and the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
            before being passed to this method.
        """
        # Convert tensors to numpy
        if torch.is_tensor(y_pred):
            y_pred = y_pred.detach().cpu().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.detach().cpu().numpy()

        # Flatten spatial dimensions to (N, Channels) if necessary
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y_true = y_true.reshape(-1, y_true.shape[-1])

        # Initialize counters if this is the first update
        if self._col_sse is None:
            num_cols = y_pred.shape[1]
            self._col_sse = np.zeros(num_cols, dtype=np.float64)
            self._col_count = np.zeros(num_cols, dtype=np.float64)

        # Accumulate Sum of Squared Errors (SSE) and Counts per column
        squared_diff = (y_true - y_pred) ** 2
        self._col_sse += np.sum(squared_diff, axis=0)
        self._col_count += squared_diff.shape[0]

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            mcrmse (float): The mean of the column-wise RMSEs.
        """
        # Handle edge case where no data has been updated
        if self._col_count is None or np.any(self._col_count == 0):
            return 0.0

        # Calculate RMSE per column: sqrt(Sum(SSE) / Sum(Count))
        mse = self._col_sse / self._col_count
        rmse = np.sqrt(mse)

        # Calculate MCRMSE: Mean of column RMSEs
        mcrmse = np.mean(rmse)
        return mcrmse
