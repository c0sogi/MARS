import os
import ast
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list_column(x):
    """
    Parses a string representation of a list (e.g., from a CSV) into a numpy array.

    Args:
        x (str): String representation of a list.

    Returns:
        np.ndarray: Numpy array of floats. Returns an empty array on failure.
    """
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


class MetricTracker:
    """
    Tracks and computes the Mean Columnwise Root Mean Squared Error (MCRMSE).
    Accumulates squared errors and counts globally to avoid batch-size bias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the tracker."""
        self.sum_sq_errors = 0.0
        self.total_count = 0.0
        self.initialized = False

    def update(self, preds, targets, mask):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (B, Seq_Len, 5).
            targets (torch.Tensor): Ground truth of shape (B, Seq_Len, 5).
            mask (torch.Tensor): Mask of shape (B, Seq_Len) indicating valid positions.
        """
        # Filter for the specific columns used in scoring
        indices = Config.SCORED_TARGET_INDICES
        preds_filtered = preds[:, :, indices]
        targets_filtered = targets[:, :, indices]

        # Ensure mask is broadcastable: (B, Seq_Len) -> (B, Seq_Len, 1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        # Calculate squared errors
        sq_diff = (preds_filtered - targets_filtered) ** 2

        # Apply mask (zero out invalid positions)
        masked_sq_diff = sq_diff * mask

        # Sum squared errors over batch and sequence dimensions
        batch_sse = masked_sq_diff.sum(dim=(0, 1))

        # Count valid positions per column
        # Expand mask to match channel dimension: (B, Seq_Len, 1) -> (B, Seq_Len, 3)
        batch_counts = mask.expand_as(preds_filtered).sum(dim=(0, 1))

        # Initialize internal storage on first update to match device/dtype
        if not self.initialized:
            self.sum_sq_errors = torch.zeros_like(batch_sse, dtype=torch.float64)
            self.total_count = torch.zeros_like(batch_counts, dtype=torch.float64)
            self.initialized = True

        # Accumulate
        self.sum_sq_errors += batch_sse.double()
        self.total_count += batch_counts.double()

    def compute(self):
        """
        Computes the final MCRMSE metric.

        Returns:
            float: The computed MCRMSE value.
        """
        if not self.initialized:
            return 0.0

        # Calculate MSE per column, adding epsilon to avoid division by zero
        mse = self.sum_sq_errors / (self.total_count + 1e-12)

        # RMSE per column
        rmse = torch.sqrt(mse)

        # MCRMSE is the mean of the column-wise RMSEs
        mcrmse = torch.mean(rmse)

        return mcrmse.item()
