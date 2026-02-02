import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility across
    runs. Sets seeds for random, os, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSE:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) globally
    across the entire dataset.

    This class accumulates squared errors and counts for each batch and
    computes the final metric only when requested. This avoids the bias
    introduced by averaging RMSEs calculated per-batch.
    """

    def __init__(self, scored_indices=[0, 1, 3]):
        """
        Args:
            scored_indices (list): List of column indices to include in the metric.
                                   Default is [0, 1, 3] corresponding to:
                                   reactivity, deg_Mg_pH10, deg_Mg_50C.
        """
        self.scored_indices = scored_indices
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sum_squared_errors = {idx: 0.0 for idx in self.scored_indices}
        self.counts = {idx: 0 for idx in self.scored_indices}

    def update(self, preds, targets, mask=None):
        """
        Updates the metric with a new batch of predictions.

        Args:
            preds (torch.Tensor or np.ndarray): Predictions of shape (B, L, C).
            targets (torch.Tensor or np.ndarray): Ground truth of shape (B, L, C).
            mask (torch.Tensor or np.ndarray, optional): Boolean or binary mask of shape (B, L).
                                                         1 indicates a valid position to score.
        """
        # Ensure inputs are numpy arrays on CPU
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        for idx in self.scored_indices:
            # Extract specific column
            p = preds[:, :, idx]
            t = targets[:, :, idx]

            # Calculate squared error
            diff_sq = (p - t) ** 2

            if mask is not None:
                # Apply mask
                valid_mask = mask.astype(bool)
                # Select only valid positions
                valid_diff_sq = diff_sq[valid_mask]

                self.sum_squared_errors[idx] += np.sum(valid_diff_sq)
                self.counts[idx] += valid_diff_sq.size
            else:
                self.sum_squared_errors[idx] += np.sum(diff_sq)
                self.counts[idx] += diff_sq.size

    def compute(self):
        """
        Computes the final MCRMSE score based on accumulated data.

        Returns:
            float: The mean of the RMSEs of the scored columns.
        """
        column_rmses = []
        for idx in self.scored_indices:
            total_sse = self.sum_squared_errors[idx]
            count = self.counts[idx]

            if count > 0:
                # Global RMSE for this column
                rmse = np.sqrt(total_sse / count)
            else:
                rmse = 0.0
            column_rmses.append(rmse)

        # MCRMSE is the mean of the column RMSEs
        return np.mean(column_rmses)
