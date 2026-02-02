import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricTracker:
    """
    Accumulates Sum of Squared Errors (SSE) and counts to compute
    the mathematically correct global MCRMSE across batches.
    """

    def __init__(self):
        # Determine indices of scored columns based on Config
        # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS
        self.scored_indices = [self.target_cols.index(col) for col in self.scored_cols]

        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sum_squared_errors = np.zeros(len(self.scored_cols), dtype=np.float64)
        self.n_samples = 0

    def update(self, y_true, y_pred):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            y_true: Ground truth tensor/array. Shape (Batch, SeqLen, Num_Targets) or (N, Num_Targets)
            y_pred: Prediction tensor/array. Shape (Batch, SeqLen, Num_Targets) or (N, Num_Targets)
        """
        # Ensure inputs are numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # Reshape to (-1, Num_Targets) to handle (Batch, SeqLen, Targets)
        # This flattens the batch and sequence dimensions
        if y_true.ndim > 2:
            y_true = y_true.reshape(-1, y_true.shape[-1])
        if y_pred.ndim > 2:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])

        # Select only the scored columns
        # y_true/y_pred shape becomes (N_samples, 3)
        y_true_scored = y_true[:, self.scored_indices]
        y_pred_scored = y_pred[:, self.scored_indices]

        # Calculate squared errors
        squared_errors = (y_true_scored - y_pred_scored) ** 2

        # Sum over the batch (axis 0) to get SSE per column
        batch_sse = np.sum(squared_errors, axis=0)
        batch_count = squared_errors.shape[0]

        self.sum_squared_errors += batch_sse
        self.n_samples += batch_count

    def compute(self):
        """
        Computes the global MCRMSE.

        Returns:
            float: The Mean Columnwise Root Mean Squared Error.
        """
        if self.n_samples == 0:
            return 0.0

        # Mean Squared Error per column
        mse_per_col = self.sum_squared_errors / self.n_samples

        # Root Mean Squared Error per column
        rmse_per_col = np.sqrt(mse_per_col)

        # Mean Columnwise RMSE (average across the 3 columns)
        mcrmse = np.mean(rmse_per_col)

        return mcrmse
