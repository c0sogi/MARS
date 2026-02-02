import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricTracker:
    """
    Accumulates statistics to calculate the global MCRMSE (Mean Columnwise Root Mean Squared Error)
    over the entire validation set, avoiding batch-averaging bias.

    Only tracks the columns specified in Config.SCORED_TARGET_INDICES.
    """

    def __init__(self):
        self.reset()
        self.scored_indices = Config.SCORED_TARGET_INDICES
        self.num_scored = len(self.scored_indices)

    def reset(self):
        """Resets the internal state."""
        self.sse = np.zeros(len(Config.SCORED_TARGET_INDICES), dtype=np.float64)
        self.count = 0

    def update(self, y_true, y_pred):
        """
        Updates the running statistics with a new batch of predictions.

        Args:
            y_true (torch.Tensor or np.ndarray): Ground truth values.
                                                 Shape: (Batch, Seq_Len, 5) or (N, 5)
            y_pred (torch.Tensor or np.ndarray): Predicted values.
                                                 Shape: (Batch, Seq_Len, 5) or (N, 5)
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # Flatten if necessary to (N_samples, N_targets)
        # We assume the last dimension is the target dimension (size 5)
        if y_true.ndim == 3:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])

        # Filter only the scored columns
        y_true_scored = y_true[:, self.scored_indices]
        y_pred_scored = y_pred[:, self.scored_indices]

        # Calculate squared errors
        squared_errors = (y_true_scored - y_pred_scored) ** 2

        # Accumulate Sum of Squared Errors per column
        self.sse += np.sum(squared_errors, axis=0)

        # Accumulate count of samples (rows)
        self.count += y_true_scored.shape[0]

    def result(self):
        """
        Calculates the final metrics based on accumulated data.

        Returns:
            dict: Dictionary containing 'mcrmse' and individual column RMSEs.
        """
        if self.count == 0:
            return {"mcrmse": 0.0}

        # Calculate RMSE for each scored column: sqrt(Sum(Errors^2) / N)
        column_rmses = np.sqrt(self.sse / self.count)

        # Calculate MCRMSE: Mean of the column RMSEs
        mcrmse = np.mean(column_rmses)

        metrics = {"mcrmse": mcrmse}

        # Add individual column metrics for debugging/analysis
        # Mapping index to name based on Config
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Config.SCORED_TARGET_INDICES = [0, 1, 3]

        col_names = [Config.TARGET_COLS[i] for i in self.scored_indices]
        for name, val in zip(col_names, column_rmses):
            metrics[f"rmse_{name}"] = val

        return metrics
