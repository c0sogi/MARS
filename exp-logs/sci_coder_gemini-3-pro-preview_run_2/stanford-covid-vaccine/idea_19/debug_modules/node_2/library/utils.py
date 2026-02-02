import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


class GlobalMCRMSE:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) globally
    across the entire dataset, avoiding batch-averaging bias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sum_sq_errors = None
        self.n_items = 0

    def update(self, preds, targets):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds: Predicted values (Tensor or numpy array).
                   Expected shape: (Batch, SeqLen, N_Targets) or (N_Samples, N_Targets).
            targets: Ground truth values (Tensor or numpy array).
                     Expected shape matches preds.
        """
        # Ensure inputs are numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Initialize storage if this is the first update
        # We assume the last dimension is the number of target columns
        if self.sum_sq_errors is None:
            self.sum_sq_errors = np.zeros(preds.shape[-1], dtype=np.float64)

        # Compute squared differences
        diff = preds - targets
        sq_diff = diff**2

        # Sum squared errors across all dimensions except the last (columns)
        # If shape is (Batch, Seq, Cols), we sum over (0, 1)
        # If shape is (N, Cols), we sum over (0)
        reduction_axes = tuple(range(preds.ndim - 1))

        self.sum_sq_errors += np.sum(sq_diff, axis=reduction_axes)

        # Count total number of scalar elements per column contributing to the sum
        # Since the shape is rectangular, this is product of dimensions excluding the last
        self.n_items += np.prod(preds.shape[:-1])

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            float: The MCRMSE value.
        """
        if self.n_items == 0:
            return 0.0

        # Calculate Mean Squared Error per column
        mse = self.sum_sq_errors / self.n_items

        # Calculate Root Mean Squared Error per column
        rmse = np.sqrt(mse)

        # Calculate Mean of RMSEs across columns (MCRMSE)
        mcrmse = np.mean(rmse)

        return mcrmse
