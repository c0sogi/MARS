import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


class GlobalMetricTracker:
    """
    Tracks the MCRMSE metric globally across batches.
    Accumulates Sum of Squared Errors (SSE) and counts to compute
    the 'Correct Global RMSE' without batch-averaging bias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal state of the tracker."""
        # sum_squared_errors will store the sum of (y - y_hat)^2 per column
        self.sum_squared_errors = 0.0
        self.total_count = 0
        self.num_targets = None

    def update(self, preds, targets):
        """
        Updates the metric tracker with a batch of predictions and targets.
        Assumes inputs are already sliced/masked to the relevant scoring positions.

        Args:
            preds (torch.Tensor or np.ndarray): Predictions. Shape (B, N, C) or (B, C).
            targets (torch.Tensor or np.ndarray): Ground truth. Shape (B, N, C) or (B, C).
        """
        # Convert torch tensors to numpy if necessary
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Ensure shapes match
        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"
            )

        # Initialize num_targets based on the last dimension (Channels)
        if self.num_targets is None:
            self.num_targets = preds.shape[-1]
        elif self.num_targets != preds.shape[-1]:
            raise ValueError(
                f"Number of targets changed. Expected {self.num_targets}, got {preds.shape[-1]}"
            )

        # Calculate squared errors
        squared_errors = (preds - targets) ** 2

        # Sum errors.
        # If shape is (B, N, C), we sum over B and N (axes 0 and 1), keeping C.
        # If shape is (B, C), we sum over B (axis 0), keeping C.
        if preds.ndim == 3:
            # (Batch, SeqLen, Channels)
            batch_sse = np.sum(squared_errors, axis=(0, 1))
            count = preds.shape[0] * preds.shape[1]
        elif preds.ndim == 2:
            # (Batch, Channels)
            batch_sse = np.sum(squared_errors, axis=0)
            count = preds.shape[0]
        else:
            raise ValueError(f"Unsupported input dimension: {preds.ndim}")

        self.sum_squared_errors += batch_sse
        self.total_count += count

    def compute(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            float: The Mean Columnwise Root Mean Squared Error.
        """
        if self.total_count == 0:
            return 0.0

        # Mean Squared Error per column
        mse_per_col = self.sum_squared_errors / self.total_count

        # Root Mean Squared Error per column
        rmse_per_col = np.sqrt(mse_per_col)

        # Mean Columnwise RMSE
        mcrmse = np.mean(rmse_per_col)

        return mcrmse
