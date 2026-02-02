import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


class MetricTracker:
    """
    Accumulates Sum of Squared Errors (SSE) and counts to calculate
    Mean Columnwise Root Mean Squared Error (MCRMSE) globally,
    avoiding the statistical bias of batch-averaging.
    """

    def __init__(self, scored_indices=None):
        """
        Initialize the tracker.

        Args:
            scored_indices (list, optional): Indices of the columns to include in the final MCRMSE score.
                                             Defaults to Config.SCORED_INDICES.
        """
        self.scored_indices = (
            scored_indices if scored_indices is not None else Config.SCORED_INDICES
        )
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.sse = None
        self.n_samples = 0

    def update(self, y_true, y_pred):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            y_true: Ground truth values (numpy array or torch tensor).
            y_pred: Predicted values (numpy array or torch tensor).
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        # Reshape to (N_total_positions, N_targets) to handle (Batch, Seq, Target) or (Batch, Target) inputs
        # We flatten the batch and sequence dimensions together
        num_targets = y_true.shape[-1]
        y_true_flat = y_true.reshape(-1, num_targets)
        y_pred_flat = y_pred.reshape(-1, num_targets)

        # Calculate Sum of Squared Errors for this batch
        batch_sse = np.sum((y_true_flat - y_pred_flat) ** 2, axis=0)
        batch_count = y_true_flat.shape[0]

        if self.sse is None:
            self.sse = batch_sse
        else:
            self.sse += batch_sse

        self.n_samples += batch_count

    def result(self):
        """
        Computes the global MCRMSE over the accumulated data.

        Returns:
            float: The MCRMSE value calculated on the specified scored columns.
        """
        if self.n_samples == 0:
            return 0.0

        # Mean Squared Error per column
        mse = self.sse / self.n_samples

        # Root Mean Squared Error per column
        rmse = np.sqrt(mse)

        # Filter for scored columns and compute mean
        # We check if indices are valid to avoid errors if dimensions mismatch
        valid_indices = [i for i in self.scored_indices if i < len(rmse)]
        if not valid_indices:
            return 0.0

        scored_rmse = rmse[valid_indices]
        mcrmse = np.mean(scored_rmse)

        return mcrmse
