import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricTracker:
    """
    Statefully tracks the Mean Columnwise Root Mean Squared Error (MCRMSE)
    across batches to compute the correct global metric at the end of an epoch.
    """

    def __init__(self):
        # Identify the indices of the columns that are actually scored
        # Config.TARGET_COLS contains all 5 targets
        # Config.SCORED_COLS contains the subset (reactivity, deg_Mg_pH10, deg_Mg_50C)
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS
        self.scored_indices = [self.target_cols.index(col) for col in self.scored_cols]
        self.reset()

    def reset(self):
        """Resets the internal state."""
        # Sum of Squared Errors for each scored column
        self.sse = np.zeros(len(self.scored_cols), dtype=np.float64)
        # Total count of elements (pixels/bases) processed per column
        self.count = 0

    def update(self, preds, targets):
        """
        Updates the metric with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor or np.ndarray): Predictions of shape (B, L, C)
            targets (torch.Tensor or np.ndarray): Ground truth of shape (B, L, C)
        """
        # Ensure inputs are numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Slice to the scored sequence length (first 68 positions)
        # The inputs might be length 107, but we only score the first 68.
        if preds.shape[1] > Config.SCORED_SEQ_LENGTH:
            preds = preds[:, : Config.SCORED_SEQ_LENGTH, :]
        if targets.shape[1] > Config.SCORED_SEQ_LENGTH:
            targets = targets[:, : Config.SCORED_SEQ_LENGTH, :]

        # Filter for the specific columns that are scored
        # preds/targets shape becomes (B, 68, 3) typically
        preds_scored = preds[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # Compute squared errors
        squared_diff = (targets_scored - preds_scored) ** 2

        # Sum errors over batch and sequence length, keeping column separation
        # Result shape: (num_scored_cols,)
        batch_sse = np.sum(squared_diff, axis=(0, 1))

        # Update state
        self.sse += batch_sse
        # Count is Batch_Size * Sequence_Length
        self.count += preds_scored.shape[0] * preds_scored.shape[1]

    def result(self):
        """
        Computes the final MCRMSE based on accumulated data.

        Returns:
            float: The Mean Columnwise RMSE.
        """
        if self.count == 0:
            return 0.0

        # Mean Squared Error per column
        mse = self.sse / self.count

        # Root Mean Squared Error per column
        rmse = np.sqrt(mse)

        # Mean of the RMSEs (MCRMSE)
        mcrmse = np.mean(rmse)

        return mcrmse
