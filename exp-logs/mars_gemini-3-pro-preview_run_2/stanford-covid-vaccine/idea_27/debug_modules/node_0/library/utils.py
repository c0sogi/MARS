import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations where possible.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSEMetric:
    """
    Tracks the Mean Columnwise Root Mean Squared Error (MCRMSE) globally
    across batches. Accumulates Sum of Squared Errors (SSE) and counts
    to compute the exact metric at the end of an epoch, avoiding batch-averaging bias.
    """

    def __init__(self):
        # Identify indices of the columns used for scoring based on Config
        # Config.TARGET_COLS contains all 5 targets
        # Config.SCORED_COLS contains the subset (e.g., reactivity, deg_Mg_pH10, deg_Mg_50C)
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # Store SSE for each scored column separately
        self.total_sse = np.zeros(len(self.scored_indices), dtype=np.float64)
        self.total_count = 0

    def update(self, preds, targets):
        """
        Updates the metric with a new batch of predictions and targets.

        Args:
            preds: (Batch, Seq_Len, Num_Targets) tensor or numpy array.
            targets: (Batch, Seq_Len, Num_Targets) tensor or numpy array.
        """
        # Ensure inputs are numpy arrays
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Slice data to the scoring length (first 68 positions)
        # We use Config.PRED_LEN as the definitive scoring length
        score_len = Config.PRED_LEN

        # Safety check for sequence length
        current_len = preds.shape[1]
        if current_len < score_len:
            # If for some reason preds are shorter, use full length (should not happen in proper pipeline)
            score_len = current_len

        preds_sliced = preds[:, :score_len, :]
        targets_sliced = targets[:, :score_len, :]

        # Select only the specific columns required for the metric
        preds_scored = preds_sliced[:, :, self.scored_indices]
        targets_scored = targets_sliced[:, :, self.scored_indices]

        # Compute squared errors
        squared_errors = (preds_scored - targets_scored) ** 2

        # Accumulate SSE for each column (summing over batch and sequence dimensions)
        # Result is a vector of shape (num_scored_cols,)
        self.total_sse += np.sum(squared_errors, axis=(0, 1))

        # Accumulate total number of elements per column
        # (Batch_Size * Scored_Sequence_Length)
        self.total_count += preds_scored.shape[0] * preds_scored.shape[1]

    def compute(self):
        """
        Computes the final MCRMSE metric.

        Returns:
            float: The Mean Columnwise Root Mean Squared Error.
        """
        if self.total_count == 0:
            return 0.0

        # Calculate MSE per column
        mse_per_col = self.total_sse / self.total_count

        # Calculate RMSE per column
        rmse_per_col = np.sqrt(mse_per_col)

        # Calculate Mean of RMSEs (MCRMSE)
        mcrmse = np.mean(rmse_per_col)

        return mcrmse
