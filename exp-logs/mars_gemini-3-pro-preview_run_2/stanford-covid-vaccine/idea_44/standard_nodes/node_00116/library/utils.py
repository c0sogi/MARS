import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GlobalMetrics:
    """
    Accumulates metrics over the entire validation set to compute the
    Correct Global MCRMSE, avoiding batch-averaging bias.
    """

    def __init__(self, scored_indices=None, seq_scored=None):
        """
        Args:
            scored_indices (list): Indices of the columns to score.
                                   Defaults to Config.SCORED_INDICES.
            seq_scored (int): Number of positions at the start of the sequence to score.
                              Defaults to Config.SEQ_SCORED.
        """
        self.scored_indices = (
            scored_indices if scored_indices is not None else Config.SCORED_INDICES
        )
        self.seq_scored = seq_scored if seq_scored is not None else Config.SEQ_SCORED

        # We track SSE and Count per scored column to compute column-wise RMSE first
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # sse_per_col: Sum of Squared Errors for each scored column
        self.sse_per_col = torch.zeros(len(self.scored_indices), dtype=torch.float64)
        self.count = 0

    def update(self, y_true, y_pred):
        """
        Updates the metrics with a new batch of predictions.

        Args:
            y_true (torch.Tensor): Ground truth tensor. Shape (B, Seq_Len, 5) or (B, Seq_Scored, 5).
            y_pred (torch.Tensor): Prediction tensor. Shape (B, Seq_Len, 5).
        """
        # Ensure inputs are on CPU for accumulation to save GPU memory and avoid sync issues
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()

        # 1. Slice to the scored sequence length (e.g., first 68 bases)
        # We assume y_true has at least seq_scored length.
        # y_pred is typically full length (107), so we crop it.
        y_true_sliced = y_true[:, : self.seq_scored, :]
        y_pred_sliced = y_pred[:, : self.seq_scored, :]

        # 2. Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # scored_indices maps to the indices in the 5-channel output
        y_true_filtered = y_true_sliced[:, :, self.scored_indices]
        y_pred_filtered = y_pred_sliced[:, :, self.scored_indices]

        # 3. Compute Squared Errors
        squared_errors = (y_true_filtered - y_pred_filtered) ** 2

        # 4. Accumulate Sum of Squared Errors per column
        # Sum over Batch and Sequence dimensions
        batch_sse = torch.sum(squared_errors, dim=(0, 1))

        self.sse_per_col += batch_sse

        # 5. Accumulate count of valid elements
        # Total elements = Batch_Size * Seq_Scored
        # We assume no NaNs in the scored region of y_true for this task.
        batch_size = y_true.shape[0]
        self.count += batch_size * self.seq_scored

    def compute(self):
        """
        Computes the final MCRMSE metric.

        MCRMSE = Mean( RMSE(col_1), RMSE(col_2), ... )

        Returns:
            float: The mean columnwise root mean squared error.
        """
        if self.count == 0:
            return 0.0

        # Compute MSE per column
        mse_per_col = self.sse_per_col / self.count

        # Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Compute Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse.item()
