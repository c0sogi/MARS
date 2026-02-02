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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSEMetric:
    """
    Accumulates Sum of Squared Errors (SSE) and counts over batches to compute
    the global Mean Columnwise Root Mean Squared Error (MCRMSE) on the scored positions.
    """

    def __init__(self):
        # Determine indices of scored columns based on Config
        # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        self.reset()

    def reset(self):
        """Resets the internal state."""
        # Store SSE as float64 for precision
        self.total_sse = torch.zeros(len(self.scored_indices), dtype=torch.float64)
        self.total_count = 0

    def update(self, preds, targets):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets)
        """
        # Ensure inputs are on CPU and detached for accumulation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # 1. Slice to the scored sequence length (e.g., first 68 positions)
        preds_sliced = preds[:, : Config.SCORED_LEN, :]
        targets_sliced = targets[:, : Config.SCORED_LEN, :]

        # 2. Select only the scored columns
        preds_filtered = preds_sliced[:, :, self.scored_indices]
        targets_filtered = targets_sliced[:, :, self.scored_indices]

        # 3. Compute Squared Errors
        squared_errors = (preds_filtered - targets_filtered) ** 2

        # 4. Accumulate SSE per column (sum over batch and sequence dimensions)
        # Shape of squared_errors: (Batch, Scored_Len, Num_Scored_Cols)
        batch_sse = squared_errors.sum(dim=(0, 1))

        # Count is Batch * Scored_Len
        batch_count = squared_errors.shape[0] * squared_errors.shape[1]

        self.total_sse += batch_sse.double()
        self.total_count += batch_count

    def compute(self):
        """
        Computes the final MCRMSE metric.

        Returns:
            float: The mean columnwise root mean squared error.
        """
        if self.total_count == 0:
            return 0.0

        # Calculate MSE per column
        mse_per_col = self.total_sse / self.total_count

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Calculate Mean of RMSEs
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse.item()
