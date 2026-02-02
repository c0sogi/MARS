import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


class MetricTracker:
    """
    Tracks and computes the Global Mean Columnwise Root Mean Squared Error (MCRMSE).

    Accumulates Sum of Squared Errors (SSE) and counts for each scored column
    across batches to calculate the metric over the entire dataset, rather than
    averaging batch-level metrics.
    """

    def __init__(self):
        """
        Initialize the tracker. Identifies which columns are scored based on Config.
        """
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS

        # Identify indices of columns that are used for scoring
        self.scored_indices = [
            i for i, col in enumerate(self.target_cols) if col in self.scored_cols
        ]

        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # Dictionary to store Sum of Squared Errors for each scored column index
        self.sse = {i: 0.0 for i in self.scored_indices}
        # Dictionary to store count of samples for each scored column index
        self.counts = {i: 0 for i in self.scored_indices}

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update the metrics with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predicted values. Shape (Batch, Seq, Num_Targets) or (N, Num_Targets).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq, Num_Targets) or (N, Num_Targets).
        """
        # Detach and move to CPU numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Align sequence lengths if both are 3D (Batch, Seq, Channels)
        if preds.ndim == 3 and targets.ndim == 3:
            if preds.shape[1] > targets.shape[1]:
                preds = preds[:, : targets.shape[1], :]
            elif preds.shape[1] < targets.shape[1]:
                targets = targets[:, : preds.shape[1], :]

        # Flatten spatial dimensions if present to (N_samples, N_channels)
        if preds.ndim == 3:
            preds = preds.reshape(-1, preds.shape[-1])
        if targets.ndim == 3:
            targets = targets.reshape(-1, targets.shape[-1])

        # Iterate over only the scored columns
        for i in self.scored_indices:
            p_col = preds[:, i]
            t_col = targets[:, i]

            # Create a mask for valid targets (ignoring NaNs if any exist in data)
            # In this dataset, targets are generally dense, but this adds robustness.
            mask = ~np.isnan(t_col)

            if np.any(mask):
                diff = p_col[mask] - t_col[mask]
                squared_error = np.sum(diff**2)
                count = len(diff)

                self.sse[i] += squared_error
                self.counts[i] += count

    def result(self) -> float:
        """
        Calculate the final MCRMSE score.

        Returns:
            float: The Mean Columnwise Root Mean Squared Error.
        """
        rmses = []
        for i in self.scored_indices:
            count = self.counts[i]
            if count > 0:
                mse = self.sse[i] / count
                rmse = np.sqrt(mse)
                rmses.append(rmse)
            else:
                # If no samples were seen for a column, we assume 0 error or handle gracefully
                rmses.append(0.0)

        if not rmses:
            return 0.0

        # MCRMSE is the mean of the RMSEs of the scored columns
        mcrmse = np.mean(rmses)
        return float(mcrmse)
