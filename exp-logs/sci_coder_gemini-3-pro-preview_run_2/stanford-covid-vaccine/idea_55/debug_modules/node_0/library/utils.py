import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The selected device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricTracker:
    """
    Accumulates squared errors and counts to compute the global MCRMSE
    (Mean Columnwise Root Mean Squared Error) over the entire validation set.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        # Accumulator for Sum of Squared Errors per scored target column
        self.sse = None
        # Accumulator for the total count of valid predictions per column
        self.count = 0

    def update(self, preds, targets):
        """
        Updates the metrics with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Model predictions of shape (N, 5, L).
            targets (torch.Tensor): Ground truth targets of shape (N, L, 5).
        """
        # Ensure tensors are on CPU to save GPU memory during accumulation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Align target shape to (N, 5, L) if necessary
        if targets.shape[1] != 5 and targets.shape[2] == 5:
            targets = targets.permute(0, 2, 1)

        # Select only the scored target columns defined in Config
        # Indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        preds_scored = preds[:, Config.SCORED_TARGETS, :]
        targets_scored = targets[:, Config.SCORED_TARGETS, :]

        # Select only the scored sequence positions (first 68 bases)
        preds_scored = preds_scored[:, :, : Config.PRED_LEN]
        targets_scored = targets_scored[:, :, : Config.PRED_LEN]

        # Calculate squared errors: (N, num_scored, pred_len)
        squared_errors = (preds_scored - targets_scored) ** 2

        # Sum errors over batch (dim 0) and sequence length (dim 2)
        # Result shape: (num_scored,)
        batch_sse = torch.sum(squared_errors, dim=(0, 2))

        # Calculate total number of elements contributing to the sum per column
        batch_count = preds_scored.shape[0] * preds_scored.shape[2]

        # Initialize sse accumulator if it's the first update
        if self.sse is None:
            self.sse = torch.zeros_like(batch_sse)

        self.sse += batch_sse
        self.count += batch_count

    def result(self):
        """
        Computes the final MCRMSE metric based on accumulated data.

        Returns:
            float: The global MCRMSE value.
        """
        if self.count == 0:
            return 0.0

        # Calculate Mean Squared Error per column
        mse = self.sse / self.count

        # Calculate Root Mean Squared Error per column
        rmse = torch.sqrt(mse)

        # Calculate Mean of the column-wise RMSEs
        mcrmse = torch.mean(rmse)

        return mcrmse.item()
