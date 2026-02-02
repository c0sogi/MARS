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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GlobalMCRMSE:
    """
    Accumulates Sum of Squared Errors (SSE) and counts over the entire validation set
    to compute the MCRMSE (Mean Columnwise Root Mean Squared Error) correctly,
    avoiding the statistical bias of averaging batch-level RMSEs.
    """

    def __init__(self):
        self.reset()
        # Scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        self.scored_indices = [0, 1, 3]
        self.scored_len = Config.SCORED_LEN

    def reset(self):
        """Resets the internal state."""
        self.total_sse = 0.0
        self.total_count = 0

    def update(self, preds, targets):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predictions of shape (B, L, 5)
            targets (torch.Tensor): Ground truth of shape (B, L, 5)
        """
        # Ensure inputs are on the same device (CPU is sufficient for accumulation)
        # Using detach() to prevent graph retention
        preds = preds.detach()
        targets = targets.detach()

        # 1. Slice to the scored sequence length (first 68 positions)
        # Shape becomes (B, SCORED_LEN, 5)
        preds_valid = preds[:, : self.scored_len, :]
        targets_valid = targets[:, : self.scored_len, :]

        # 2. Select only the scored columns [0, 1, 3]
        # Shape becomes (B, SCORED_LEN, 3)
        preds_scored = preds_valid[:, :, self.scored_indices]
        targets_scored = targets_valid[:, :, self.scored_indices]

        # 3. Compute Squared Errors
        squared_diff = (preds_scored - targets_scored) ** 2

        # 4. Accumulate SSE per column
        # Sum over Batch (0) and Length (1), keeping Column dimension (2)
        # Result shape: (3,)
        batch_sse = torch.sum(squared_diff, dim=(0, 1))

        # 5. Accumulate Count
        # Number of elements per column = Batch Size * Scored Length
        batch_count = preds_scored.shape[0] * preds_scored.shape[1]

        if isinstance(self.total_sse, float):
            self.total_sse = batch_sse
        else:
            self.total_sse += batch_sse

        self.total_count += batch_count

    def compute(self):
        """
        Computes the final MCRMSE metric.

        Returns:
            float: The mean columnwise RMSE.
        """
        if self.total_count == 0:
            return 0.0

        # MSE per column
        mse = self.total_sse / self.total_count

        # RMSE per column
        rmse = torch.sqrt(mse)

        # Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse)

        return mcrmse.item()
