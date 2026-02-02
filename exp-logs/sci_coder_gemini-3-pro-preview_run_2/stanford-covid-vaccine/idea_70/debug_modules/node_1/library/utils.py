import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior in cuDNN backends.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Custom MCRMSE Loss that implements strict masking for the RHI-DFN architecture.

    Logic:
    1. Sequence Masking: Only considers the first `seq_scored` (68) positions.
       Ignores the zero-padded tail (indices 68-106) to prevent zero-anchoring bias
       (Lesson 00139), ensuring gradients are only derived from valid physics.
    2. Column Masking: Only considers the scored columns defined in Config.SCORED_INDICES
       (reactivity, deg_Mg_pH10, deg_Mg_50C). Unscored columns are ignored.

    Formula:
    Mean Columnwise Root Mean Squared Error.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED
        self.scored_indices = Config.SCORED_INDICES

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predictions of shape (Batch, Seq_Len, 5)
            targets: Ground truth of shape (Batch, Seq_Len, 5)

        Returns:
            mcrmse: Scalar tensor representing the loss.
        """
        # 1. Sequence Masking: Slice to valid scored length (0 to 67)
        # Shape: (Batch, 68, 5)
        preds_sliced = preds[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Column Masking: Select only scored columns (indices 0, 1, 3)
        # Shape: (Batch, 68, 3)
        preds_scored = preds_sliced[:, :, self.scored_indices]
        targets_scored = targets_sliced[:, :, self.scored_indices]

        # 3. Compute MSE per column
        # Average over Batch and Sequence dimensions
        # Result Shape: (3,)
        mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Average RMSE across columns to get final scalar
        mcrmse = torch.mean(rmse)

        return mcrmse


class GlobalMCRMSE:
    """
    Accumulator for calculating the 'Correct Global RMSE' over the entire validation set.

    Rationale:
    Simple averaging of batch-wise RMSEs introduces a bias because batch sizes might vary
    or the metric is non-linear (sqrt). The correct approach is to accumulate
    Sum of Squared Errors (SSE) and element counts across all batches, then compute the
    square root at the very end (Lesson 00131).
    """

    def __init__(self):
        self.seq_scored = Config.SEQ_SCORED
        self.scored_indices = Config.SCORED_INDICES
        self.reset()

    def reset(self):
        """Resets the internal state of the accumulator."""
        self.sse = None  # Will be a tensor of shape (Num_Scored_Cols,)
        self.total_count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Updates the accumulator with a batch of predictions and targets.

        Args:
            preds: Predictions of shape (Batch, Seq_Len, 5)
            targets: Ground truth of shape (Batch, Seq_Len, 5)
        """
        # Detach and move to CPU to save GPU memory and facilitate accumulation
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # 1. Sequence Masking
        preds_sliced = preds[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Column Masking
        preds_scored = preds_sliced[:, :, self.scored_indices]
        targets_scored = targets_sliced[:, :, self.scored_indices]

        # 3. Calculate Squared Errors
        squared_errors = (preds_scored - targets_scored) ** 2

        # 4. Sum over Batch and Sequence dimensions
        # Result Shape: (3,)
        batch_sse = torch.sum(squared_errors, dim=(0, 1))

        # 5. Accumulate
        if self.sse is None:
            self.sse = torch.zeros_like(batch_sse)

        self.sse += batch_sse
        self.total_count += preds_scored.shape[0] * preds_scored.shape[1]

    def compute(self) -> float:
        """
        Computes the final Global MCRMSE based on accumulated stats.

        Returns:
            float: The MCRMSE value.
        """
        if self.total_count == 0:
            return 0.0

        # MSE per column
        mse = self.sse / self.total_count

        # RMSE per column
        rmse = torch.sqrt(mse)

        # Mean Columnwise RMSE
        mcrmse = torch.mean(rmse)

        return mcrmse.item()
