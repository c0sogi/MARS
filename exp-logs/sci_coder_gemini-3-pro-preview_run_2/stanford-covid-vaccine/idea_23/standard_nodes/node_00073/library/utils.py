import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class GlobalMCRMSE:
    """
    Accumulates Sum of Squared Errors (SSE) and counts across the entire validation set
    to compute the mathematically correct Mean Columnwise Root Mean Squared Error (MCRMSE).

    This class ensures that the metric is calculated globally rather than averaging
    batch-level metrics, which can introduce statistical bias.
    """

    def __init__(self):
        """
        Initializes the MCRMSE accumulator.
        """
        self.target_indices = Config.SCORED_TARGET_INDICES
        self.seq_scored = Config.SEQ_SCORED
        self.device = Config.DEVICE
        self.reset()

    def reset(self):
        """
        Resets the internal accumulators for SSE and sample counts.
        """
        # We track SSE and Count separately for each scored column
        # to compute column-wise RMSE first, then average.
        self.running_sse = None
        self.running_count = None

    def update(self, preds, targets):
        """
        Updates the running statistics with a new batch of predictions and targets.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Num_Targets).
        """
        # Ensure inputs are on the correct device
        if preds.device != self.device:
            preds = preds.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

        # 1. Slice Sequence Dimension
        # Only the first SEQ_SCORED positions are evaluated
        preds_sliced = preds[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Slice Target Dimension
        # Only specific columns (reactivity, deg_Mg_pH10, deg_Mg_50C) are scored
        preds_scored = preds_sliced[:, :, self.target_indices]
        targets_scored = targets_sliced[:, :, self.target_indices]

        # 3. Initialize accumulators if necessary
        if self.running_sse is None:
            num_scored_cols = len(self.target_indices)
            # Use float64 for precision during accumulation
            self.running_sse = torch.zeros(
                num_scored_cols, device=self.device, dtype=torch.float64
            )
            self.running_count = torch.zeros(
                num_scored_cols, device=self.device, dtype=torch.float64
            )

        # 4. Compute Squared Errors
        squared_errors = (targets_scored - preds_scored) ** 2

        # 5. Accumulate
        # Sum over Batch (dim 0) and Sequence (dim 1) dimensions
        batch_sse = squared_errors.sum(dim=(0, 1))

        # Count number of elements contributing to the sum
        # (Batch_Size * Seq_Scored)
        batch_size = preds.shape[0]
        num_elements = batch_size * self.seq_scored

        self.running_sse += batch_sse
        self.running_count += num_elements

    def compute(self):
        """
        Computes the final MCRMSE score based on accumulated statistics.

        Returns:
            float: The Mean Columnwise Root Mean Squared Error.
        """
        if self.running_sse is None or self.running_count is None:
            return 0.0

        # Avoid division by zero
        safe_count = torch.clamp(self.running_count, min=1.0)

        # Calculate MSE per column
        mse_per_col = self.running_sse / safe_count

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Calculate Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse.item()
