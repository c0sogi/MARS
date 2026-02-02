import os
import ast
import random
import numpy as np
import torch
from library.config import SCORED_TARGETS, ALL_TARGETS


def parse_list_column(x):
    """
    Parses a string representation of a list into a numpy float32 array.
    Handles standard list strings and potential malformed inputs.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=np.float32)

    if isinstance(x, str):
        try:
            # Safely evaluate the string literal
            val = ast.literal_eval(x)
            return np.array(val, dtype=np.float32)
        except (ValueError, SyntaxError):
            # Return empty array on failure
            return np.array([], dtype=np.float32)

    return np.array([], dtype=np.float32)


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)


class GlobalMCRMSE:
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) globally
    across the entire dataset, avoiding batch-averaging bias.

    Logic:
    1. Accumulate Sum of Squared Errors (SSE) for each column (position x target) separately.
    2. Accumulate total sample count.
    3. Compute RMSE per column: sqrt(SSE / N).
    4. Return mean of all column RMSEs.
    """

    def __init__(self, device="cpu", seq_scored=68):
        self.device = device
        self.seq_scored = seq_scored

        # Identify indices of the targets that are actually scored
        # ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Indices should be [0, 1, 3]
        self.scored_indices = [
            i for i, t in enumerate(ALL_TARGETS) if t in SCORED_TARGETS
        ]

        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self.total_sse = None  # Will be initialized to tensor on first update
        self.n_samples = 0

    def update(self, preds, targets):
        """
        Updates the metric with a batch of predictions and targets.

        Args:
            preds: (Batch, SeqLen, NumTargets)
            targets: (Batch, SeqLen, NumTargets)
        """
        # Ensure inputs are tensors on the correct device
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, device=self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device)

        # 1. Slice to the scored sequence length (usually 68)
        # 2. Select only the scored target channels
        p = preds[:, : self.seq_scored, self.scored_indices]
        t = targets[:, : self.seq_scored, self.scored_indices]

        # Calculate Squared Error: (y - y_hat)^2
        # Shape: (Batch, SeqScored, NumScoredTargets)
        squared_errors = (p - t) ** 2

        # Sum errors over the batch dimension
        # Shape: (SeqScored, NumScoredTargets)
        batch_sse = torch.sum(squared_errors, dim=0)

        # Accumulate
        if self.total_sse is None:
            self.total_sse = batch_sse
        else:
            self.total_sse += batch_sse

        self.n_samples += preds.shape[0]

    def compute(self):
        """
        Computes the final MCRMSE metric.
        """
        if self.n_samples == 0:
            return 0.0

        # 1. Mean Squared Error per column
        # Shape: (SeqScored, NumScoredTargets)
        mse_per_column = self.total_sse / self.n_samples

        # 2. Root Mean Squared Error per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 3. Mean over all columns (MCRMSE)
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse.item()
