import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    This loss function:
    1. Slices predictions to the scored sequence length (Config.PRED_LEN).
    2. Selects only the scored columns (Config.SCORED_COLS) for calculation.
    3. Computes the RMSE for each column and then averages them.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # Determine indices of scored columns within the target list
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        self.pred_len = Config.PRED_LEN

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions tensor of shape (Batch, SeqLen_Model, Num_Targets)
            targets: Ground Truth tensor of shape (Batch, SeqLen_Target, Num_Targets)
        """
        # Slice predictions to match the scored sequence length (first 68 positions)
        inputs_sliced = inputs[:, : self.pred_len, :]
        targets_sliced = targets[:, : self.pred_len, :]

        # Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        inputs_scored = inputs_sliced[:, :, self.scored_indices]
        targets_scored = targets_sliced[:, :, self.scored_indices]

        # Compute MSE per column: mean over batch and sequence dimensions
        mse = torch.mean((inputs_scored - targets_scored) ** 2, dim=(0, 1))

        # Compute RMSE per column
        rmse = torch.sqrt(mse)

        # Compute Mean of RMSEs (MCRMSE)
        loss = torch.mean(rmse)

        return loss


class MetricTracker:
    """
    Accumulates Sum of Squared Errors (SSE) and counts to compute global MCRMSE
    correctly over the entire validation set.

    This avoids the statistical bias introduced by averaging RMSEs computed
    on individual small batches.
    """

    def __init__(self):
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]
        self.pred_len = Config.PRED_LEN
        self.reset()

    def reset(self):
        # Initialize SSE per column
        self.sse = np.zeros(len(self.scored_indices), dtype=np.float64)
        self.total_elements = 0

    def update(self, inputs, targets):
        """
        Updates the tracker with a batch of predictions and targets.

        Args:
            inputs: Predictions (Batch, SeqLen, Num_Targets) as Tensor or ndarray
            targets: Ground Truth (Batch, SeqLen, Num_Targets) as Tensor or ndarray
        """
        # Convert to numpy if tensors
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Slice sequence length to the scored region
        inputs_sliced = inputs[:, : self.pred_len, :]
        targets_sliced = targets[:, : self.pred_len, :]

        # Select scored columns
        inputs_scored = inputs_sliced[:, :, self.scored_indices]
        targets_scored = targets_sliced[:, :, self.scored_indices]

        # Calculate squared errors
        squared_errors = (inputs_scored - targets_scored) ** 2

        # Sum errors per column (sum over batch and sequence dimensions)
        batch_sse = np.sum(squared_errors, axis=(0, 1))

        # Update state
        self.sse += batch_sse
        # Count total elements per column (Batch * Scored_Seq_Len)
        self.total_elements += inputs_scored.shape[0] * inputs_scored.shape[1]

    def compute(self):
        """Calculates the final global MCRMSE."""
        if self.total_elements == 0:
            return 0.0

        # MSE per column (SSE / N)
        mse = self.sse / self.total_elements

        # RMSE per column
        rmse = np.sqrt(mse)

        # Mean Columnwise RMSE
        mcrmse = np.mean(rmse)

        return mcrmse


def compute_global_mcrmse(preds, targets):
    """
    Convenience function to compute MCRMSE on full datasets/tensors at once.

    Args:
        preds: Full predictions tensor/array
        targets: Full targets tensor/array

    Returns:
        float: The MCRMSE score
    """
    tracker = MetricTracker()
    tracker.update(preds, targets)
    return tracker.compute()
