import torch
import torch.nn as nn
import numpy as np
import random
import os
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    This loss function calculates the RMSE for each of the specific scored columns
    and then averages these RMSE values.

    It handles:
    1. Column Masking: Only scores 'reactivity', 'deg_Mg_pH10', and 'deg_Mg_50C'.
    2. Sequence Slicing: Truncates predictions to match the length of the ground truth
       (e.g., predicting 107 positions but scoring only the first 68).
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # Identify indices of columns to score based on Config
        # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Indices: 0, 1, 3
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions. Shape (Batch, Seq_Len_Pred, Num_Targets)
            targets: Ground truth. Shape (Batch, Seq_Len_Target, Num_Targets)
        """
        # 1. Sequence Length Slicing
        # If predictions are longer than targets (e.g. 107 vs 68), slice predictions.
        if inputs.shape[1] > targets.shape[1]:
            inputs = inputs[:, : targets.shape[1], :]

        # 2. Column Selection
        # Select only the columns that contribute to the score
        inputs_scored = inputs[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # 3. Compute MSE per column
        # Shape: (Batch, Seq, Scored_Cols) -> Mean over Batch and Seq -> (Scored_Cols)
        mse_per_col = torch.mean((inputs_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Average RMSE across columns to get MCRMSE
        loss = torch.mean(rmse_per_col)

        return loss


class GlobalMetricTracker:
    """
    Accumulates metrics over the entire validation loop to compute
    unbiased global scores, avoiding batch-averaging bias.
    """

    def __init__(self):
        self.reset()
        # Indices for scoring: 0, 1, 3 corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

    def reset(self):
        """Resets the internal state."""
        self.sse_per_col = 0.0  # Sum of Squared Errors per scored column
        self.count = 0  # Total number of elements (Batch * Seq_Len)

    def update(self, preds, targets):
        """
        Updates the running statistics with a new batch of data.

        Args:
            preds: Model predictions (Batch, Seq_Len_Pred, Num_Targets)
            targets: Ground truth (Batch, Seq_Len_Target, Num_Targets)
        """
        # Ensure tensor operations are on CPU for accumulation to save GPU memory
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Slice predictions to match target sequence length
        if preds.shape[1] > targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        # Select scored columns
        preds_scored = preds[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # Calculate squared errors
        # Sum over batch and sequence dimensions, keep column dimension
        squared_errors = torch.sum((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # Update state
        self.sse_per_col += squared_errors
        # Count is Batch_Size * Sequence_Length
        self.count += targets.shape[0] * targets.shape[1]

    def get_score(self):
        """
        Computes the final MCRMSE score based on accumulated data.
        """
        if self.count == 0:
            return 0.0

        # Mean Squared Error per column
        mse_per_col = self.sse_per_col / self.count

        # Root Mean Squared Error per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Mean Columnwise RMSE
        mcrmse = torch.mean(rmse_per_col).item()

        return mcrmse
