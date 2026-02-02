import os
import random
import ast
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list_column(x):
    """
    Parses a stringified list into a numpy array.

    Args:
        x (str): A string representation of a list (e.g., "[0.1, 0.2]").

    Returns:
        np.ndarray: A numpy array of type float32. Returns empty array on failure.
    """
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.
    Calculates the MCRMSE specifically for the scored targets defined in Config.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.all_targets = Config.ALL_TARGETS
        self.scored_targets = Config.SCORED_TARGETS

        # Determine indices of scored targets within the full target list.
        # This allows the model to output all 5 targets (helping representation learning),
        # while the loss is strictly computed on the 3 scored targets.
        self.scored_indices = [self.all_targets.index(t) for t in self.scored_targets]

    def forward(self, inputs, targets):
        """
        Forward pass for MCRMSE Loss.

        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Select only the columns that are scored
        # We assume the last dimension corresponds to the targets in Config.ALL_TARGETS
        inputs_scored = inputs[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # Calculate Squared Error
        squared_error = (inputs_scored - targets_scored) ** 2

        # Calculate Mean Squared Error per column (averaging over batch and sequence dims)
        # dim=(0, 1) corresponds to Batch and Seq_Len dimensions
        mse_per_column = torch.mean(squared_error, dim=(0, 1))

        # Calculate Root Mean Squared Error per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate Mean of RMSEs across the scored columns
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
