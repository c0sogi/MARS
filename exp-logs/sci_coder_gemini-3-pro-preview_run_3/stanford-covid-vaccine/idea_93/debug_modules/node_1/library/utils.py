import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Computes the RMSE for each target column separately and then averages them.
    This matches the competition metric.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Length, Channels) or (N, Channels).
            targets (torch.Tensor): Ground truth of shape (Batch, Length, Channels) or (N, Channels).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten the batch and sequence dimensions if present
        # Shape becomes (N_total, C_channels)
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])

        # Compute MSE per column
        mse = torch.mean((inputs - targets) ** 2, dim=0)

        # Compute RMSE per column
        rmse = torch.sqrt(mse)

        # Return mean of column RMSEs
        return torch.mean(rmse)


def get_structure_indices(structure):
    """
    Parses a dot-bracket structure string to find pairing indices.

    Args:
        structure (str): A string representing RNA secondary structure (e.g., "((..))").

    Returns:
        np.ndarray: An array of integers where arr[i] is the index of the base paired with i.
                    If i is unpaired, arr[i] is -1.
    """
    length = len(structure)
    indices = np.full(length, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                indices[i] = j
                indices[j] = i

    return indices
