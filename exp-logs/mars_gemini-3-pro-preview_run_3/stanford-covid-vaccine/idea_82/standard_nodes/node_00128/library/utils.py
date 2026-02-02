import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Formula:
    MCRMSE = (1 / Nt) * sum_j( sqrt( (1 / n) * sum_i( (y_ij - yhat_ij)^2 ) ) )

    Where:
    - i iterates over samples (batch dimension)
    - j iterates over target columns (flattened feature dimension)
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values. Shape (Batch, ...).
            targets (torch.Tensor): Ground truth values. Shape (Batch, ...).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate squared difference
        squared_diff = (inputs - targets) ** 2

        # Average over the batch dimension (dim=0) to get MSE per column
        mse = torch.mean(squared_diff, dim=0)

        # Calculate RMSE per column (adding epsilon for stability)
        rmse = torch.sqrt(mse + 1e-8)

        # Average over all columns (remaining dimensions)
        mcrmse = torch.mean(rmse)

        return mcrmse


def parse_structure_to_adj(structure: str) -> np.ndarray:
    """
    Parses a dot-bracket secondary structure string into an adjacency index map.

    This function is critical for the GLU-Refined Decoupled Interaction Module.
    It maps each position 'i' to its paired position 'j'.

    Args:
        structure (str): Dot-bracket notation string (e.g., '...((...))...').

    Returns:
        np.ndarray: An array of shape (len(structure),) containing indices.
                    - If position i is paired with j, adj[i] = j.
                    - If position i is unpaired, adj[i] = -1.
    """
    length = len(structure)
    adj = np.full(length, -1, dtype=np.int64)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                start = stack.pop()
                # Bidirectional connection
                adj[start] = i
                adj[i] = start
            else:
                # Unbalanced structure handling (though data is expected to be clean)
                pass

    return adj
