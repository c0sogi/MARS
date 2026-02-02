import os
import random
import numpy as np
import torch
from typing import Dict


def seed_everything(seed: int = 42):
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


def get_pair_distance_vector(structure: str) -> np.ndarray:
    """
    Parses a dot-bracket structure string into a 1D signed distance array.

    This function converts the secondary structure string into a numerical vector
    representing the pairing distance.

    Args:
        structure: A string containing '(', ')', and '.' characters representing
                   the RNA secondary structure.

    Returns:
        A numpy array of integers of the same length as the input structure.
        - Unpaired bases ('.') are assigned 0.
        - Paired bases are assigned the signed distance to their partner:
          (partner_index - current_index).
          Example: If index 2 is paired with index 8:
            vector[2] = 8 - 2 = 6   (Downstream partner is 6 bases away)
            vector[8] = 2 - 8 = -6  (Upstream partner is 6 bases back)
    """
    n = len(structure)
    # Initialize with 0 (unpaired)
    matrix = np.zeros(n, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # j is the index of '(', i is the index of ')'
                # j < i
                # Distance for j: i - j (positive)
                # Distance for i: j - i (negative)
                matrix[j] = i - j
                matrix[i] = j - i
            else:
                # Handle unbalanced closing brackets if necessary.
                # For this task, we assume 0 (unpaired) or ignore.
                pass

    return matrix


def calculate_mcrmse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Column-wise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each column (target) independently over all samples and positions.
    2. Taking the mean of these column-wise RMSE values.

    This matches the competition metric:
    MCRMSE = (1/Nt) * Sum_{j=1}^{Nt} sqrt( (1/n) * Sum_{i=1}^{n} (y_ij - y_hat_ij)^2 )

    Args:
        preds: Predicted values tensor. Can be shape (Batch, Seq, Targets) or (N, Targets).
        targets: Ground truth values tensor. Must match shape of preds.

    Returns:
        A scalar Tensor representing the MCRMSE.
    """
    # Ensure inputs are float tensors
    preds = preds.float()
    targets = targets.float()

    # Compute squared errors
    squared_diff = (preds - targets) ** 2

    # If 3D (Batch, Seq, Targets), flatten to (Batch*Seq, Targets)
    # This aggregates all scored positions across all samples for the column-wise calculation
    if squared_diff.dim() == 3:
        squared_diff = squared_diff.view(-1, squared_diff.shape[-1])

    # Compute Mean Squared Error per column (averaging over all samples/positions)
    # Shape: (Num_Targets,)
    mse_per_column = torch.mean(squared_diff, dim=0)

    # Compute RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Compute Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse_per_column)

    return mcrmse


# ==========================================
# Constants / Mappings
# ==========================================

TOKEN_MAP: Dict[str, int] = {"A": 0, "G": 1, "C": 2, "U": 3}

LOOP_TYPE_MAP: Dict[str, int] = {
    "S": 0,  # Stem
    "M": 1,  # Multiloop
    "I": 2,  # Internal loop
    "B": 3,  # Bulge
    "H": 4,  # Hairpin loop
    "E": 5,  # dangling End
    "X": 6,  # eXternal loop
}
