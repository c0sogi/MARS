import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_pair_map(structure):
    """
    Parses a dot-bracket secondary structure string to generate a mapping of paired base indices.

    This function is essential for the 'Spatial Augmentation' strategy, enabling the model
    to explicitly reference paired features.

    Args:
        structure (str): A string representing RNA secondary structure (e.g., ".(..).").

    Returns:
        np.ndarray: An integer array of shape (len(structure),). If index i is paired with j,
                    arr[i] = j. If index i is unpaired, arr[i] = -1.
    """
    length = len(structure)
    pair_map = np.full(length, -1, dtype=int)
    stack = []

    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_map[i] = j
                pair_map[j] = i
            else:
                # In case of malformed structure, we ignore the closing bracket
                pass

    return pair_map


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Calculates the RMSE for each target column separately and returns the mean of these RMSEs.
    This matches the competition metric and the strategy to optimize unweighted MCRMSE on all targets.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted values. Shape (Batch, Seq, Targets) or (N, Targets).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq, Targets) or (N, Targets).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Flatten batch and sequence dimensions to (Total_Samples, Num_Targets)
        # This ensures we calculate metrics over the entire batch/sequence set per column
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

        # Compute Squared Error per element
        squared_error = self.mse(inputs, targets)

        # Compute Mean Squared Error per column (averaging over samples)
        mse_per_column = torch.mean(squared_error, dim=0)

        # Compute RMSE per column (adding epsilon for numerical stability)
        rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

        # Average the RMSEs across all columns
        loss = torch.mean(rmse_per_column)

        return loss


def save_checkpoint(state, filename):
    """
    Saves the model training state to a file.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        filename (str): Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    """
    Loads a model checkpoint from a file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(filename):
        print(f"Error: Checkpoint file {filename} not found.")
        return None

    checkpoint = torch.load(filename, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback if the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state if provided and available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint
