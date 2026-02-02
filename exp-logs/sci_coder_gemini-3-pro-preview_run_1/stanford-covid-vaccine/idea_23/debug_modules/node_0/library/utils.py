import os
import random
import copy
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each target column separately (aggregating over samples and sequence positions).
    2. Taking the average of these column-wise RMSEs.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values. Shape (N, ...) or (N, Seq_Len, Channels).
        y_pred (np.ndarray or torch.Tensor): Predicted values. Shape must match y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure shapes match
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Calculate squared errors
    squared_diff = (y_true - y_pred) ** 2

    # Determine axes to average over. We want to average over everything EXCEPT the last dimension (columns/targets).
    # If shape is (Batch, Seq, Channels), we average over (0, 1).
    # If shape is (Batch, Channels), we average over (0).
    reduce_axes = tuple(range(y_true.ndim - 1))

    # Mean Squared Error per column
    mse_per_col = np.mean(squared_diff, axis=reduce_axes)

    # Root Mean Squared Error per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Mean of the column RMSEs
    return np.mean(rmse_per_col)


def average_weights(state_dicts):
    """
    Computes the arithmetic mean of a list of model state dictionaries.
    Used for Stochastic Weight Averaging (SWA).

    Args:
        state_dicts (list): A list of PyTorch state_dict objects (OrderedDict).

    Returns:
        dict: A new state_dict containing the averaged weights.
    """
    if not state_dicts:
        raise ValueError("No state dictionaries provided for averaging.")

    # Initialize the averaged state with the structure of the first model
    avg_state = copy.deepcopy(state_dicts[0])

    num_models = len(state_dicts)

    for key in avg_state.keys():
        # We only average tensors (parameters and buffers)
        if isinstance(avg_state[key], torch.Tensor):
            # Start with the tensor from the first model
            sum_tensor = avg_state[key].clone()

            # Add tensors from the remaining models
            for i in range(1, num_models):
                sum_tensor += state_dicts[i][key]

            # Compute the average
            avg_state[key] = sum_tensor / num_models

    return avg_state
