import numpy as np
import torch
from library.config import set_seed as lib_set_seed


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Wraps the implementation provided in library.config.

    Args:
        seed (int): The seed value to use.
    """
    lib_set_seed(seed)


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) between predicted and actual pressures
    during the inspiratory phase of each breath. The expiratory phase is not scored.

    The inspiratory phase is defined as time steps where u_out == 0.

    Args:
        preds (np.array or torch.Tensor): Predicted pressure values.
        targets (np.array or torch.Tensor): Actual pressure values.
        u_out (np.array or torch.Tensor): The control input for the expiratory valve.
                                          0 represents closed (inspiratory),
                                          1 represents open (expiratory).

    Returns:
        float: The Mean Absolute Error for the inspiratory phase.
    """
    # Convert PyTorch tensors to NumPy arrays if inputs are tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(u_out, torch.Tensor):
        u_out = u_out.detach().cpu().numpy()

    # Flatten arrays to ensure consistent 1D shapes for element-wise operations
    preds = preds.flatten()
    targets = targets.flatten()
    u_out = u_out.flatten()

    # Create a mask for the inspiratory phase (u_out == 0)
    inspiratory_mask = u_out == 0

    # Check if there are any inspiratory steps to avoid division by zero
    if np.sum(inspiratory_mask) == 0:
        return 0.0

    # Calculate absolute error only for the inspiratory phase
    errors = np.abs(preds[inspiratory_mask] - targets[inspiratory_mask])

    # Return the mean of these errors
    return np.mean(errors)
