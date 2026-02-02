import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior for cuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pearson(predictions, targets):
    """
    Computes the Pearson correlation coefficient between predictions and targets.

    Args:
        predictions: Predicted scores (list, numpy array, or torch.Tensor).
        targets: Ground truth scores (list, numpy array, or torch.Tensor).

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are flattened 1D arrays
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # Basic validation to prevent errors with empty or too small inputs
    if len(predictions) < 2 or len(targets) < 2:
        return 0.0

    # Check for zero variance to avoid RuntimeWarnings and NaNs during calculation
    # If either vector is constant, correlation is undefined (conceptually 0 for this task)
    if np.std(predictions) == 0 or np.std(targets) == 0:
        return 0.0

    # Compute Pearson correlation
    # np.corrcoef returns the correlation matrix: [[1.0, r], [r, 1.0]]
    correlation = np.corrcoef(predictions, targets)[0, 1]

    return correlation
