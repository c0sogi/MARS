import numpy as np
import random
import torch
from library.config import SEED, set_seed


def rmse_score(y_true, y_pred):
    """
    Computes the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray or list): Ground truth values.
        y_pred (np.ndarray or list): Predicted values.

    Returns:
        float: The computed RMSE score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # Return Root Mean Squared Error
    return np.sqrt(mse)


def worker_init_fn(worker_id):
    """
    Worker initialization function for PyTorch DataLoaders to ensure deterministic behavior.
    Sets the random seed for NumPy and Python's random module based on the global SEED
    and the worker's ID.

    Args:
        worker_id (int): The ID of the worker process.
    """
    # Create a unique seed for this worker based on the global seed
    worker_seed = SEED + worker_id

    # Set seeds for numpy and random
    np.random.seed(worker_seed)
    random.seed(worker_seed)
