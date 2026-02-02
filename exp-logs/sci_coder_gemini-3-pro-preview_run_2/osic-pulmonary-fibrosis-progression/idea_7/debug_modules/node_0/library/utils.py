import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Calculates the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true: True FVC values (numpy array or torch tensor).
        y_pred: Predicted FVC values (numpy array or torch tensor).
        sigma: Predicted confidence (standard deviation) (numpy array or torch tensor).

    Returns:
        The average metric score (scalar).
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Flatten arrays to ensure element-wise operations work correctly
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    sigma = np.array(sigma).flatten()

    # 1. Clip sigma (confidence) at 70 ml
    sigma_clipped = np.maximum(sigma, 70)

    # 2. Calculate delta (absolute error), clipped at 1000 ml
    delta = np.minimum(np.abs(y_true - y_pred), 1000)

    # 3. Compute metric
    sqrt_2 = np.sqrt(2)
    metric = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)

    return np.mean(metric)


def save_numpy(path, data):
    """
    Saves a numpy array to a file, ensuring the directory exists.

    Args:
        path: File path to save the data (e.g., 'data.npy').
        data: Numpy array to save.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, data)


def load_numpy(path):
    """
    Loads a numpy array from a file.

    Args:
        path: File path to load the data from.

    Returns:
        The loaded numpy array, or None if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    try:
        # allow_pickle=False ensures we strictly use npy format as requested
        return np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
