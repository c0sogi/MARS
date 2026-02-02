import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_log_loss(y_true, y_pred):
    """
    Calculates the multi-class logarithmic loss with specific rescaling and clipping.

    Steps:
    1. Rescale each row of probabilities so they sum to 1.
    2. Clip probabilities to the range [1e-15, 1 - 1e-15].
    3. Calculate log loss.

    Args:
        y_true: Ground truth labels (n_samples,). Can be class indices or string labels.
        y_pred: Predicted probabilities (n_samples, n_classes).

    Returns:
        float: The calculated log loss.
    """
    # Ensure predictions are a numpy array of floats
    y_pred = np.array(y_pred, dtype=np.float64)

    # 1. Rescale: each row is divided by the row sum
    row_sums = y_pred.sum(axis=1)
    # Handle potential zero sums (though unlikely) to avoid division by zero
    row_sums[row_sums == 0] = 1.0
    y_pred = y_pred / row_sums[:, np.newaxis]

    # 2. Clip: max(min(p, 1-10^-15), 10^-15)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # 3. Calculate Log Loss
    # sklearn.metrics.log_loss handles the cross-entropy calculation
    return log_loss(y_true, y_pred)


def save_artifact(data, filepath):
    """
    Saves data to a file using .npy (for arrays) or .parquet (for DataFrames).
    Automatically creates the parent directory if it does not exist.

    Args:
        data: The data to save (numpy.ndarray or pandas.DataFrame).
        filepath: The full path where the file should be saved.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if filepath.endswith(".npy"):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        np.save(filepath, data)
    elif filepath.endswith(".parquet"):
        if not isinstance(data, pd.DataFrame):
            # Attempt to convert to DataFrame if it's not one
            data = pd.DataFrame(data)
        data.to_parquet(filepath, index=False)
    else:
        raise ValueError(
            f"Unsupported file extension for {filepath}. Use .npy or .parquet"
        )


def load_artifact(filepath):
    """
    Loads data from a .npy or .parquet file.

    Args:
        filepath: The full path to the file.

    Returns:
        The loaded data (numpy.ndarray or pandas.DataFrame), or None if the file does not exist.
    """
    if not os.path.exists(filepath):
        return None

    if filepath.endswith(".npy"):
        # Load numpy array
        # allow_pickle=True is used to support object arrays if necessary,
        # though numerical arrays are preferred.
        try:
            return np.load(filepath)
        except ValueError:
            return np.load(filepath, allow_pickle=True)
    elif filepath.endswith(".parquet"):
        # Load pandas DataFrame
        return pd.read_parquet(filepath)
    else:
        raise ValueError(
            f"Unsupported file extension for {filepath}. Use .npy or .parquet"
        )
