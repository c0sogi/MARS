import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and OS environments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def ensure_directory(path):
    """
    Ensures that the directory for the given path exists.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def clip_probabilities(preds):
    """
    Clips probabilities to the range [1e-15, 1 - 1e-15] to avoid log(0) errors.
    Matches the competition metric definition.

    Args:
        preds (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    eps = 1e-15
    return np.clip(preds, eps, 1 - eps)


def normalize_probabilities(preds):
    """
    Rescales probabilities so that each row sums to 1.
    This mimics the competition's scoring mechanism where submitted probabilities
    are rescaled prior to being scored.

    Args:
        preds (np.ndarray): Array of predicted probabilities.

    Returns:
        np.ndarray: Normalized probabilities.
    """
    # Ensure float64 for precision as per Config
    preds = preds.astype(Config.FLOAT_TYPE)
    row_sums = preds.sum(axis=1, keepdims=True)

    # Avoid division by zero (handle edge case of all-zero row)
    row_sums[row_sums == 0] = 1.0

    return preds / row_sums


def log_loss_score(y_true, y_pred):
    """
    Calculates the Multi-class Log Loss with normalization and clipping.

    Args:
        y_true (array-like): True labels (integer encoded).
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The log loss score.
    """
    # Normalize rows
    y_pred_norm = normalize_probabilities(y_pred)

    # Clip probabilities
    y_pred_clipped = clip_probabilities(y_pred_norm)

    # Calculate log loss
    # We assume y_true are integer class indices corresponding to columns of y_pred.
    # We pass labels explicitly to handle cases where a batch might miss a class.
    labels = list(range(y_pred.shape[1]))
    return log_loss(y_true, y_pred_clipped, labels=labels)


def save_submission(ids, class_names, probs, filename=Config.SUBMISSION_PATH):
    """
    Saves the submission file in the required format.

    Args:
        ids (array-like): Image IDs.
        class_names (list): List of species names (column headers).
        probs (np.ndarray): Predicted probabilities matrix.
        filename (str): Path to save the CSV.
    """
    ensure_directory(filename)

    # Create DataFrame
    df = pd.DataFrame(probs, columns=class_names)
    df.insert(0, "id", ids)

    # Save
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def load_metadata(path):
    """
    Safely loads metadata CSV files.

    Args:
        path (str): Path to the metadata CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_csv(path)


def save_cache(data, path):
    """
    Saves data to cache using numpy (npy) or pandas (parquet).
    Adheres to the requirement of not using pickle.

    Args:
        data: The data to save (np.ndarray or pd.DataFrame).
        path (str): The file path.
    """
    ensure_directory(path)
    if isinstance(data, pd.DataFrame):
        data.to_parquet(path)
    elif isinstance(data, np.ndarray):
        np.save(path, data)
    else:
        raise ValueError("Unsupported data type for caching. Use DataFrame or ndarray.")


def load_cache(path):
    """
    Loads data from cache if it exists.

    Args:
        path (str): The file path.

    Returns:
        The loaded data or None if file does not exist.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".npy"):
        return np.load(path)
    return None
