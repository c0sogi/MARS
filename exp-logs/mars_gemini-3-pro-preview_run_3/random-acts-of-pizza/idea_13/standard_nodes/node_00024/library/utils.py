import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(split: str) -> pd.DataFrame:
    """
    Loads the specific dataset split (train, val, or test) from the metadata directory.

    Args:
        split (str): One of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_parquet(path)


def ensure_cache_dir():
    """Ensures the cache directory exists."""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)


def save_cache_npy(data: np.ndarray, filename: str) -> None:
    """
    Saves a numpy array to the cache directory.

    Args:
        data (np.ndarray): Data to save.
        filename (str): Filename (e.g., 'X_train_lexical.npy').
    """
    ensure_cache_dir()
    file_path = os.path.join(Config.CACHE_DIR, filename)
    np.save(file_path, data)


def load_cache_npy(filename: str) -> np.ndarray:
    """
    Loads a numpy array from the cache directory.

    Args:
        filename (str): Filename to load.

    Returns:
        np.ndarray: The loaded data, or None if not found.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=False)
    return None


def save_cache_npz(data_dict: dict, filename: str) -> None:
    """
    Saves a dictionary of arrays or sparse matrices to a compressed .npz file.
    Useful for sparse matrices (scipy.sparse) if converted or generic dict storage.
    Note: For scipy sparse matrices, use scipy.sparse.save_npz directly in feature code,
    but this helper handles standard numpy archives.
    """
    ensure_cache_dir()
    file_path = os.path.join(Config.CACHE_DIR, filename)
    np.savez_compressed(file_path, **data_dict)


def load_cache_npz(filename: str):
    """
    Loads a .npz file.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=False)
    return None


def save_cache_parquet(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a DataFrame to the cache directory as Parquet.

    Args:
        df (pd.DataFrame): Data to save.
        filename (str): Filename (e.g., 'train_features.parquet').
    """
    ensure_cache_dir()
    file_path = os.path.join(Config.CACHE_DIR, filename)
    df.to_parquet(file_path, index=False)


def load_cache_parquet(filename: str) -> pd.DataFrame:
    """
    Loads a DataFrame from the cache directory.

    Args:
        filename (str): Filename to load.

    Returns:
        pd.DataFrame: The loaded data, or None if not found.
    """
    file_path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None


def save_submission(
    request_ids: np.ndarray,
    probabilities: np.ndarray,
    output_path: str = Config.SUBMISSION_PATH,
) -> None:
    """
    Generates and saves the submission file.

    Args:
        request_ids (np.ndarray): Array of request IDs.
        probabilities (np.ndarray): Array of predicted probabilities (float).
        output_path (str): Path to save the CSV.
    """
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def print_metric(metric_name: str, value: float) -> None:
    """
    Prints a metric with full precision.

    Args:
        metric_name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{metric_name}: {value}")


def compute_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Area Under the ROC Curve.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)
