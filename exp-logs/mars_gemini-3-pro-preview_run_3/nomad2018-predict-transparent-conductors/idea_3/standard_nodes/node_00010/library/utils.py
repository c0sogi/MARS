import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def compute_column_wise_rmsle(y_true, y_pred, target_cols=None):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error.

    Args:
        y_true (np.ndarray or pd.DataFrame): Ground truth values.
        y_pred (np.ndarray or pd.DataFrame): Predicted values.
        target_cols (list, optional): List of column names for reporting.

    Returns:
        float: The average RMSLE across all columns.
        dict: Dictionary containing RMSLE for each column.
    """
    # Convert to numpy arrays if necessary
    if isinstance(y_true, pd.DataFrame):
        if target_cols is None:
            target_cols = y_true.columns.tolist()
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Ensure inputs are the same shape
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Clip predictions to be non-negative to avoid log errors
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate RMSLE per column
    # RMSLE = sqrt( mean( (log1p(true) - log1p(pred))^2 ) )
    log_diff = np.log1p(y_true) - np.log1p(y_pred)
    squared_log_diff = np.square(log_diff)
    mean_squared_log_error = np.mean(squared_log_diff, axis=0)
    rmsle_per_col = np.sqrt(mean_squared_log_error)

    # Average across columns
    mean_rmsle = np.mean(rmsle_per_col)

    # Create result dictionary
    results = {"mean_rmsle": mean_rmsle}
    if target_cols:
        for i, col in enumerate(target_cols):
            results[f"rmsle_{col}"] = rmsle_per_col[i]

    return mean_rmsle, results


def load_metadata(split="train"):
    """
    Loads the metadata CSV file for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    df = pd.read_csv(path)
    # print(f"Loaded {split} metadata: {df.shape}")
    return df


def save_submission(ids, predictions, target_cols, output_path):
    """
    Saves the submission file in the required format.

    Args:
        ids (list or np.array): List of IDs corresponding to the predictions.
        predictions (np.array): Predicted values (shape: [n_samples, n_targets]).
        target_cols (list): List of target column names.
        output_path (str): Path to save the CSV file.
    """
    # Ensure predictions match target columns
    if predictions.shape[1] != len(target_cols):
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match number of target columns {len(target_cols)}"
        )

    submission_df = pd.DataFrame(predictions, columns=target_cols)
    submission_df.insert(0, Config.ID_COL, ids)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def load_or_compute(cache_path, compute_func, load_cached_data=True, **kwargs):
    """
    Generic caching utility. Loads data from cache_path if it exists and load_cached_data is True.
    Otherwise, runs compute_func(**kwargs), saves the result to cache_path, and returns it.

    Args:
        cache_path (str): Path to the cache file (supports .parquet and .npy).
        compute_func (callable): Function to compute the data if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to compute_func.

    Returns:
        The loaded or computed data (pd.DataFrame or np.ndarray).
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path, allow_pickle=True)
            else:
                raise ValueError("Unsupported cache file format. Use .parquet or .npy")
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    print(f"Computing data...")
    data = compute_func(**kwargs)

    # 3. Save
    print(f"Saving data to {cache_path}...")
    if isinstance(data, pd.DataFrame):
        data.to_parquet(cache_path, index=False)
    elif isinstance(data, np.ndarray):
        np.save(cache_path, data)
    else:
        print(
            f"Warning: Data type {type(data)} not supported for automatic saving in load_or_compute."
        )

    return data
