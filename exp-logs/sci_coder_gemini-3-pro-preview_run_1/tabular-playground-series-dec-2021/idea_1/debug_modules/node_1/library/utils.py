import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data(
    split="train",
    load_cached_data=True,
    data_dir="./metadata",
    cache_dir="./working/idea_1",
    sample_size=None,
):
    """
    Loads the dataset for a specific split (train, val, test).
    Implements caching using Parquet to speed up subsequent loads.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        data_dir (str): Directory containing the metadata CSV files.
        cache_dir (str): Directory to store/retrieve cached Parquet files.
        sample_size (int, optional): If provided, returns a random sample of this size for debugging.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{split}.parquet")
    csv_path = os.path.join(data_dir, f"{split}.csv")

    df = None

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            # If cache load fails, fall back to CSV
            df = None

    # 2. If not loaded from cache, load from CSV and save to cache
    if df is None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Save to cache for future use
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass  # Non-critical failure

    # 3. Subsample if requested (for debugging)
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


def save_submission(ids, predictions, output_path="./submission/submission.csv"):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        ids (array-like): List or array of sample IDs.
        predictions (array-like): List or array of predicted class labels.
        output_path (str): Path to save the submission CSV.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df = pd.DataFrame({"Id": ids, "Cover_Type": predictions})

    submission_df.to_csv(output_path, index=False)


def print_metrics(metrics):
    """
    Prints metric values with full precision.

    Args:
        metrics (dict): Dictionary containing metric names and values.
    """
    for key, value in metrics.items():
        print(f"{key}: {value}")
