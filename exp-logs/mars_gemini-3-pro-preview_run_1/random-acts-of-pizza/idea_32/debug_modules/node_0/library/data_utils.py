import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(debug=False):
    """
    Loads the train, validation, and test datasets from the metadata CSVs.

    Args:
        debug (bool): If True, loads only a small subset of the data for debugging.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Ensure reproducibility when loading/sampling
    seed_everything(Config.RANDOM_SEED)

    print(f"Loading datasets from {Config.METADATA_DIR}...")

    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {Config.TRAIN_DATA_PATH}")
    if not os.path.exists(Config.VAL_DATA_PATH):
        raise FileNotFoundError(f"Validation data not found at {Config.VAL_DATA_PATH}")
    if not os.path.exists(Config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {Config.TEST_DATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    if debug:
        print(f"Debug mode enabled. Sampling {Config.DEBUG_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SIZE).copy()
        val_df = val_df.head(Config.DEBUG_SIZE).copy()
        test_df = test_df.head(Config.DEBUG_SIZE).copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    return train_df, val_df, test_df


def enforce_column_intersection(
    train_df, val_df, test_df, target_col="requester_received_pizza"
):
    """
    Restricts the dataframes to only include columns present in all sets (train, val, test),
    preserving the target column in train and val.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        target_col (str): Name of the target column.

    Returns:
        tuple: (train_df, val_df, test_df) with aligned columns.
    """
    print("Enforcing column intersection...")

    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    test_cols = set(test_df.columns)

    # Find common features across all datasets
    common_cols = train_cols.intersection(val_cols).intersection(test_cols)

    # Ensure target is not treated as a feature if it happens to be in test (unlikely but safe)
    if target_col in common_cols:
        common_cols.remove(target_col)

    common_cols_list = sorted(list(common_cols))

    # Columns to keep for train/val: common features + target
    train_keep_cols = common_cols_list + [target_col]

    # Columns to keep for test: common features only
    test_keep_cols = common_cols_list

    # Filter dataframes
    train_df_filtered = train_df[train_keep_cols].copy()
    val_df_filtered = val_df[train_keep_cols].copy()
    test_df_filtered = test_df[test_keep_cols].copy()

    print(f"Common features found: {len(common_cols_list)}")
    print(f"Train shape after intersection: {train_df_filtered.shape}")
    print(f"Val shape after intersection:   {val_df_filtered.shape}")
    print(f"Test shape after intersection:  {test_df_filtered.shape}")

    return train_df_filtered, val_df_filtered, test_df_filtered


def save_submission(predictions, test_df, filename=None):
    """
    Saves the predictions to a CSV file in the required format.

    Args:
        predictions (array-like): Predicted probabilities.
        test_df (pd.DataFrame): Test dataframe containing 'request_id'.
        filename (str, optional): Output filename. Defaults to Config.SUBMISSION_PATH.
    """
    if filename is None:
        filename = Config.SUBMISSION_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create submission DataFrame
    # Ensure we use the request_id from the test dataframe
    if "request_id" not in test_df.columns:
        raise KeyError("test_df must contain 'request_id' column for submission.")

    submission = pd.DataFrame(
        {"request_id": test_df["request_id"], "requester_received_pizza": predictions}
    )

    # Save to CSV
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    print("First 5 rows of submission:")
    print(submission.head())
