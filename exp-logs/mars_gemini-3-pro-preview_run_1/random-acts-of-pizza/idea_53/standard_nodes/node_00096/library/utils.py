import os
import json
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.RANDOM_SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN if needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_json(path: str):
    """
    Loads a JSON file from the specified path.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        The data parsed from the JSON file (usually a list or dict).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found at: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_submission(request_ids, probabilities, filename: str = Config.SUBMISSION_PATH):
    """
    Saves the prediction results to a CSV file in the required format.

    Args:
        request_ids (list or np.array): List of request IDs.
        probabilities (list or np.array): List of predicted probabilities.
        filename (str): The output file path. Defaults to Config.SUBMISSION_PATH.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({Config.ID_COL: request_ids, Config.TARGET_COL: probabilities})

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")


def get_common_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols: list = None
):
    """
    Identifies the intersection of columns between the training and test DataFrames
    to prevent feature leakage.

    Args:
        train_df (pd.DataFrame): The training DataFrame.
        test_df (pd.DataFrame): The test DataFrame.
        exclude_cols (list, optional): List of columns to explicitly exclude
                                       (e.g., target column). Defaults to None.

    Returns:
        list: A sorted list of column names present in both DataFrames.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    common_cols = train_cols.intersection(test_cols)

    if exclude_cols:
        common_cols = common_cols - set(exclude_cols)

    return sorted(list(common_cols))


def load_metadata_splits():
    """
    Loads the Train, Validation, and Test splits from the metadata CSVs defined in Config.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )
    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(f"Val metadata not found at {Config.VAL_METADATA_PATH}")
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    return train_df, val_df, test_df
