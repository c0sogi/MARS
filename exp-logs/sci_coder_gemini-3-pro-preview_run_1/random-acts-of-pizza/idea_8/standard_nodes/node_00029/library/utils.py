import os
import json
import pandas as pd
import numpy as np
from library.config import Config, set_seed as _set_seed


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the implementation provided in library.config.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    _set_seed(seed)


def load_data(path, file_type=None, nrows=None):
    """
    Loads data from a file into a pandas DataFrame. Supports CSV, JSON, and Parquet formats.

    Args:
        path (str): The file path to load data from.
        file_type (str, optional): The type of file ('csv', 'json', 'parquet').
                                   If None, it is inferred from the file extension.
        nrows (int, optional): The number of rows to read. Useful for debugging with smaller datasets.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file type is unsupported or cannot be inferred.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if file_type is None:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".csv":
            file_type = "csv"
        elif ext == ".json":
            file_type = "json"
        elif ext == ".parquet":
            file_type = "parquet"
        else:
            raise ValueError(f"Could not infer file type from extension: {ext}")

    if file_type == "csv":
        return pd.read_csv(path, nrows=nrows)
    elif file_type == "json":
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if nrows is not None:
            return df.head(nrows)
        return df
    elif file_type == "parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            return df.head(nrows)
        return df
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def get_common_columns(df1, df2, exclude=None):
    """
    Identifies the intersection of columns between two DataFrames, optionally excluding specific columns.
    This is useful for ensuring feature consistency between training and test sets and preventing leakage.

    Args:
        df1 (pd.DataFrame): First DataFrame (e.g., train).
        df2 (pd.DataFrame): Second DataFrame (e.g., test).
        exclude (list, optional): List of column names to exclude from the result.

    Returns:
        list: A sorted list of column names present in both DataFrames.
    """
    common_cols = set(df1.columns).intersection(set(df2.columns))

    if exclude:
        common_cols = common_cols - set(exclude)

    return sorted(list(common_cols))


def save_submission(
    ids,
    predictions,
    path=None,
    id_col="request_id",
    target_col="requester_received_pizza",
):
    """
    Formats and saves the prediction results to a CSV file.

    Args:
        ids (array-like): The sequence of request IDs.
        predictions (array-like): The sequence of prediction scores or labels.
        path (str): The output file path. Defaults to Config.SUBMISSION_PATH.
        id_col (str): The name of the ID column in the output CSV.
        target_col (str): The name of the target column in the output CSV.
    """
    if path is None:
        path = Config.SUBMISSION_PATH

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Flatten arrays to ensure correct shape
    ids = np.array(ids).flatten()
    predictions = np.array(predictions).flatten()

    submission_df = pd.DataFrame({id_col: ids, target_col: predictions})

    submission_df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
