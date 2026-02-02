import os
import random
import re
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_digit_token(text: str) -> bool:
    """
    Checks if the token contains any digit characters.
    Used for the routing logic to identify tokens requiring neural normalization.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r"\d", text))


def ensure_dir(file_path: str):
    """
    Ensures the directory for a given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def safe_save_numpy(data: np.ndarray, path: str):
    """
    Safely saves a numpy array to a file, ensuring the directory exists.
    """
    ensure_dir(path)
    np.save(path, data)


def safe_load_numpy(path: str) -> np.ndarray:
    """
    Safely loads a numpy array from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Numpy file not found at {path}")
    return np.load(path)


def safe_save_dataframe(df: pd.DataFrame, path: str):
    """
    Safely saves a pandas DataFrame to a parquet file, ensuring the directory exists.
    """
    ensure_dir(path)
    df.to_parquet(path, index=False)


def safe_load_dataframe(path: str) -> pd.DataFrame:
    """
    Safely loads a pandas DataFrame from a parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet file not found at {path}")
    return pd.read_parquet(path)


def safe_save_model(model_state: dict, path: str):
    """
    Safely saves a PyTorch model state dictionary, ensuring the directory exists.
    """
    ensure_dir(path)
    torch.save(model_state, path)


def safe_load_model(path: str, device: torch.device = torch.device("cpu")) -> dict:
    """
    Safely loads a PyTorch model state dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return torch.load(path, map_location=device)


def load_metadata(split: str, config: Config) -> pd.DataFrame:
    """
    Loads the raw metadata CSV for a specific split (train, val, test).
    Handles dtype specification to ensure text is read as strings and handles missing values.
    """
    if split == "train":
        path = config.train_data_path
    elif split == "val":
        path = config.val_data_path
    elif split == "test":
        path = config.test_data_path
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    # Specify dtypes to prevent pandas from inferring numeric strings as numbers
    dtypes = {
        "sentence_id": int,
        "token_id": int,
        "before": object,
        "after": object,
        "class": object,
    }

    # Test set doesn't have 'after' or 'class' columns
    if split == "test":
        dtypes.pop("after", None)
        dtypes.pop("class", None)

    df = pd.read_csv(path, dtype=dtypes)

    # Fill NaNs with empty strings for text columns and UNKNOWN for class
    if "before" in df.columns:
        df["before"] = df["before"].fillna("")
    if "after" in df.columns:
        df["after"] = df["after"].fillna("")
    if "class" in df.columns:
        df["class"] = df["class"].fillna("UNKNOWN")

    return df


def calculate_accuracy(predictions: list, targets: list) -> float:
    """
    Calculates exact match accuracy between predictions and targets.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    if not targets:
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)


def print_metrics(metrics: dict, prefix: str = ""):
    """
    Prints metrics with full precision as required.
    """
    print_str = f"{prefix} " if prefix else ""
    for k, v in metrics.items():
        if isinstance(v, float):
            # Print full precision for floats
            print_str += f"{k}: {v:.20f}  "
        else:
            print_str += f"{k}: {v}  "
    print(print_str.strip())
