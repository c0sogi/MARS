import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).

    Returns:
        torch.device: The device to use for tensor computations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_metadata(split: str, metadata_dir: str = "./metadata") -> pd.DataFrame:
    """
    Loads the metadata CSV file for a specific data split.

    Args:
        split (str): The split to load ('train', 'val', or 'test').
        metadata_dir (str): The directory containing the metadata CSVs. Defaults to "./metadata".

    Returns:
        pd.DataFrame: The loaded metadata dataframe.

    Raises:
        ValueError: If the split is not one of 'train', 'val', or 'test'.
        FileNotFoundError: If the metadata file does not exist.
    """
    if split not in ["train", "val", "test"]:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of 'train', 'val', 'test'."
        )

    file_path = os.path.join(metadata_dir, f"{split}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Metadata file for split '{split}' not found at {file_path}"
        )

    return pd.read_csv(file_path)


def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Area Under the ROC Curve (AUC) metric.

    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_pred (np.ndarray): Predicted probabilities.

    Returns:
        float: The ROC AUC score. Returns 0.5 if calculation fails (e.g., single class in batch).
    """
    try:
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        # This can happen if y_true only has one class present in the current batch/subset
        return 0.5
