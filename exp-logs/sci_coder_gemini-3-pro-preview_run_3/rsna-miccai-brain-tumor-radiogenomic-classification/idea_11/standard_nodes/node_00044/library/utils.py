import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Returns the appropriate PyTorch device (GPU if available, else CPU).

    Returns:
        torch.device: The device to perform computations on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_metadata(split: str) -> pd.DataFrame:
    """
    Loads the metadata for a specific split (train, val, or test) from the ./metadata directory.

    Args:
        split (str): The dataset split to load. Must be one of 'train', 'val', or 'test'.

    Returns:
        pd.DataFrame: The loaded metadata DataFrame containing patient IDs, labels (if applicable),
                      and file paths.

    Raises:
        ValueError: If the split is not valid.
        FileNotFoundError: If the metadata file does not exist.
    """
    valid_splits = ["train", "val", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

    path = os.path.join("./metadata", f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    return pd.read_parquet(path)
