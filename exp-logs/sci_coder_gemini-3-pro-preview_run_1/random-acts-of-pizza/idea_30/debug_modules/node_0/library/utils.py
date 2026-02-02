import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed: int = Config.RANDOM_STATE):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set. Defaults to Config.RANDOM_STATE.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(file_path: str):
    """
    Ensures that the directory for a given file path exists.

    Args:
        file_path (str): The path to the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_feature_intersection(
    train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols: list = None
) -> list:
    """
    Returns the list of columns present in both train and test dataframes,
    excluding specific columns provided in the exclude_cols list.

    This ensures strict alignment of columns between train and test sets to prevent leakage
    and shape mismatch errors during inference.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        test_df (pd.DataFrame): Test dataframe.
        exclude_cols (list, optional): List of column names to exclude (e.g., target, ID).

    Returns:
        list: Sorted list of common column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Find intersection
    common_cols = train_cols.intersection(test_cols)

    # Remove excluded columns
    final_cols = [c for c in common_cols if c not in exclude_cols]

    return sorted(final_cols)


def save_artifact(obj, path: str):
    """
    Saves a Python object to a file using pickle.

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_artifact(path: str):
    """
    Loads a Python object from a pickle file.

    Args:
        path (str): The source file path.

    Returns:
        The loaded object, or None if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model_checkpoint(model, path: str):
    """
    Saves a PyTorch model state dictionary.

    Args:
        model: The PyTorch model.
        path (str): The destination file path.
    """
    ensure_dir(path)
    torch.save(model.state_dict(), path)


def load_model_checkpoint(model, path: str, device=Config.DEVICE):
    """
    Loads a PyTorch model state dictionary into a model instance.

    Args:
        model: The PyTorch model instance to load weights into.
        path (str): The source file path.
        device (str): The device to map the weights to.

    Returns:
        model: The model with loaded weights.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
