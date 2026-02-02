import os
import pandas as pd
import numpy as np
from library.config import Config, seed_everything


def get_config_hash():
    """
    Returns the unique configuration hash based on the current hyperparameters.
    """
    return Config.get_hash()


def get_artifact_path(name):
    """
    Returns the versioned path for an artifact (e.g., 'model.pt' -> 'model_1a2b3c.pt').
    Ensures the artifact directory exists.
    """
    return Config.get_artifact_path(name)


def load_metadata(split="train"):
    """
    Loads the metadata CSV for the specified split ('train', 'val', or 'test').

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded dataframe with all columns as strings.
    """
    if split == "train":
        path = Config.TRAIN_META_PATH
    elif split == "val":
        path = Config.VAL_META_PATH
    elif split == "test":
        path = Config.TEST_META_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    # Load as object/string to preserve exact token formatting (e.g. keeping "007" as "007")
    df = pd.read_csv(path, dtype=str).fillna("")
    return df


def compute_accuracy(preds, targets):
    """
    Computes the exact match prediction accuracy.

    Args:
        preds (list or np.array): List of predicted strings.
        targets (list or np.array): List of ground truth strings.

    Returns:
        float: The accuracy (0.0 to 1.0).
    """
    if len(preds) != len(targets):
        raise ValueError(
            f"Predictions ({len(preds)}) and targets ({len(targets)}) must have the same length."
        )

    preds_arr = np.array(preds, dtype=str)
    targets_arr = np.array(targets, dtype=str)

    correct = np.sum(preds_arr == targets_arr)
    total = len(targets_arr)

    return correct / total if total > 0 else 0.0


def save_npy(data, name):
    """
    Saves data to a .npy file using the versioned path.

    Args:
        data: The data to save (numpy array, dictionary, etc.).
        name (str): The base filename (e.g., 'ngram_stats.npy').

    Returns:
        str: The path where the file was saved.
    """
    path = get_artifact_path(name)
    # Config.get_artifact_path ensures the directory exists, but we double check parent dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)
    return path


def load_npy(name):
    """
    Loads data from a .npy file if it exists.

    Args:
        name (str): The base filename.

    Returns:
        The loaded data, or None if the file does not exist.
    """
    path = get_artifact_path(name)
    if os.path.exists(path):
        # allow_pickle is required if the npy file contains python objects (like dicts)
        data = np.load(path, allow_pickle=True)
        # If data is a 0-d array wrapping an object (common when saving dicts), extract it
        if data.ndim == 0 and data.dtype == object:
            return data.item()
        return data
    return None
