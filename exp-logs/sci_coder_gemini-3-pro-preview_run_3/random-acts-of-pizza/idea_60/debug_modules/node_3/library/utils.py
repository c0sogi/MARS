import os
import random
import numpy as np
import pandas as pd
import joblib
import scipy.sparse
import torch


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """
    # Python random
    random.seed(seed)

    # Environment variable for hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy
    np.random.seed(seed)

    # Torch
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in torch backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        # If torch is not installed or fails, we proceed without it
        print(f"Warning: Could not set torch seed: {e}")


def save_artifact(obj, path: str):
    """
    Saves an artifact to the specified path, handling directory creation and
    format dispatch based on file extension.

    Supported formats:
        .parquet -> pandas.DataFrame
        .npy     -> numpy.ndarray
        .npz     -> scipy.sparse matrix
        .joblib  -> generic python object (models, vectorizers)

    Args:
        obj: The object to save.
        path (str): The destination file path.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Dispatch based on extension
    if path.endswith(".parquet"):
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(path, index=False)
        else:
            raise ValueError(
                f"Expected pandas DataFrame for .parquet extension, got {type(obj)}"
            )

    elif path.endswith(".npy"):
        if isinstance(obj, np.ndarray):
            np.save(path, obj)
        else:
            raise ValueError(
                f"Expected numpy ndarray for .npy extension, got {type(obj)}"
            )

    elif path.endswith(".npz"):
        if scipy.sparse.issparse(obj):
            scipy.sparse.save_npz(path, obj)
        elif isinstance(obj, dict):  # Support saving dict of arrays as npz
            np.savez(path, **obj)
        else:
            raise ValueError(
                f"Expected scipy sparse matrix or dict for .npz extension, got {type(obj)}"
            )

    elif path.endswith(".joblib"):
        joblib.dump(obj, path)

    else:
        # Default to joblib for unspecified extensions, but warn or prefer explicit .joblib
        joblib.dump(obj, path)


def load_artifact(path: str):
    """
    Loads an artifact from the specified path, dispatching based on file extension.

    Args:
        path (str): The file path to load from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}")

    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    elif path.endswith(".npy"):
        return np.load(path)

    elif path.endswith(".npz"):
        # Try loading as sparse matrix first, if fails, load as npz archive
        try:
            return scipy.sparse.load_npz(path)
        except:
            return np.load(path)

    elif path.endswith(".joblib"):
        return joblib.load(path)

    else:
        # Fallback
        return joblib.load(path)
