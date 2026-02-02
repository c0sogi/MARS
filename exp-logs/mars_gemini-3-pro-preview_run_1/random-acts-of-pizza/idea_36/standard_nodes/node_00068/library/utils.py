import os
import random
import numpy as np
import torch
import pandas as pd


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

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    """
    Ensures that the directory for the given file path exists.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def save_file(data, path, **kwargs):
    """
    Saves data to a file based on the extension.
    Supports .parquet (pandas) and .npy/.npz (numpy).
    """
    ensure_dir(path)

    if path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(path, index=False, **kwargs)
        else:
            raise ValueError(
                f"Data must be a pandas DataFrame to save as parquet: {path}"
            )

    elif path.endswith(".npy"):
        np.save(path, data)

    elif path.endswith(".npz"):
        if isinstance(data, dict):
            np.savez_compressed(path, **data)
        else:
            np.savez_compressed(path, data)

    else:
        raise ValueError(f"Unsupported file extension for saving: {path}")


def load_file(path, **kwargs):
    """
    Loads data from a file based on the extension.
    Returns None if the file does not exist.
    """
    if not os.path.exists(path):
        return None

    if path.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)

    elif path.endswith(".npy"):
        return np.load(path)

    elif path.endswith(".npz"):
        # Convert NpzFile to dict to ensure data is loaded into memory
        # and avoid issues with closed file handles
        with np.load(path) as data:
            return dict(data)

    else:
        raise ValueError(f"Unsupported file extension for loading: {path}")


def save_model(model, path):
    """
    Saves a PyTorch model's state dictionary.
    """
    ensure_dir(path)
    torch.save(model.state_dict(), path)


def load_model(model, path, device="cpu"):
    """
    Loads a PyTorch model's state dictionary.
    Returns the model with loaded weights, or None if file missing.
    """
    if not os.path.exists(path):
        return None

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model
