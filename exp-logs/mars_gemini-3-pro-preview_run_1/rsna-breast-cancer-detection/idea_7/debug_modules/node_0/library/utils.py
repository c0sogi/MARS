import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path, size=None):
    """
    Robustly loads an image from the specified path.
    Tries to use OpenCV first, then falls back to PIL.
    Raises FileNotFoundError if file doesn't exist.
    Raises IOError if image cannot be loaded.

    Args:
        path (str): Path to the image file.
        size (int, optional): Target size to resize the image (size, size).

    Returns:
        np.ndarray: Loaded image array.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Attempt to load with OpenCV
    # IMREAD_UNCHANGED is used to preserve depth (e.g. 16-bit) if present
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Fallback to PIL if OpenCV fails (e.g. specific DICOM/JP2K formats)
    if img is None:
        try:
            with Image.open(path) as pil_img:
                img = np.array(pil_img)
        except Exception:
            img = None

    if img is None:
        raise IOError(
            f"Failed to load image at {path}. The format might not be supported."
        )

    # Resize if requested
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    return img


def get_cached_data(cache_path, processing_func, load_cached=True, **kwargs):
    """
    Manages data caching using Parquet (for DataFrames) or NPY (for arrays).

    Logic:
    1. If load_cached is True and cache file exists, load and return it.
    2. Otherwise, execute processing_func(**kwargs), save the result to cache_path, and return it.

    Args:
        cache_path (str): Destination path for the cache file.
        processing_func (callable): Function to generate data if cache is missed.
        load_cached (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to processing_func.

    Returns:
        The processed data (pd.DataFrame or np.ndarray).
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Attempt to load from cache
    if load_cached and os.path.exists(cache_path):
        try:
            if cache_path.endswith(".parquet"):
                return pd.read_parquet(cache_path)
            elif cache_path.endswith(".npy"):
                return np.load(cache_path, allow_pickle=False)
            else:
                # If extension is unknown, we can't load safely, proceed to compute
                pass
        except Exception as e:
            # If loading fails (e.g. corruption), proceed to recompute
            print(
                f"Warning: Failed to load cache at {cache_path} ({e}). Recomputing..."
            )

    # Compute data
    data = processing_func(**kwargs)

    # Save to cache
    if cache_path.endswith(".parquet"):
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path)
        else:
            raise ValueError(
                f"Expected pandas DataFrame for .parquet cache, got {type(data)}."
            )
    elif cache_path.endswith(".npy"):
        if isinstance(data, np.ndarray):
            np.save(cache_path, data)
        else:
            raise ValueError(
                f"Expected numpy ndarray for .npy cache, got {type(data)}."
            )
    else:
        raise ValueError("Cache file extension must be .parquet or .npy")

    return data
