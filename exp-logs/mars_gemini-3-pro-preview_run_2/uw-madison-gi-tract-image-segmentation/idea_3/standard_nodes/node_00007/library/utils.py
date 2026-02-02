import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right.

    Args:
        img (np.ndarray): Binary mask of shape (Height, Width).
                          1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited string of start positions and run lengths
             (e.g., '1 3 10 5').
    """
    # Flatten column-wise (Fortran style) to match the top-to-bottom, left-to-right convention
    pixels = img.flatten(order="F")

    # Prepend and append 0 to detect transitions at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def min_max_normalize(img):
    """
    Normalizes image pixel values to the range [0, 1] using min-max scaling.
    Handles cases where the image is constant (max == min).

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized image as float32.
    """
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        # If image is constant (e.g., all black), return zeros
        img = np.zeros_like(img)

    return img


def load_metadata(split):
    """
    Loads the metadata CSV file for a specific data split.

    Args:
        split (str): The dataset split to load. Options: 'train', 'val', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata dataframe.

    Raises:
        ValueError: If an invalid split name is provided.
        FileNotFoundError: If the metadata file does not exist.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        path = Config.VAL_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    return pd.read_csv(path)
