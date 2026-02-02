import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited list of pairs (start, length) or '-' if the mask is empty.
    """
    # Flatten column-wise (Fortran order) as per task specification
    pixels = mask.flatten(order="F")

    # Handle empty prediction case
    if np.sum(pixels) == 0:
        return "-"

    # Prepend and append 0 to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[::2] are start indices, runs[1::2] are end indices
    # Calculate lengths by subtracting start from end
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coefficient(y_pred, y_true, smooth=1e-6):
    """
    Computes the Dice Coefficient.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted mask (probabilities or binary).
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    if torch.is_tensor(y_pred):
        # Flatten tensors to compute global intersection and union
        y_pred = y_pred.view(-1).float()
        y_true = y_true.view(-1).float()

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()

        return (2.0 * intersection + smooth) / (union + smooth)
    else:
        # Handle NumPy arrays
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()

        return (2.0 * intersection + smooth) / (union + smooth)


def load_metadata(split="train"):
    """
    Loads the metadata CSV for a specific dataset split.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata dataframe.
    """
    if split == "train":
        path = Config.TRAIN_METADATA_PATH
    elif split == "validation":
        path = Config.VALIDATION_METADATA_PATH
    elif split == "test":
        path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(
            f"Unknown split: {split}. Must be 'train', 'validation', or 'test'."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at: {path}")

    return pd.read_csv(path)
