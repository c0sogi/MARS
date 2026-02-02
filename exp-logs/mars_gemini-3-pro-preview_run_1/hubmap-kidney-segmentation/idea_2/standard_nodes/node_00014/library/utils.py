import numpy as np
import torch
import os
import random
from library.config import Config


def set_seed(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the centralized Config.setup_reproducibility method.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    Config.setup_reproducibility(seed)


def get_device() -> torch.device:
    """
    Returns the PyTorch device based on the configuration settings.

    Returns:
        torch.device: The configured device (e.g., 'cuda' or 'cpu').
    """
    return torch.device(Config.DEVICE)


def rle_encode(img: np.ndarray) -> str:
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right,
    corresponding to column-major flattening.

    Args:
        img (np.ndarray): Binary mask array (0 for background, 1 for object).

    Returns:
        str: Space-separated RLE string (start length start length ...).
    """
    # Flatten in column-major order (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = img.flatten(order="F")

    # Prepend and append 0 to correctly detect the start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are the start indices (inclusive)
    # runs[1::2] are the end indices (exclusive)
    # The length of the run is end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape: tuple) -> np.ndarray:
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): The RLE string.
        shape (tuple): The shape of the output mask (height, width).

    Returns:
        np.ndarray: The decoded binary mask.
    """
    if not isinstance(mask_rle, str) or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Extract starts and lengths from the RLE string
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing (from RLE) to 0-based indexing (for Python)
    starts -= 1

    # Calculate end indices
    ends = starts + lengths

    # Initialize a flat array of zeros
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill in the masked regions
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to the original image dimensions using column-major order
    return img.reshape(shape, order="F")
