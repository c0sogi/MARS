import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    # Benchmark = True is faster for fixed input sizes, which we have (1024x1024)
    torch.backends.cudnn.benchmark = True


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The pixels are numbered from top to bottom, then left to right (Fortran-style).

    Args:
        img (np.ndarray): Binary mask of shape (height, width), where 1 indicates the object.

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask of shape (height, width).
    """
    # Check for NaN (float), None, or empty string
    if mask_rle != mask_rle or mask_rle is None or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def dice_coeff(pred, target, smooth=1e-6):
    """
    Calculates the Dice Coefficient for validation.

    Args:
        pred (torch.Tensor or np.ndarray): Predicted mask (probabilities or binary).
        target (torch.Tensor or np.ndarray): Ground truth mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)

    # Flatten tensors
    pred = pred.contiguous().view(-1).float()
    target = target.contiguous().view(-1).float()

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()
