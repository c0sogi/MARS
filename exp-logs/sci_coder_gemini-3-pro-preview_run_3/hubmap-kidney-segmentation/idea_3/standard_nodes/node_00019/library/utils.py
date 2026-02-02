import os
import random
import numpy as np
import torch
import logging
import sys


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The pixels are numbered from top to bottom, then left to right:
    1 is pixel (1,1), 2 is pixel (2,1), etc. This corresponds to Fortran-style flattening.

    Args:
        img (np.ndarray): Binary mask image (0s and 1s).

    Returns:
        str: RLE string "start length start length ..."
    """
    # Flatten column-wise (Fortran-style)
    pixels = img.flatten(order="F")

    # We need to prepend and append 0 to detect runs at the start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are starts of 1s (because we prepended 0)
    # runs[1::2] are ends of 1s (start of next 0 run)
    # The length is end - start
    runs[1::2] -= runs[0::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # The RLE is 1-based index, so subtract 1 for 0-based index
    starts -= 1

    # Ends are exclusive in python slice, so starts + lengths
    ends = starts + lengths

    # Initialize flattened array (size = H * W)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image (Fortran-style to match encoding order)
    return img.reshape(shape, order="F")


def compute_dice_score(y_true, y_pred, epsilon=1e-7):
    """
    Computes the Dice Coefficient between ground truth and prediction.

    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: Dice coefficient.
    """
    # Flatten arrays
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)

    # If both are empty, score is 1.0
    if union == 0:
        return 1.0

    return (2.0 * intersection + epsilon) / (union + epsilon)


def get_logger(name="Training"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # clear handlers if they exist to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
