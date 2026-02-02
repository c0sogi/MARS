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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The mask is flattened in column-major order (top-to-bottom, then left-to-right).
    The output format is a space-delimited list of pairs: 'start length'.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 indicates the object.

    Returns:
        str: RLE string or '-' if the mask is empty.
    """
    # Flatten in column-major order (F) as required by the task
    pixels = mask.flatten(order="F")

    # Handle empty masks
    if np.sum(pixels) == 0:
        return "-"

    # Pad with 0s at start and end to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is end of first run (exclusive)
    # The submission format requires length, so we subtract start from end
    # runs[1::2] are the end indices, runs[::2] are the start indices
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coef_metric(y_pred, y_true, smooth=1e-6, threshold=Config.PIXEL_THRESHOLD):
    """
    Calculates the Dice Coefficient between predictions and ground truth.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
        y_true (torch.Tensor or np.ndarray): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.
        threshold (float): Threshold to binarize probabilities. Defaults to Config.PIXEL_THRESHOLD.

    Returns:
        float: The Dice coefficient.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)

    # Ensure tensors are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)

    # Flatten the tensors to compute global intersection/union for the batch/image
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Binarize predictions if they are probabilities (floating point)
    if torch.is_floating_point(y_pred):
        y_pred = (y_pred > threshold).float()

    # Ensure ground truth is float for calculation
    if not torch.is_floating_point(y_true):
        y_true = y_true.float()

    # Calculate Intersection and Union
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()

    # Compute Dice
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()
