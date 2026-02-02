import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).

    The output format is a space-delimited list of pairs: 'start length'.
    Pixels are numbered from top to bottom, then left to right (column-major).
    1 is pixel (1,1).

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (Height, Width).
                                           Values should be 0 (background) or 1 (foreground).

    Returns:
        str: RLE string or '-' if the mask is empty.
    """
    # Ensure input is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Flatten column-major (Fortran style) as per task description
    # "top to bottom, then left to right"
    pixels = mask.flatten(order="F")

    # Check if empty
    if np.sum(pixels) == 0:
        return "-"

    # Pad with zeros to detect start/end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    # runs[1::2] are the end indices (exclusive) of runs of 1s
    # runs[::2] are the start indices of runs of 1s
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def dice_coefficient(preds, targets, smooth=1e-6):
    """
    Computes the Dice coefficient for the provided predictions and targets.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    If 'preds' and 'targets' represent a batch, this computes the Dice score
    treating the entire batch as a single set (Global Dice for that batch).
    To compute the metric over the entire dataset, the caller should accumulate
    intersections and unions separately or pass the concatenated full dataset.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions. Can be probabilities or binary.
        targets (torch.Tensor or np.ndarray): Ground truth binary masks.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Convert numpy to torch if necessary
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Move to same device if necessary (assuming preds is the reference)
    if preds.device != targets.device:
        targets = targets.to(preds.device)

    # Flatten predictions and targets to treat them as global sets X and Y
    preds_flat = preds.reshape(-1).float()
    targets_flat = targets.reshape(-1).float()

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum()

    dice = (2.0 * intersection) / (union + smooth)

    return dice.item()
