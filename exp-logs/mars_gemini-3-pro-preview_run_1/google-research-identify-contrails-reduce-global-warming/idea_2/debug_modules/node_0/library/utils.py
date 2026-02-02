import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

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

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def dice_score(pred, target, threshold=0.5, epsilon=1e-6):
    """
    Calculates the Dice coefficient between predicted and target masks.

    Args:
        pred (torch.Tensor or np.ndarray): Predicted probabilities or masks.
        target (torch.Tensor or np.ndarray): Ground truth masks.
        threshold (float): Threshold to binarize predictions.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Convert numpy arrays to tensors if necessary
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # Ensure inputs are on the same device
    if pred.device != target.device:
        target = target.to(pred.device)

    # Binarize predictions based on threshold
    pred_mask = (pred > threshold).float()
    target_mask = target.float()

    # Flatten tensors to 1D
    pred_flat = pred_mask.view(-1)
    target_flat = target_mask.view(-1)

    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Handle case where both masks are empty (perfect match of background)
    if union == 0:
        return 1.0

    dice = (2.0 * intersection) / (union + epsilon)
    return dice.item()


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: RLE string 'start length start length ...' or '-' if empty.
    """
    # Flatten column-major (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # Check if mask is empty
    if np.sum(pixels) == 0:
        return "-"

    # Concatenate with 0s at ends to detect changes at the very beginning or end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes (0 to 1 or 1 to 0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # In the 'runs' array:
    # Even indices (0, 2, 4...) correspond to starts of 1s
    # Odd indices (1, 3, 5...) correspond to ends of 1s

    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    # Format as space-delimited string
    return " ".join(str(x) for x in runs)
