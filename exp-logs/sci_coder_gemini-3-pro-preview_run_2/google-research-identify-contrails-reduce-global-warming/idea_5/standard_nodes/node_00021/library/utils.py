import os
import numpy as np
import torch
from library.config import seed_everything


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility by delegating to the
    configuration library's seeding function.

    Args:
        seed (int): The seed value to use.
    """
    seed_everything(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format for submission.

    The mask is flattened in column-major order (top-to-bottom, then left-to-right).
    Pixels are 1-indexed.

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (H, W).
                                           Values > 0.5 are treated as 1 (masked).

    Returns:
        str: Space-delimited list of pairs 'start length', or '-' if the mask is empty.
    """
    # Convert torch tensor to numpy if necessary
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure mask is strictly binary (0 or 1)
    mask = (mask > 0.5).astype(np.uint8)

    # Flatten in column-major order (Fortran-style) as per task requirement
    pixels = mask.flatten(order="F")

    # If mask is empty, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Pad with zeros to detect transitions at the start and end of the array
    # This simplifies logic for runs starting at index 0 or ending at index -1
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns 0-based indices. Adding 1 converts them to 1-based indexing
    # required for the RLE format relative to the original unpadded array.
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array now contains [start1, end1, start2, end2, ...]
    # Calculate lengths by subtracting starts from ends
    # Assign lengths to the odd indices (where the ends were stored)
    runs[1::2] -= runs[::2]

    # Join into a space-delimited string
    return " ".join(str(x) for x in runs)


def metric_global_dice(predictions, ground_truth):
    """
    Calculates the Global Dice Coefficient.

    Formula: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of predicted pixels and Y is the set of ground truth pixels.
    The metric is computed over the entire input arrays (flattened).

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted binary masks.
        ground_truth (np.ndarray or torch.Tensor): Ground truth binary masks.

    Returns:
        float: The Global Dice score.
    """
    # Handle Torch Tensors
    if isinstance(predictions, torch.Tensor) or isinstance(ground_truth, torch.Tensor):
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
        if not isinstance(ground_truth, torch.Tensor):
            ground_truth = torch.tensor(ground_truth)

        p_flat = predictions.flatten().float()
        t_flat = ground_truth.flatten().float()

        intersection = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()

        # If both sets are empty, the Dice score is 1.0 (perfect agreement on background)
        if union == 0:
            return 1.0

        dice = (2.0 * intersection) / union
        return dice.item()

    # Handle Numpy Arrays
    else:
        p_flat = np.asarray(predictions).flatten().astype(float)
        t_flat = np.asarray(ground_truth).flatten().astype(float)

        intersection = np.sum(p_flat * t_flat)
        union = np.sum(p_flat) + np.sum(t_flat)

        if union == 0:
            return 1.0

        dice = (2.0 * intersection) / union
        return float(dice)
