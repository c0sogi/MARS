import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format for submission.
    The pixels are 1-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (Height, Width).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten column-wise (Fortran-style) to match competition indexing
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def calculate_map(
    preds, gts, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
):
    """
    Calculates the Mean Average Precision (mAP) at specified IoU thresholds.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted masks of shape (B, H, W).
                                            Can be probabilities or binary.
        gts (np.ndarray or torch.Tensor): Ground truth masks of shape (B, H, W).
        thresholds (tuple): Sequence of IoU thresholds to evaluate.

    Returns:
        float: The mean average precision across the batch and thresholds.
    """
    # Convert PyTorch tensors to NumPy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.detach().cpu().numpy()

    # Ensure inputs are binary (Threshold at 0.5)
    preds = (preds > 0.5).astype(np.uint8)
    gts = (gts > 0.5).astype(np.uint8)

    # Flatten spatial dimensions: (Batch_Size, Pixels)
    preds = preds.reshape(preds.shape[0], -1)
    gts = gts.reshape(gts.shape[0], -1)

    # Calculate Intersection and Union per image
    intersection = np.logical_and(preds, gts).sum(axis=1)
    union = np.logical_or(preds, gts).sum(axis=1)

    # Calculate IoU
    # Handle edge case: if Union is 0, both masks are empty -> IoU = 1.0
    iou = np.where(union == 0, 1.0, intersection / union)

    # Calculate Precision at each threshold
    # Expand dims for broadcasting: (Batch, 1) vs (1, Thresholds)
    iou_expanded = iou[:, np.newaxis]
    thresholds_expanded = np.array(thresholds)[np.newaxis, :]

    # Determine hits: IoU > Threshold
    matches = iou_expanded > thresholds_expanded

    # Average over thresholds for each image to get AP per image
    image_scores = np.mean(matches, axis=1)

    # Average over the batch to get mAP
    return np.mean(image_scores)
