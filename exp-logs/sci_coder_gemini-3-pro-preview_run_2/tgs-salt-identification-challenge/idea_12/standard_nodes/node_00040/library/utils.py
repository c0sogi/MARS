import numpy as np
import torch
import random
import os


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The format is a space-delimited list of pairs (start_pixel, run_length).
    Pixels are 1-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: RLE string.
    """
    # Flatten column-wise (Fortran order) to match the top-to-bottom, left-to-right requirement
    pixels = mask.T.flatten()

    # Pad with 0s to detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    # pixels[1:] != pixels[:-1] gives a boolean array of changes
    # np.where returns indices, we add 1 to adjust for 0-based indexing and the shift
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    # The length of a run is end_pos - start_pos
    if len(runs) >= 2:
        runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def calculate_iou_map(preds, masks, threshold=0.5):
    """
    Calculates the Mean Average Precision at IoU thresholds ranging from 0.5 to 0.95.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted masks (N, H, W) or (N, 1, H, W).
                                            Can be probabilities or binary.
        masks (np.ndarray or torch.Tensor): Ground truth masks (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions if provided as probabilities.

    Returns:
        float: The mean average precision score.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()

    # Remove channel dimension if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # Binarize predictions and masks
    preds = (preds > threshold).astype(np.uint8)
    masks = (masks > 0.5).astype(np.uint8)

    batch_size = preds.shape[0]
    scores = []

    # Define thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(batch_size):
        p = preds[i]
        m = masks[i]

        intersection = np.sum(p & m)
        union = np.sum(p | m)

        # Handle empty masks case
        if union == 0:
            # If both prediction and mask are empty, IoU is 1.0
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate precision for this image:
        # A "hit" is when IoU > threshold.
        # The score for the image is the mean precision over all thresholds.
        matches = iou > iou_thresholds
        score = np.mean(matches)
        scores.append(score)

    return np.mean(scores)
