import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 0 for background, 1 for object.

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten column-wise (Fortran-style) as per competition requirement
    pixels = mask.flatten(order="F")

    # Concatenate 0 at both ends to detect transitions at the start/end of the array
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run
    # runs[1] is the end of the first run
    # The length is runs[1] - runs[0]
    # We update every second element (lengths) by subtracting the start index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def calc_map_score(pred_mask, true_mask):
    """
    Calculates the Mean Average Precision at IoU thresholds (0.5 to 0.95, step 0.05).

    Args:
        pred_mask (np.ndarray or torch.Tensor): Predicted masks (probabilities or binary).
                                                Shape (H, W) or (B, H, W).
        true_mask (np.ndarray or torch.Tensor): Ground truth masks.
                                                Shape (H, W) or (B, H, W).

    Returns:
        float: The mean average precision score across the batch.
    """
    # Convert tensors to numpy if necessary
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()

    # Binarize predictions and masks
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    true_mask = (true_mask > 0.5).astype(np.uint8)

    # Ensure inputs are 3D (Batch, H, W) for consistent processing
    if pred_mask.ndim == 2:
        pred_mask = pred_mask[np.newaxis, ...]
        true_mask = true_mask[np.newaxis, ...]

    thresholds = np.arange(0.5, 0.96, 0.05)
    scores = []

    for i in range(len(pred_mask)):
        p = pred_mask[i]
        t = true_mask[i]

        p_sum = p.sum()
        t_sum = t.sum()

        # Case 1: Both masks are empty -> Perfect match
        if p_sum == 0 and t_sum == 0:
            scores.append(1.0)
            continue

        # Case 2: One mask is empty, the other is not -> No match
        if p_sum == 0 or t_sum == 0:
            scores.append(0.0)
            continue

        # Case 3: Both masks are non-empty -> Calculate IoU
        intersection = np.sum(p & t)
        union = np.sum(p | t)
        iou = intersection / union

        # Calculate precision at each threshold
        # If IoU > threshold: TP=1, FP=0, FN=0 -> Precision = 1
        # If IoU <= threshold: TP=0, FP=1, FN=1 -> Precision = 0
        matches = iou > thresholds
        scores.append(np.mean(matches))

    return np.mean(scores)
