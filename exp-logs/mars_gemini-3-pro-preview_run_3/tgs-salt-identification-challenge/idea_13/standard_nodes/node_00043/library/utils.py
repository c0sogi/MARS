import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask (0s and 1s).

    Returns:
        str: Space-delimited string of start positions and run lengths.
             Uses 1-based indexing and column-major order.
    """
    # Flatten column-major (Fortran style) as per competition spec
    pixels = img.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major
    return img.reshape(shape, order="F")


def calculate_iou_map(predictions, ground_truths, verbose=False):
    """
    Calculates the Mean Average Precision at different IoU thresholds.
    Thresholds range from 0.5 to 0.95 with a step of 0.05.

    Args:
        predictions (list or np.ndarray): List of predicted binary masks.
        ground_truths (list or np.ndarray): List of ground truth binary masks.
        verbose (bool): If True, prints the calculated metric.

    Returns:
        float: The mean average precision score.
    """
    thresholds = np.arange(0.5, 0.96, 0.05)
    precisions = []

    for pred, gt in zip(predictions, ground_truths):
        # Flatten arrays to ensure correct pixel-wise comparison
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        intersection = np.sum(pred_flat * gt_flat)
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection

        # Handle empty mask cases
        if union == 0:
            # Both masks are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate precision for this image:
        # A "hit" (TP) is when IoU > threshold.
        # Score is average of hits across all thresholds.
        matches = iou > thresholds
        score = np.mean(matches)
        precisions.append(score)

    mean_precision = np.mean(precisions)

    if verbose:
        print(f"Mean Average Precision (mAP): {mean_precision}")

    return mean_precision
