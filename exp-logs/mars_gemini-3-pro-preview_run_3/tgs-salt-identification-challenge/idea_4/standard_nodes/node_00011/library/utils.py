import numpy as np
import torch
import random
import os


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The mask is flattened in column-major order (Fortran-style) before encoding.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten in column-major order as per competition requirement
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions at start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Output shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def calc_map(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds [0.5, 0.95] with step 0.05.

    Args:
        y_true (np.ndarray): Ground truth masks, shape (N, H, W).
        y_pred_probs (np.ndarray): Predicted probabilities, shape (N, H, W).
        threshold (float): Threshold to binarize predicted probabilities.

    Returns:
        float: The mean average precision score.
    """
    # Binarize predictions
    y_pred = (y_pred_probs > threshold).astype(np.uint8)
    y_true = y_true.astype(np.uint8)

    # Flatten spatial dimensions for batch processing
    # Shapes become (N, H*W)
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (y_true_f * y_pred_f).sum(axis=1)
    union = y_true_f.sum(axis=1) + y_pred_f.sum(axis=1) - intersection

    # Calculate IoU
    # Handle case where union is 0 (both ground truth and prediction are empty)
    # If both are empty, IoU is defined as 1.0
    iou = np.ones_like(intersection, dtype=np.float32)
    non_empty_union = union > 0
    iou[non_empty_union] = intersection[non_empty_union] / union[non_empty_union]

    # Calculate score over thresholds
    # Thresholds: 0.5, 0.55, 0.6, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoU against thresholds
    # Result shape: (N, len(thresholds))
    # matches[i, j] is True if IoU of image i > threshold j
    matches = iou[:, None] > iou_thresholds[None, :]

    # Average over thresholds for each image
    image_scores = matches.mean(axis=1)

    # Return mean over the batch
    return image_scores.mean()


def optimize_thresholds(y_true, y_pred_probs, verbose=True):
    """
    Finds the optimal binarization threshold that maximizes the mAP score.

    Args:
        y_true (np.ndarray): Ground truth masks.
        y_pred_probs (np.ndarray): Predicted probabilities.
        verbose (bool): Whether to print the best threshold found.

    Returns:
        float: The optimal threshold.
    """
    best_threshold = 0.5
    best_score = -1.0

    # Sweep range 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 50)

    for th in thresholds:
        score = calc_map(y_true, y_pred_probs, threshold=th)
        if score > best_score:
            best_score = score
            best_threshold = th

    if verbose:
        print(f"Optimal Threshold: {best_threshold:.4f}, Best mAP: {best_score}")

    return best_threshold
