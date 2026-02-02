import numpy as np
import torch


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           1 - salt, 0 - background.

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten column-wise (Fortran-style) as per competition spec
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def calc_iou(pred, target):
    """
    Calculates Intersection over Union (IoU) for a single pair of masks.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        # Both masks are empty, which is a perfect match
        return 1.0

    return intersection / union


def get_score(preds, targets):
    """
    Calculates the competition metric: Mean Average Precision at IoU thresholds.
    Thresholds range from 0.5 to 0.95 with a step of 0.05.

    Args:
        preds (np.ndarray): Batch of predicted binary masks (N, H, W).
        targets (np.ndarray): Batch of ground truth binary masks (N, H, W).

    Returns:
        float: The mean average precision score.
    """
    thresholds = np.arange(0.5, 1.0, 0.05)
    scores = []

    for pred, target in zip(preds, targets):
        iou = calc_iou(pred, target)

        # Calculate precision for this image across all thresholds
        # If IoU > threshold, it's a Hit (Precision=1), else Miss (Precision=0)
        matches = iou > thresholds
        image_score = np.mean(matches.astype(float))
        scores.append(image_score)

    return np.mean(scores)


def optimize_threshold(preds, targets):
    """
    Finds the optimal binarization threshold that maximizes the competition metric.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities (N, H, W) or (N, 1, H, W).
        targets (np.ndarray or torch.Tensor): Ground truth binary masks (N, H, W).

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Handle channel dimensions safely
    if preds.ndim == 4 and preds.shape[1] == 1:
        preds = preds[:, 0, :, :]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0, :, :]

    thresholds = np.arange(0.3, 0.75, 0.05)
    best_th = 0.5
    best_score = 0.0

    for th in thresholds:
        bin_preds = (preds > th).astype(np.uint8)
        score = get_score(bin_preds, targets)

        if score > best_score:
            best_score = score
            best_th = th

    return best_th, best_score
