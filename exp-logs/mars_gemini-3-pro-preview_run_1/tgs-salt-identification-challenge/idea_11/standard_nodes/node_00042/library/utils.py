import numpy as np
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string (start length start length ...).
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

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


def calc_iou(pred_mask, true_mask):
    """
    Calculates the Intersection over Union (IoU) for a single pair of masks.
    Handles the case where both masks are empty (IoU = 1.0).

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        true_mask (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    # Ensure inputs are treated as boolean/binary
    p = pred_mask > 0
    t = true_mask > 0

    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calc_map(preds, targets, thresholds=None):
    """
    Calculates the Mean Average Precision (mAP) over a range of IoU thresholds.
    The metric sweeps over thresholds from 0.5 to 0.95 with a step of 0.05.

    Args:
        preds (list or np.ndarray): Collection of predicted masks.
        targets (list or np.ndarray): Collection of ground truth masks.
        thresholds (list, optional): List of IoU thresholds. Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        float: The mean average precision score.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    ious = []
    for p, t in zip(preds, targets):
        ious.append(calc_iou(p, t))

    ious = np.array(ious)

    # Calculate average precision at each threshold
    precisions = []
    for thresh in thresholds:
        # A match is defined as IoU > threshold
        # For a single image, precision is 1 if match, 0 if not.
        # We take the mean over all images to get precision at this threshold.
        matches = ious > thresh
        precisions.append(np.mean(matches))

    # The final score is the mean of precisions over all thresholds
    return np.mean(precisions)
