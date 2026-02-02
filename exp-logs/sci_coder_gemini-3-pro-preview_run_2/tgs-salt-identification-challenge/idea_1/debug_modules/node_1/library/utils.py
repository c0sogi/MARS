import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
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
    The task specifies pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) where 1 indicates salt.

    Returns:
        str: Space-delimited string of start and length pairs.
    """
    # Flatten column-wise (Fortran-style) as per task requirement
    pixels = mask.flatten(order="F")

    # Pad with zeros to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Convert to start and length
    # runs[::2] are starts, runs[1::2] are ends
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_string, shape=(101, 101)):
    """
    Decodes an RLE string into a binary mask.

    Args:
        rle_string (str): Space-delimited RLE string.
        shape (tuple): The shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    # Handle NaN or empty strings
    if (
        rle_string is None
        or (isinstance(rle_string, float) and np.isnan(rle_string))
        or rle_string == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert 1-based indexing to 0-based
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def metric_iou(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """
    Calculates the Intersection over Union (IoU) for a batch of predictions.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth masks.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities or masks.
        threshold (float): Threshold to binarize predictions if they are probabilities.
        smooth (float): Unused in exact calculation, kept for API compatibility.

    Returns:
        float: The average IoU score for the batch.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are at least 3D (Batch, H, W) or (Batch, C, H, W)
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, ...]
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis, ...]

    # Binarize predictions
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > 0.5).astype(np.uint8)

    # Flatten to (N, -1) to compute IoU per image
    batch_size = y_true.shape[0]
    y_true_f = y_true_bin.reshape(batch_size, -1)
    y_pred_f = y_pred_bin.reshape(batch_size, -1)

    intersection = np.sum(y_true_f * y_pred_f, axis=1)
    union = np.sum(y_true_f, axis=1) + np.sum(y_pred_f, axis=1) - intersection

    # IoU per image - Exact Calculation
    iou = np.zeros(batch_size, dtype=np.float32)

    # If union is 0, it means both ground truth and prediction are empty -> IoU = 1.0
    mask_empty = union == 0
    iou[mask_empty] = 1.0

    # Standard IoU for non-empty union
    mask_non_empty = ~mask_empty
    iou[mask_non_empty] = intersection[mask_non_empty] / union[mask_non_empty]

    return np.mean(iou)


def calculate_competition_metric(y_true, y_pred, threshold=0.5):
    """
    Calculates the competition metric: Mean Average Precision at different IoU thresholds.
    Thresholds: 0.5 to 0.95 with step 0.05.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth masks.
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities.
        threshold (float): Threshold to binarize predictions before IoU calculation.

    Returns:
        float: The mean average precision score.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are at least 3D
    if y_true.ndim == 2:
        y_true = y_true[np.newaxis, ...]
    if y_pred.ndim == 2:
        y_pred = y_pred[np.newaxis, ...]

    # Binarize
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > 0.5).astype(np.uint8)

    batch_size = y_true.shape[0]
    scores = []

    # IoU Thresholds for the metric: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 0.95 + 1e-6, 0.05)

    for i in range(batch_size):
        yt = y_true_bin[i].flatten()
        yp = y_pred_bin[i].flatten()

        intersection = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp) - intersection

        if union == 0:
            # Both empty -> Perfect match (IoU = 1.0)
            iou = 1.0
        else:
            iou = intersection / union

        # Vectorized match check
        # A match is counted if IoU > threshold
        matches = iou > iou_thresholds

        # Precision for this image is the mean of matches across thresholds
        scores.append(np.mean(matches))

    return np.mean(scores)
