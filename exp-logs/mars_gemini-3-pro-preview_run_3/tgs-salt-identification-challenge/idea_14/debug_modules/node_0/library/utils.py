import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
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
    Encodes a binary mask to Run-Length Encoding (RLE).
    The mask is expected to be a numpy array.
    Pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten in column-major order (Fortran-style)
    pixels = mask.flatten(order="F")
    # Pad with 0s at ends to detect transitions at boundaries
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (runs[1::2] are ends, runs[::2] are starts)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Output shape (H, W).

    Returns:
        np.ndarray: Binary mask.
    """
    if str(mask_rle) == "nan" or mask_rle is None or str(mask_rle).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def calculate_iou_batch(y_pred, y_true, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at IoU thresholds [0.5, 0.95] step 0.05.
    Handles automatic cropping if predictions are padded (e.g., 128x128 vs 101x101).

    Args:
        y_pred: Predicted probabilities or logits. Shape (Batch, H, W) or (Batch, 1, H, W).
                Can be numpy array or torch tensor.
        y_true: Ground truth masks. Shape (Batch, H, W) or (Batch, 1, H, W).
        threshold: Threshold to binarize predicted probabilities (default 0.5).

    Returns:
        float: The mean average precision over the batch.
    """
    # Convert tensors to numpy
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    # Squeeze channels if present (B, 1, H, W) -> (B, H, W)
    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(1)
    if y_true.ndim == 4:
        y_true = y_true.squeeze(1)

    # Handle resizing/cropping if shapes mismatch
    # The task requires validation on 101x101.
    # If predictions are larger (e.g. 128x128 padded), we center crop to 101x101.
    target_h, target_w = 101, 101

    if y_pred.shape[-2:] != (target_h, target_w):
        h, w = y_pred.shape[-2:]
        if h > target_h and w > target_w:
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            y_pred = y_pred[
                :, start_h : start_h + target_h, start_w : start_w + target_w
            ]

    if y_true.shape[-2:] != (target_h, target_w):
        h, w = y_true.shape[-2:]
        if h > target_h and w > target_w:
            start_h = (h - target_h) // 2
            start_w = (w - target_w) // 2
            y_true = y_true[
                :, start_h : start_h + target_h, start_w : start_w + target_w
            ]

    # Binarize predictions and truths
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > 0.5).astype(np.uint8)

    batch_size = y_pred.shape[0]
    metric = []

    # IoU Thresholds: 0.5 to 0.95 step 0.05
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(batch_size):
        p = y_pred_bin[i]
        t = y_true_bin[i]

        intersection = np.sum(p & t)
        union = np.sum(p | t)

        if union == 0:
            # Both empty: Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate precision for this image across all thresholds
        # Since this is a single-class segmentation task per image:
        # If IoU > threshold: TP=1, FP=0, FN=0 -> Precision = 1
        # If IoU <= threshold: TP=0, FP=1 (or FN=1) -> Precision = 0
        matches = iou > iou_thresholds
        score = np.mean(matches)
        metric.append(score)

    return np.mean(metric)
