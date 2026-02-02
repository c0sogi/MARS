import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten column-wise (Fortran-style) as per competition requirement
    pixels = img.flatten(order="F")
    # Pad with zeros to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    # Find transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

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


def calculate_iou_map(preds, labels, pixel_threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        preds (torch.Tensor or np.ndarray): Predictions (probabilities or binary).
                                            Shape (B, 1, H, W) or (B, H, W).
        labels (torch.Tensor or np.ndarray): Ground truth masks.
                                             Shape (B, 1, H, W) or (B, H, W).
        pixel_threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The mean average precision over the batch.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Handle dimensions: remove channel dim if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if labels.ndim == 4:
        labels = labels.squeeze(1)

    # Binarize predictions
    preds_bin = (preds > pixel_threshold).astype(np.uint8)
    labels_bin = (labels > 0.5).astype(np.uint8)

    batch_size = preds.shape[0]
    metric = []

    # IoU thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for i in range(batch_size):
        p = preds_bin[i]
        t = labels_bin[i]

        p_flat = p.flatten()
        t_flat = t.flatten()

        sum_p = np.sum(p_flat)
        sum_t = np.sum(t_flat)

        if sum_p == 0 and sum_t == 0:
            # Both empty: Perfect match
            iou = 1.0
        elif sum_p > 0 and sum_t == 0:
            # Pred not empty, GT empty: Miss
            iou = 0.0
        elif sum_p == 0 and sum_t > 0:
            # Pred empty, GT not empty: Miss
            iou = 0.0
        else:
            # Both not empty: Calculate IoU
            intersection = np.sum((p_flat * t_flat) > 0)
            union = np.sum((p_flat + t_flat) > 0)
            iou = intersection / union

        # Calculate score for this image
        # A hit is counted if IoU > threshold
        matches = iou > iou_thresholds
        score = np.mean(matches)
        metric.append(score)

    return np.mean(metric)
