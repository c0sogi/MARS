import numpy as np
import torch
import os
import random


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across numpy, torch, and python random.

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


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        img (np.array): Binary mask of shape (H, W), where 1 is salt and 0 is background.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited string of start positions and run lengths.
        shape (tuple): The shape (H, W) of the output mask.

    Returns:
        np.array: Binary mask of shape (H, W).
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


def calculate_iou_map(preds, gts, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95, step 0.05).

    Args:
        preds (np.array or torch.Tensor): Predicted masks. Shape (B, H, W) or (B, 1, H, W).
                                          Values should be probabilities or binary.
        gts (np.array or torch.Tensor): Ground truth masks. Shape (B, H, W) or (B, 1, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities. Default 0.5.

    Returns:
        float: The mean average precision over the batch.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.detach().cpu().numpy()

    # Handle dimensions (remove channel dim if present)
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if gts.ndim == 4:
        gts = gts.squeeze(1)

    # Binarize predictions
    preds = (preds > threshold).astype(np.uint8)
    gts = (gts > 0.5).astype(np.uint8)

    batch_size = preds.shape[0]
    metric = []

    # Thresholds for the metric: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    for i in range(batch_size):
        pred_mask = preds[i]
        gt_mask = gts[i]

        gt_sum = gt_mask.sum()
        pred_sum = pred_mask.sum()

        if gt_sum == 0:
            if pred_sum == 0:
                # Both empty: Perfect match (IoU is technically undefined/1.0)
                metric.append(1.0)
            else:
                # GT empty, Pred not: False Positive
                metric.append(0.0)
        else:
            if pred_sum == 0:
                # GT not empty, Pred empty: False Negative
                metric.append(0.0)
            else:
                # Both not empty: Calculate IoU
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = intersection / union if union > 0 else 0.0

                # Calculate average precision for this image:
                # Precision is 1 if IoU > t, else 0.
                # The score is the mean of these precisions over all thresholds.
                matches = iou > iou_thresholds
                score = np.mean(matches)
                metric.append(score)

    return np.mean(metric)
