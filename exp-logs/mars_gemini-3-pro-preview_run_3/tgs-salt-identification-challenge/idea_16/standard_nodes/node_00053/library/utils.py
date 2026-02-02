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


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask of shape (H, W) where 1 is salt, 0 is background.

    Returns:
        str: Space-delimited string of start positions and run lengths.
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    # Flatten column-wise (Fortran order)
    pixels = img.T.flatten()
    # Pad with 0s to detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited string of start positions and run lengths.
        shape (tuple): The shape (H, W) of the output mask.

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if mask_rle is None or str(mask_rle) == "nan" or str(mask_rle).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise (Fortran order)
    return img.reshape(shape, order="F")


def calc_iou(pred, target):
    """
    Calculates the Intersection over Union (IoU) for a single pair of masks.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score. Returns 1.0 if both masks are empty.
    """
    # Ensure inputs are binary
    pred = (pred > 0.5).astype(bool)
    target = (target > 0.5).astype(bool)

    intersection = (pred & target).sum()
    union = (pred | target).sum()

    if union == 0:
        # Both masks are empty, which is a perfect match
        return 1.0
    else:
        return intersection / union


def calculate_map(preds, targs):
    """
    Calculates the Mean Average Precision (mAP) over IoU thresholds [0.5, 0.55, ..., 0.95].

    Args:
        preds (np.ndarray): Batch of predicted masks (B, H, W) or probabilities.
        targs (np.ndarray): Batch of ground truth masks (B, H, W).

    Returns:
        float: The mean average precision score.
    """
    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targs = targs[np.newaxis, ...]

    thresholds = np.arange(0.5, 0.96, 0.05)
    scores = []

    for i in range(len(preds)):
        pred = preds[i]
        targ = targs[i]

        # Binarize predictions (assuming 0.5 threshold for the pixel classification)
        pred_mask = (pred > 0.5).astype(np.uint8)
        targ_mask = (targ > 0.5).astype(np.uint8)

        iou = calc_iou(pred_mask, targ_mask)

        # Calculate score for this image
        # For a single object task, precision at threshold t is 1 if IoU > t, else 0.
        # We average this binary score over all thresholds.
        matches = iou > thresholds
        image_score = np.mean(matches)
        scores.append(image_score)

    return np.mean(scores)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
