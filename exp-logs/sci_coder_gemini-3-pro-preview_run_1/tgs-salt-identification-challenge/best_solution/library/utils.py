import os
import numpy as np
import pandas as pd
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the Config library.

    Args:
        seed (int): The seed value to set.
    """
    Config.set_seed(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask of shape (H, W).
                                           1 - salt, 0 - background.

    Returns:
        str: Space-delimited string of start positions and lengths.
             Pixels are numbered from top to bottom, then left to right.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Flatten column-wise (Fortran style) as per competition spec
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        rle_str (str): Space-delimited string of start positions and lengths.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if pd.isna(rle_str) or rle_str == "" or not isinstance(rle_str, str):
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Fill runs
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions (Column-major)
    return img.reshape(shape, order="F")


def calculate_iou_batch(preds, targets, threshold=0.5):
    """
    Calculates IoU for a batch of predictions and targets.
    Handles the case where both are empty (IoU = 1).

    Args:
        preds (np.ndarray or torch.Tensor): Predictions (N, H, W) or (N, 1, H, W).
                                            Probabilities or Binary.
        targets (np.ndarray or torch.Tensor): Ground truth (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        np.ndarray: IoU scores for each image in the batch (N,).
    """
    # Convert to numpy if tensor
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Flatten spatial dimensions: Shape becomes (N, -1)
    preds_flat = (preds.reshape(preds.shape[0], -1) > threshold).astype(np.uint8)
    targets_flat = (targets.reshape(targets.shape[0], -1) > 0.5).astype(np.uint8)

    # Calculate Intersection and Union
    intersection = np.logical_and(preds_flat, targets_flat).sum(axis=1)
    union = np.logical_or(preds_flat, targets_flat).sum(axis=1)

    # Initialize IoU array
    ious = np.zeros(preds.shape[0], dtype=np.float32)

    # Case 1: Union > 0 (Normal IoU)
    mask_union = union > 0
    ious[mask_union] = intersection[mask_union] / union[mask_union]

    # Case 2: Union == 0 (Both Empty) -> IoU = 1.0
    # If union is 0, it means both pred and target are empty.
    mask_empty = union == 0
    ious[mask_empty] = 1.0

    return ious


def calculate_map(preds, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    The metric sweeps over a range of IoU thresholds (0.5 to 0.95).
    At each threshold, precision is calculated. For a single image,
    precision is 1 if IoU > threshold, else 0.
    The score for an image is the mean of these precisions.
    The final score is the mean over the batch.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions.
        targets (np.ndarray or torch.Tensor): Ground truth.
        thresholds (np.ndarray): Array of IoU thresholds.

    Returns:
        float: The mean average precision over the batch.
    """
    # Calculate raw IoUs for the batch
    ious = calculate_iou_batch(preds, targets)

    # Compare IoUs to thresholds
    # ious: (N,)
    # thresholds: (T,)
    # matches: (N, T) - True if iou > threshold
    matches = ious[:, None] > thresholds[None, :]

    # Calculate Average Precision per image (mean over thresholds)
    # This represents the fraction of thresholds passed
    ap_per_image = matches.mean(axis=1)

    # Calculate Mean Average Precision over the batch
    return ap_per_image.mean()
