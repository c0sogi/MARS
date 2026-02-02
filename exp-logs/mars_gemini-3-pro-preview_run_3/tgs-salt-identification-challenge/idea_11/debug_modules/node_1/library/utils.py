import os
import random
import numpy as np
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across numpy, random, and torch.
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
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited RLE string.
    """
    # Transpose to handle column-major order (top-to-bottom, then left-to-right)
    pixels = mask.T.flatten()

    # We need to handle the case where the mask starts or ends with 1s
    # Concatenating 0s at both ends ensures we detect all transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the pixel value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array currently holds start and end indices of 1s
    # We need lengths for the even indices (lengths = end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W) of the mask.

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    # Handle empty or NaN masks
    if (
        pd.isna(mask_rle)
        or str(mask_rle).strip() == ""
        or str(mask_rle).lower() == "nan"
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Create flat array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape and Transpose to recover original orientation
    return img.reshape(shape[1], shape[0]).T


def calculate_map_score(preds, targets, decision_threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds ranging from 0.5 to 0.95
    with a step size of 0.05.

    Args:
        preds: (B, H, W) or (B, 1, H, W) predictions (probabilities or binary).
               Can be torch.Tensor or np.ndarray.
        targets: (B, H, W) or (B, 1, H, W) ground truth masks.
                 Can be torch.Tensor or np.ndarray.
        decision_threshold (float): Threshold to convert probability maps to binary masks.

    Returns:
        float: The mean average precision score over the batch.
    """
    # Convert Tensors to Numpy if necessary
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Flatten spatial dimensions: (B, H*W)
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    # Binarize predictions and ensure targets are binary
    preds = (preds > decision_threshold).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    # Calculate Intersection and Union per sample
    intersection = (preds & targets).sum(axis=1)
    union = (preds | targets).sum(axis=1)

    # Calculate IoU
    # Handle edge case: if Union is 0 (both pred and target are empty), IoU is defined as 1.0
    ious = np.ones(preds.shape[0])
    non_empty = union > 0
    ious[non_empty] = intersection[non_empty] / union[non_empty]

    # Define metric thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 0.96, 0.05)

    # Calculate Precision at each threshold for each image
    # For a single image, Precision(t) = 1 if IoU > t else 0
    # We compare IoU (B,) against Thresholds (T,) -> Result (B, T)
    matches = ious[:, None] > thresholds[None, :]

    # Average Precision per image is the mean over the thresholds
    ap_per_image = matches.mean(axis=1)

    # Final score is the mean AP over the batch
    return ap_per_image.mean()
