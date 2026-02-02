import os
import random
import numpy as np
import torch
import cv2


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 is salt and 0 is background.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten column-major
    pixels = mask.T.flatten()
    # Pad with 0s to detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def pad_image(image, target_size=128):
    """
    Pads an image to the target size using reflection padding.
    Designed to convert 101x101 images to 128x128 for ResNet compatibility.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).
        target_size (int): The target spatial dimension (default 128).

    Returns:
        np.ndarray: Padded image of shape (target_size, target_size, ...).
    """
    h, w = image.shape[:2]

    if h == target_size and w == target_size:
        return image

    diff_h = target_size - h
    diff_w = target_size - w

    # Calculate padding amounts
    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # Use reflection padding to avoid boundary artifacts
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )

    return padded_image


def unpad_image(image, original_size=101):
    """
    Crops the center of an image to restore the original dimensions.

    Args:
        image (np.ndarray): Padded image of shape (H, W) or (H, W, C).
        original_size (int): The original spatial dimension (default 101).

    Returns:
        np.ndarray: Cropped image of shape (original_size, original_size, ...).
    """
    h, w = image.shape[:2]

    if h == original_size and w == original_size:
        return image

    diff_h = h - original_size
    diff_w = w - original_size

    pad_top = diff_h // 2
    pad_left = diff_w // 2

    return image[pad_top : pad_top + original_size, pad_left : pad_left + original_size]


def calc_map(preds, targets, binarization_threshold=0.5, thresholds=None):
    """
    Calculates the Mean Average Precision (mAP) at various IoU thresholds.
    Metric:
        - IoU is calculated for each image.
        - Precision is calculated at thresholds [0.5, 0.55, ..., 0.95].
        - At a specific threshold t, precision is 1 if IoU > t, else 0.
        - The score for an image is the mean precision across all thresholds.
        - The final mAP is the mean score across the batch.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions of shape (N, H, W). Assumed binary or probabilities.
        targets (np.ndarray or torch.Tensor): Ground truth of shape (N, H, W).
        binarization_threshold (float): Threshold to binarize probability maps.
        thresholds (list, optional): List of IoU thresholds. Defaults to 0.5-0.95.

    Returns:
        float: The mean average precision score.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    # Convert tensors to numpy
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions and targets
    preds = (preds > binarization_threshold).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    # Flatten spatial dimensions for vectorization: (N, H*W)
    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (preds_flat * targets_flat).sum(axis=1)
    # Union = Area(A) + Area(B) - Intersection
    union = preds_flat.sum(axis=1) + targets_flat.sum(axis=1) - intersection

    # Calculate IoU
    # Handle edge case: if union is 0 (both empty), IoU is defined as 1
    ious = np.zeros(preds.shape[0])
    union_nonzero = union > 0
    ious[union_nonzero] = intersection[union_nonzero] / union[union_nonzero]
    ious[~union_nonzero] = 1.0

    # Calculate Precision at each threshold
    # precisions shape: (num_thresholds, N)
    precisions = []
    for t in thresholds:
        # For a single image, precision is 1 if IoU > t, else 0
        precisions.append(ious > t)

    precisions = np.array(precisions)

    # Average over thresholds for each image -> shape (N,)
    ap_per_image = precisions.mean(axis=0)

    # Average over the batch -> scalar
    map_score = ap_per_image.mean()

    return float(map_score)
