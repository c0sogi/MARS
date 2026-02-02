import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_image(image):
    """
    Pads an image from Config.ORIG_SIZE to Config.IMG_SIZE using reflection.
    Handles both HxW and HxWxC images.

    Args:
        image (np.array): Image array of shape (101, 101) or (101, 101, C).

    Returns:
        np.array: Padded image of shape (128, 128) or (128, 128, C).
    """
    target_size = Config.IMG_SIZE
    orig_size = Config.ORIG_SIZE

    pad_total = target_size - orig_size
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Handle Grayscale (H, W) vs Multi-channel (H, W, C)
    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
    else:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        # cv2.copyMakeBorder might drop the channel dim if it's 1-channel 3D array
        if len(padded.shape) == 2 and len(image.shape) == 3:
            padded = np.expand_dims(padded, axis=-1)

    return padded


def unpad_image(image):
    """
    Crops an image from Config.IMG_SIZE back to Config.ORIG_SIZE (center crop).

    Args:
        image (np.array): Image array of shape (128, 128) or (128, 128, C).

    Returns:
        np.array: Cropped image of shape (101, 101) or (101, 101, C).
    """
    target_size = Config.IMG_SIZE
    orig_size = Config.ORIG_SIZE

    pad_total = target_size - orig_size
    pad_top = pad_total // 2

    return image[pad_top : pad_top + orig_size, pad_top : pad_top + orig_size]


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    The mask should be 1-indexed, column-major (top-down, then left-right).

    Args:
        mask (np.array): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    pixels = mask.T.flatten()
    # Concatenate 0 at ends to detect edges efficiently
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start, runs[1] is end, etc.
    # Calculate lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def iou_metric(y_pred_bin, y_true):
    """
    Calculates the IoU for a single pair of masks.
    Handles empty masks: if both empty, IoU=1. If one empty, IoU=0.
    """
    y_pred_bin = y_pred_bin.flatten()
    y_true = y_true.flatten()

    intersection = np.sum(y_pred_bin * y_true)
    union = np.sum(y_pred_bin) + np.sum(y_true) - intersection

    if union == 0:
        # Both are empty
        return 1.0

    return intersection / union


def calc_map_score(pred_masks, true_masks, thresholds=None):
    """
    Calculates the mean Average Precision at different IoU thresholds.

    Args:
        pred_masks (np.array): Binary predictions (N, H, W).
        true_masks (np.array): Binary ground truth (N, H, W).
        thresholds (list, optional): List of IoU thresholds. Defaults to 0.5:0.95:0.05.

    Returns:
        float: The mean average precision over the batch.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    ious = []
    for p, t in zip(pred_masks, true_masks):
        ious.append(iou_metric(p, t))

    # Calculate score for each image: mean(iou > threshold) over all thresholds
    scores = []
    for iou in ious:
        matches = iou > thresholds  # boolean array
        score = np.mean(matches)
        scores.append(score)

    return np.mean(scores)


def optimize_threshold(pred_probs, true_masks):
    """
    Finds the optimal binarization threshold that maximizes the mAP score.
    Performs a linear search over a range of thresholds.

    Args:
        pred_probs (np.array): Predicted probabilities (N, H, W).
        true_masks (np.array): Ground truth masks (N, H, W).

    Returns:
        tuple: (best_threshold, best_score)
    """
    best_score = -1
    best_thresh = 0.5

    # Search range from 0.3 to 0.7
    thresholds = np.arange(0.3, 0.71, 0.05)

    for t in thresholds:
        pred_bin = (pred_probs > t).astype(np.uint8)
        score = calc_map_score(pred_bin, true_masks)

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score
