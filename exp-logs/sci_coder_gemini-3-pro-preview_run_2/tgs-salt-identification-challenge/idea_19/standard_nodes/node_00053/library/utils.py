import numpy as np
import cv2
import random
import os
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
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


def do_pad(image, pad_to=128):
    """
    Pads an image to the target size using reflection padding.
    Handles both 2D (H, W) and 3D (H, W, C) images.
    Defaults to padding 101x101 images to 128x128.
    """
    h, w = image.shape[:2]

    if h >= pad_to and w >= pad_to:
        return image

    pad_h = pad_to - h
    pad_w = pad_to - w

    # Calculate padding (symmetric if even, extra on bottom/right if odd)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def do_unpad(image, original_shape=(101, 101)):
    """
    Crops an image back to its original shape, reversing the do_pad operation.
    """
    h_curr, w_curr = image.shape[:2]
    h_orig, w_orig = original_shape

    if h_curr == h_orig and w_curr == w_orig:
        return image

    # Calculate the padding that was applied
    pad_h = h_curr - h_orig
    pad_w = w_curr - w_orig

    # Identify start indices (matching the // 2 logic in do_pad)
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return image[pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]


def rle_encode(mask):
    """
    Encodes a binary mask (H, W) into a Run-Length Encoded string.
    The competition format requires column-major (Fortran) flattening.
    """
    # Flatten in column-major order
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect runs at the start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask (H, W).
    """
    if not isinstance(rle_str, str) or len(rle_str) == 0:
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to 2D using column-major order
    return img.reshape(shape, order="F")


def calculate_iou(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.
    Returns 1.0 if both masks are empty.
    """
    # Ensure boolean/binary
    p = (pred_mask > 0.5).astype(bool)
    g = (gt_mask > 0.5).astype(bool)

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_map(pred_masks, gt_masks, thresholds=None):
    """
    Calculates the Mean Average Precision (mAP) over a range of IoU thresholds.
    Default thresholds: 0.5 to 0.95 with step 0.05.

    Args:
        pred_masks: List or array of predicted masks.
        gt_masks: List or array of ground truth masks.
        thresholds: Array of thresholds to evaluate.

    Returns:
        float: The mean average precision score.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    ious = []
    for p, g in zip(pred_masks, gt_masks):
        ious.append(calculate_iou(p, g))

    ious = np.array(ious)

    # Calculate matches for each threshold
    # Shape: (num_samples, num_thresholds)
    matches = ious[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    per_image_scores = matches.mean(axis=1)

    # Mean over the entire dataset
    return per_image_scores.mean()
