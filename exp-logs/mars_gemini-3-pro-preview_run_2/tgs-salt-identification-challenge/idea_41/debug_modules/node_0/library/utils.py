import numpy as np
import cv2
import torch
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space delimited string of start positions and run lengths.
    """
    # Flatten column-major as per competition requirement (top-to-bottom, then left-to-right)
    pixels = mask.flatten(order="F")

    # We need to find the start and end of runs of 1s
    # Prepend and append 0 to detect runs at the beginning and end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is end of first run, etc.
    # The length is end - start
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Output shape of the mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major
    return img.reshape(shape, order="F")


def pad_image(image, target_size=Config.IMG_SIZE):
    """
    Pads an image to the target size using reflection padding.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).
        target_size (int): Target spatial dimension (square).

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    diff_h = target_size - h
    diff_w = target_size - w

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    if len(image.shape) == 2:
        # Grayscale / Mask
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )
    else:
        # Multichannel
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )

    return padded


def unpad_image(image, original_size=Config.ORIG_SIZE):
    """
    Crops the image back to the original size (center crop).

    Args:
        image (np.ndarray): Padded image of shape (H, W) or (H, W, C).
        original_size (int): Original spatial dimension.

    Returns:
        np.ndarray: Unpadded image.
    """
    h, w = image.shape[:2]
    diff_h = h - original_size
    diff_w = w - original_size

    pad_top = diff_h // 2
    pad_left = diff_w // 2

    return image[pad_top : pad_top + original_size, pad_left : pad_left + original_size]


def calc_iou_batch(preds, targets):
    """
    Calculates IoU for a batch of predictions and targets.
    Handles empty ground truth cases according to competition metric.

    Args:
        preds (np.ndarray): Binary predictions (N, H, W).
        targets (np.ndarray): Binary ground truth (N, H, W).

    Returns:
        np.ndarray: IoU scores for each sample in batch (N,).
    """
    # Flatten spatial dimensions
    preds_flat = preds.reshape(len(preds), -1)
    targets_flat = targets.reshape(len(targets), -1)

    intersection = (preds_flat & targets_flat).sum(axis=1)
    union = (preds_flat | targets_flat).sum(axis=1)

    # IoU calculation
    # Standard IoU: intersection / union
    # Handle division by zero (empty union means both empty)
    iou = np.zeros(len(preds))

    non_empty_union = union > 0
    iou[non_empty_union] = intersection[non_empty_union] / union[non_empty_union]

    # Competition specific logic for empty masks:
    # If GT is empty:
    #   If Pred is empty: Score 1
    #   If Pred not empty: Score 0
    # This is naturally handled by IoU if we define IoU(empty, empty) = 1

    empty_targets = targets_flat.sum(axis=1) == 0
    empty_preds = preds_flat.sum(axis=1) == 0

    # If both empty, IoU is 1
    iou[empty_targets & empty_preds] = 1.0
    # If target empty but pred not, IoU is 0 (already set by initialization or division)
    # If target not empty but pred empty, IoU is 0 (already set)

    return iou


def calc_map_score(preds, targets, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    Args:
        preds (np.ndarray): Predicted probabilities or binary masks.
                            If float, will be binarized by `threshold`.
        targets (np.ndarray): Ground truth binary masks.
        threshold (float): Pixel binarization threshold (if preds are probabilities).

    Returns:
        float: The mean average precision score.
    """
    if np.issubdtype(preds.dtype, np.floating):
        preds_binary = (preds > threshold).astype(np.uint8)
    else:
        preds_binary = preds.astype(np.uint8)

    targets_binary = targets.astype(np.uint8)

    # Calculate exact IoU for each image
    ious = calc_iou_batch(preds_binary, targets_binary)

    # Sweep over IoU thresholds (0.5 to 0.95 step 0.05)
    iou_thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END + 1e-6, Config.THRESHOLD_STEP
    )

    # For each IoU threshold, calculate precision
    # Precision(t) = (IoU > t)
    # Average over thresholds

    precisions = []
    for t in iou_thresholds:
        # Vector of hits (1 if IoU > t, else 0)
        hits = (ious > t).astype(float)
        precisions.append(hits)

    # Shape: (num_thresholds, batch_size)
    precisions = np.array(precisions)

    # Mean over thresholds for each image -> AP per image
    ap_per_image = precisions.mean(axis=0)

    # Mean over batch -> mAP
    return np.mean(ap_per_image)


def optimize_threshold(preds, targets):
    """
    Finds the optimal pixel probability threshold that maximizes the mAP score.

    Args:
        preds (np.ndarray): Predicted probabilities.
        targets (np.ndarray): Ground truth masks.

    Returns:
        float: Best threshold.
    """
    # Search range for pixel threshold
    thresholds = np.linspace(0.3, 0.7, 21)
    best_score = -1
    best_thresh = 0.5

    for t in thresholds:
        score = calc_map_score(preds, targets, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(
        f"Threshold Optimization: Best Threshold={best_thresh:.4f}, Score={best_score:.16f}"
    )
    return best_thresh
