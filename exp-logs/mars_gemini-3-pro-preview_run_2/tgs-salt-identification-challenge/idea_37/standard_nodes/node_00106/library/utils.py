import os
import random
import numpy as np
import cv2
import torch
from library.config import (
    ORIG_HEIGHT,
    ORIG_WIDTH,
    IMG_HEIGHT,
    IMG_WIDTH,
    IOU_THRESHOLDS,
)


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Enforce deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_image(image):
    """
    Pads an image from (101, 101) to (128, 128) using reflection padding.
    This ensures the image dimensions are divisible by 32 for the network.

    Args:
        image: Numpy array of shape (101, 101) or (101, 101, C).

    Returns:
        Padded image of shape (128, 128) or (128, 128, C).
    """
    h, w = image.shape[:2]
    target_h, target_w = IMG_HEIGHT, IMG_WIDTH

    pad_h = target_h - h
    pad_w = target_w - w

    # If image is already target size or larger, return as is or resize (safety check)
    if pad_h <= 0 and pad_w <= 0:
        return image

    # Calculate padding for top/bottom and left/right
    # We center the image, so we split padding roughly evenly
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Use reflection padding to avoid boundary artifacts
    # BORDER_REFLECT_101 mirrors pixels without duplicating the boundary pixel
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )

    return padded_image


def unpad_image(image):
    """
    Crops an image from (128, 128) back to the original (101, 101).
    Reverses the padding logic applied in pad_image.

    Args:
        image: Numpy array of shape (128, 128) or (128, 128, C).

    Returns:
        Cropped image of shape (101, 101) or (101, 101, C).
    """
    h, w = image.shape[:2]
    orig_h, orig_w = ORIG_HEIGHT, ORIG_WIDTH

    pad_h = h - orig_h
    pad_w = w - orig_w

    if pad_h <= 0 and pad_w <= 0:
        return image

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    # Crop the center region
    if image.ndim == 3:
        return image[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w, :]
    else:
        return image[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) string format.
    The pixels are numbered from top to bottom, then left to right (Column-Major).

    Args:
        mask: Binary numpy array (0s and 1s).

    Returns:
        String containing space-delimited start positions and run lengths.
    """
    # Flatten in column-major order (Fortran-style)
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions at start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (every second element minus the previous)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes an RLE string back to a binary mask.

    Args:
        rle_str: String containing RLE data.
        shape: Tuple (height, width) of the output mask.

    Returns:
        Binary numpy array of the specified shape.
    """
    if (
        isinstance(rle_str, float)
        or rle_str is None
        or rle_str == ""
        or (isinstance(rle_str, str) and rle_str.strip() == "")
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert 1-based indexing to 0-based
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def calc_iou_batch(preds, labels):
    """
    Calculates the Intersection over Union (IoU) for a batch of images.

    Args:
        preds: Numpy array of predictions (N, H, W), binary.
        labels: Numpy array of ground truth (N, H, W), binary.

    Returns:
        Numpy array of shape (N,) containing IoU scores.
    """
    # Flatten spatial dimensions
    preds_flat = preds.reshape(preds.shape[0], -1)
    labels_flat = labels.reshape(labels.shape[0], -1)

    intersection = (preds_flat & labels_flat).sum(axis=1)
    union = (preds_flat | labels_flat).sum(axis=1)

    # Initialize IoU array
    iou = np.ones(preds.shape[0], dtype=np.float32)

    # Calculate IoU where union is non-zero
    # If union is 0, it means both pred and label are empty -> IoU = 1.0 (already set)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    return iou


def calc_map(preds, labels):
    """
    Calculates the Mean Average Precision (mAP) over a range of IoU thresholds.
    Thresholds: 0.5 to 0.95 with step 0.05.

    Args:
        preds: Binary predictions (N, H, W) or (N, H, W, 1). Can be Tensor or Numpy.
        labels: Binary ground truth (N, H, W) or (N, H, W, 1). Can be Tensor or Numpy.

    Returns:
        Float representing the mAP score.
    """
    # Convert Tensors to Numpy if necessary
    if hasattr(preds, "cpu"):
        preds = preds.detach().cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.detach().cpu().numpy()

    # Ensure binary format (0 or 1) and remove channel dim if present
    preds = (preds > 0.5).astype(np.uint8).squeeze()
    labels = (labels > 0.5).astype(np.uint8).squeeze()

    # Handle single image case (H, W) -> (1, H, W)
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]

    # Calculate IoU for each image in the batch
    ious = calc_iou_batch(preds, labels)

    # Calculate Precision at each threshold
    # For a single image binary segmentation:
    # Precision(t) = 1 if IoU > t else 0
    thresholds = np.array(IOU_THRESHOLDS)

    # matches shape: (N, num_thresholds)
    matches = ious[:, np.newaxis] > thresholds[np.newaxis, :]

    # Average precision per image (mean over thresholds)
    ap_per_image = matches.mean(axis=1)

    # Mean Average Precision over the dataset
    map_score = ap_per_image.mean()

    return float(map_score)
