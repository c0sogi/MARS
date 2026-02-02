import numpy as np
import cv2
import torch
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are numbered from top to bottom, then left to right (Fortran order).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

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


def pad_image(image):
    """
    Pads an image from Config.ORIG_SIZE to Config.IMG_SIZE using reflection padding.
    Handles both (H, W) and (H, W, C) inputs.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    target_h, target_w = Config.IMG_SIZE, Config.IMG_SIZE

    if h == target_h and w == target_w:
        return image

    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # cv2.copyMakeBorder handles both 2D and 3D (HWC) arrays correctly
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return padded


def unpad_image(image):
    """
    Crops an image from Config.IMG_SIZE back to Config.ORIG_SIZE (center crop).
    Handles both (H, W) and (H, W, C) inputs.

    Args:
        image (np.ndarray): Padded image.

    Returns:
        np.ndarray: Unpadded image.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = Config.ORIG_SIZE, Config.ORIG_SIZE

    if h == orig_h and w == orig_w:
        return image

    pad_h = h - orig_h
    pad_w = w - orig_w

    top = pad_h // 2
    # bottom = pad_h - top
    left = pad_w // 2
    # right = pad_w - left

    if len(image.shape) == 3:
        return image[top : top + orig_h, left : left + orig_w, :]
    else:
        return image[top : top + orig_h, left : left + orig_w]


def calc_map(preds, targets, threshold_range=None):
    """
    Calculates the Mean Average Precision (mAP) at various IoU thresholds.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted masks (N, H, W) or (N, 1, H, W).
        targets (np.ndarray or torch.Tensor): Ground truth masks (N, H, W) or (N, 1, H, W).
        threshold_range (list, optional): List of thresholds. Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        float: The mean average precision score.
    """
    if threshold_range is None:
        threshold_range = np.arange(0.5, 1.0, 0.05)

    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Remove channel dim if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    # Ensure binary format (0 or 1)
    # If predictions are probabilities, threshold at 0.5 for the base mask
    if np.issubdtype(preds.dtype, np.floating):
        preds = (preds > 0.5).astype(np.uint8)
    else:
        preds = preds.astype(np.uint8)

    targets = targets.astype(np.uint8)

    ious = []

    # Calculate IoU per image
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]

        intersection = np.sum(pred & target)
        union = np.sum(pred | target)

        if union == 0:
            # Both masks are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou)

    ious = np.array(ious)

    # Calculate Precision at each threshold
    # For a single image, Precision is 1 if IoU > threshold, else 0.
    # We average this score over all thresholds for the image, then over all images.

    # Shape: (num_thresholds, N)
    threshold_matrix = np.array(threshold_range)[:, None]
    matches = (ious > threshold_matrix).astype(float)

    # Average over thresholds -> AP per image
    image_aps = np.mean(matches, axis=0)

    # Average over dataset -> mAP
    final_map = np.mean(image_aps)

    return final_map
