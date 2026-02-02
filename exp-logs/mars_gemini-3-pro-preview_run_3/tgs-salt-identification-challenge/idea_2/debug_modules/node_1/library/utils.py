import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_image(image):
    """
    Pads an image to the size specified in Config.IMG_SIZE (128x128) using reflection padding.
    This ensures the image dimensions are divisible by 32 for the ResNeXt encoder.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).

    Returns:
        np.ndarray: Padded image of shape (Config.IMG_SIZE, Config.IMG_SIZE, ...).
    """
    target_size = Config.IMG_SIZE
    h, w = image.shape[:2]

    delta_h = target_size - h
    delta_w = target_size - w

    # Calculate padding amounts
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Use reflection padding to minimize boundary artifacts
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_REFLECT_101
    )
    return padded_image


def unpad_image(image, original_shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Crops the image back to the original shape, removing the padding applied by pad_image.

    Args:
        image (np.ndarray): Padded image/mask.
        original_shape (tuple): The target (H, W) to crop to.

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = original_shape

    delta_h = h - orig_h
    delta_w = w - orig_w

    top = delta_h // 2
    left = delta_w // 2

    return image[top : top + orig_h, left : left + orig_w]


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) string format.
    The mask is flattened in column-major order (Fortran-style) as per competition requirements.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten in column-major order
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        rle_str (str): RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    # Handle NaN or empty strings
    if (
        rle_str is None
        or (isinstance(rle_str, float) and np.isnan(rle_str))
        or str(rle_str).strip() == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = str(rle_str).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape, order="F")


def calculate_iou_map(preds, labels, pixel_threshold=0.5):
    """
    Calculates the Mean Average Precision at IoU thresholds (0.5 to 0.95, step 0.05).

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or logits.
                                            Shape (B, H, W) or (B, 1, H, W).
        labels (torch.Tensor or np.ndarray): Ground truth masks.
                                             Shape (B, H, W) or (B, 1, H, W).
        pixel_threshold (float): Threshold to convert probabilities to binary mask.

    Returns:
        float: The mean average precision score over the batch.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Squeeze channel dim if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if labels.ndim == 4:
        labels = labels.squeeze(1)

    # Apply sigmoid if predictions are logits (heuristically determined by range)
    if preds.min() < 0 or preds.max() > 1:
        preds = 1 / (1 + np.exp(-preds))

    # Binarize predictions
    pred_masks = (preds > pixel_threshold).astype(np.uint8)
    true_masks = (labels > 0.5).astype(np.uint8)

    # Metric thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    ious = []

    for p, t in zip(pred_masks, true_masks):
        intersection = np.sum((p == 1) & (t == 1))
        union = np.sum((p == 1) | (t == 1))

        # Calculate IoU
        if union == 0:
            # Both empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate Average Precision for this image
        # Precision at threshold t is 1 if IoU > t, else 0
        # We take the mean over all thresholds
        matches = iou > iou_thresholds
        average_precision = np.mean(matches)
        ious.append(average_precision)

    return np.mean(ious)
