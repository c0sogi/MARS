import numpy as np
import cv2
import random
import os
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) string.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

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


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decodes a Run-Length Encoding (RLE) string to a binary mask.

    Args:
        rle_str (str): Space-delimited RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    # Handle NaN or empty string cases
    if rle_str != rle_str or rle_str is None or rle_str == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(rle_str).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def pad_image(image, target_size=(128, 128)):
    """
    Pads an image to the target size using reflection padding.
    Used to resize 101x101 images to 128x128 for network compatibility.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).
        target_size (tuple): Target spatial dimensions (H_new, W_new).

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    if h == target_h and w == target_w:
        return image

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        return image

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def unpad_image(image, original_size=(101, 101)):
    """
    Crops an image back to the original size (center crop).
    Used to restore predictions to 101x101.

    Args:
        image (np.ndarray): Padded image.
        original_size (tuple): Original spatial dimensions (H, W).

    Returns:
        np.ndarray: Unpadded image.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = original_size

    if h == orig_h and w == orig_w:
        return image

    pad_h = h - orig_h
    pad_w = w - orig_w

    top = pad_h // 2
    left = pad_w // 2

    return image[top : top + orig_h, left : left + orig_w]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at different IoU thresholds.
    The metric sweeps over IoU thresholds from 0.5 to 0.95 with a step size of 0.05.

    Args:
        predict (np.ndarray): Predicted masks (N, H, W). Can be probabilities or binary.
        truth (np.ndarray): Ground truth masks (N, H, W). Binary.
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The calculated mAP score.
    """
    # Binarize predictions if they are probabilities
    if predict.dtype != np.uint8 and predict.dtype != bool:
        predict = (predict > threshold).astype(np.uint8)
    else:
        predict = predict.astype(np.uint8)

    truth = truth.astype(np.uint8)

    # Define thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 0.95 + 1e-5, 0.05)
    n_samples = len(predict)

    if n_samples == 0:
        return 0.0

    total_score = 0.0

    for i in range(n_samples):
        p = predict[i]
        t = truth[i]

        intersection = np.sum(p & t)
        union = np.sum(p | t)

        if union == 0:
            # Both prediction and truth are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate average precision for this image
        # For a single object task, precision at threshold t is 1 if IoU > t, else 0.
        matches = iou > iou_thresholds
        image_score = np.mean(matches)
        total_score += image_score

    return total_score / n_samples
