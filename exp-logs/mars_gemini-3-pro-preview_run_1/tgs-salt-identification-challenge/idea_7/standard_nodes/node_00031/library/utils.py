import numpy as np
import cv2
from library.config import Config


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        img (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes an RLE string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    # Handle NaN or empty values
    if (
        mask_rle is None
        or (not isinstance(mask_rle, str))
        or mask_rle.strip() == ""
        or mask_rle == "nan"
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def pad_image(img):
    """
    Pads an image to the input size specified in Config using reflection padding.
    Handles both (H, W) and (H, W, C) images.

    Args:
        img (np.ndarray): Input image.

    Returns:
        np.ndarray: Padded image.
    """
    h, w = img.shape[:2]
    target_h, target_w = Config.INPUT_HEIGHT, Config.INPUT_WIDTH

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        return img

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # cv2.BORDER_REFLECT_101 corresponds to PyTorch's ReflectionPad2d (excludes edge pixel)
    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def unpad_image(img, original_shape=(101, 101)):
    """
    Removes padding from an image to restore original dimensions.

    Args:
        img (np.ndarray): Padded image.
        original_shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Unpadded image.
    """
    h, w = img.shape[:2]
    orig_h, orig_w = original_shape

    pad_h = h - orig_h
    pad_w = w - orig_w

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return img[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at IoU thresholds 0.5:0.95:0.05.

    Args:
        predict (np.ndarray/list): Predicted masks (probabilities or binary).
        truth (np.ndarray/list): Ground truth masks (binary).
        threshold (float): Binarization threshold for predictions (default 0.5).

    Returns:
        float: The mean average precision score.
    """
    predict = np.array(predict)
    truth = np.array(truth)

    # Ensure inputs have at least 3 dims (N, H, W) for batch processing
    if predict.ndim == 2:
        predict = predict[None, ...]
    if truth.ndim == 2:
        truth = truth[None, ...]

    # Binarize predictions
    predict = predict > threshold
    truth = truth > 0.5

    # Flatten spatial dimensions for IoU calculation per image
    predict = predict.reshape(predict.shape[0], -1)
    truth = truth.reshape(truth.shape[0], -1)

    intersection = np.logical_and(predict, truth).sum(axis=1)
    union = np.logical_or(predict, truth).sum(axis=1)

    # Handle division by zero (empty union means both empty -> IoU = 1)
    iou = np.ones_like(intersection, dtype=float)
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    # Calculate score over thresholds
    # Thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # For each image, calculate average precision over thresholds
    # A hit is counted if IoU > threshold
    # Broadcast comparison: (N, 1) > (1, T) -> (N, T)
    matches = iou[:, None] > thresholds[None, :]

    # Mean over thresholds for each image
    image_scores = matches.mean(axis=1)

    # Mean over batch
    return np.mean(image_scores)
