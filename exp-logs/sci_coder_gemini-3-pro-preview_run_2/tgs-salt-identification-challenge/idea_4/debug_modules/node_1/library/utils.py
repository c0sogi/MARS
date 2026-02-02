import numpy as np
import cv2
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    # Flatten column-wise (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape of the mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def pad_image(img, target_size=Config.INPUT_SIZE):
    """
    Pads an image to the target size using reflection padding.

    Args:
        img (np.ndarray): Input image of shape (H, W) or (H, W, C).
        target_size (int): Target height/width (assumes square).

    Returns:
        np.ndarray: Padded image.
    """
    rows, cols = img.shape[:2]

    if rows == target_size and cols == target_size:
        return img

    pad_h = target_size - rows
    pad_w = target_size - cols

    if pad_h < 0 or pad_w < 0:
        # If image is larger, crop it (though this shouldn't happen in this pipeline)
        return cv2.resize(img, (target_size, target_size))

    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return cv2.copyMakeBorder(
        img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT
    )


def unpad_image(img, original_size=Config.ORIG_SIZE):
    """
    Crops an image back to the original size (center crop).

    Args:
        img (np.ndarray): Padded image of shape (H, W) or (H, W, C).
        original_size (int): Original height/width.

    Returns:
        np.ndarray: Unpadded (cropped) image.
    """
    rows, cols = img.shape[:2]

    if rows == original_size and cols == original_size:
        return img

    pad_h = rows - original_size
    pad_w = cols - original_size

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return img[pad_top : pad_top + original_size, pad_left : pad_left + original_size]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict (np.ndarray): Predicted probabilities or masks (N, H, W) or (N, 1, H, W).
        truth (np.ndarray): Ground truth masks (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: The mean average precision score.
    """
    # Ensure inputs are numpy arrays and squeeze channel dim if present
    if predict.ndim == 4:
        predict = predict.squeeze(1)
    if truth.ndim == 4:
        truth = truth.squeeze(1)

    # Binarize predictions and truth
    p_bin = (predict > threshold).astype(np.uint8)
    t_bin = (truth > 0.5).astype(np.uint8)

    # Metric thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    batch_scores = []

    for p, t in zip(p_bin, t_bin):
        intersection = np.sum(p * t)
        union = np.sum(p) + np.sum(t) - intersection

        # Calculate IoU
        if union == 0:
            # Both prediction and truth are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate score for this image: fraction of thresholds passed
        # If IoU > threshold, it's a Hit (Precision=1), else Miss (Precision=0)
        # Average Precision for the image is the mean of these hits/misses
        score = (iou > iou_thresholds).mean()
        batch_scores.append(score)

    return np.mean(batch_scores)
