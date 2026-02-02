import numpy as np
import cv2
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - salt, 0 - background.

    Returns:
        str: Space-delimited string of start positions and run lengths.
             Pixels are numbered from top to bottom, then left to right (1-indexed).
    """
    # Flatten column-major
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect runs at the start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are starts, runs[1::2] are ends
    # Lengths = ends - starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Output shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # 1-indexed to 0-indexed
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-major
    return img.reshape(shape, order="F")


def pad_image(image, target_size=128):
    """
    Pads an image to the target size using reflection padding.
    Centers the original image within the padded canvas.

    Args:
        image (np.ndarray): Input image (H, W) or (H, W, C).
        target_size (int): Target height and width.

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w

    if pad_h < 0 or pad_w < 0:
        # If image is larger than target, center crop (though unlikely in this task)
        # For safety, we just return resized or raise error.
        # Given the task, we assume input is smaller (101) and target is larger (128).
        # We will proceed with padding logic assuming positive pad.
        pass

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Handle both 2D (mask) and 3D (image) arrays
    if len(image.shape) == 3:
        # Pad (top, bottom), (left, right), (0, 0) for channels
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
    else:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

    return padded


def unpad_image(image, original_shape=(101, 101)):
    """
    Crops the center of the image to restore original dimensions.

    Args:
        image (np.ndarray): Padded image.
        original_shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Cropped image.
    """
    current_h, current_w = image.shape[:2]
    target_h, target_w = original_shape

    pad_h = current_h - target_h
    pad_w = current_w - target_w

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return image[pad_top : pad_top + target_h, pad_left : pad_left + target_w]


def calc_iou(pred, target):
    """
    Calculates the Intersection over Union (IoU) for a single pair of masks.
    Handles the empty-empty case as IoU = 1.

    Args:
        pred (np.ndarray): Predicted binary mask.
        target (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    # Flatten
    p_flat = pred.flatten() > 0.5
    t_flat = target.flatten() > 0.5

    if np.sum(t_flat) == 0 and np.sum(p_flat) == 0:
        return 1.0

    intersection = np.logical_and(p_flat, t_flat).sum()
    union = np.logical_or(p_flat, t_flat).sum()

    if union == 0:
        return 0.0  # Should be covered by empty check, but safe fallback

    return intersection / union


def calc_map_score(preds, targets):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds [0.5, 0.95, 0.05].

    Args:
        preds (list or np.ndarray): List of predicted masks (N, H, W).
        targets (list or np.ndarray): List of ground truth masks (N, H, W).

    Returns:
        float: The mean average precision score.
    """
    thresholds = np.arange(0.5, 0.96, 0.05)
    ious = []

    for p, t in zip(preds, targets):
        iou = calc_iou(p, t)

        # Calculate score for this image across all thresholds
        # If IoU > threshold, it's a "hit" (Precision=1 for that threshold)
        # If IoU <= threshold, it's a "miss" (Precision=0 for that threshold)
        # The score is the mean of these binary precisions
        matches = iou > thresholds
        score = np.mean(matches)
        ious.append(score)

    return np.mean(ious)
