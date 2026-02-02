import numpy as np
import cv2
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) string.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: RLE string.
    """
    # Flatten column-major (Fortran style)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoding (RLE) string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (H, W).

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

    # Reshape column-major
    return img.reshape(shape, order="F")


def pad_image(image):
    """
    Pads an image from ORIG_HEIGHT/WIDTH to IMG_HEIGHT/WIDTH using reflection padding.

    Args:
        image (np.ndarray): Input image of shape (H, W) or (H, W, C).

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]
    target_h, target_w = Config.IMG_HEIGHT, Config.IMG_WIDTH

    pad_h = target_h - h
    pad_w = target_w - w

    if pad_h < 0 or pad_w < 0:
        # Fallback if input is larger than target (unlikely given task specs)
        return cv2.resize(image, (target_w, target_h))

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Apply reflection padding
    padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )

    # Ensure exact dimensions
    if padded.shape[0] != target_h or padded.shape[1] != target_w:
        padded = cv2.resize(padded, (target_w, target_h))

    return padded


def unpad_image(image, orig_shape=(101, 101)):
    """
    Crops a padded image/mask back to the original dimensions.

    Args:
        image (np.ndarray): Padded image of shape (H, W) or (H, W, C).
        orig_shape (tuple): Original shape (H, W).

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]
    target_h, target_w = orig_shape

    pad_h = h - target_h
    pad_w = w - target_w

    if pad_h < 0 or pad_w < 0:
        return cv2.resize(image, (target_w, target_h))

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return image[pad_top : pad_top + target_h, pad_left : pad_left + target_w]


def calc_iou(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    pred = pred_mask > 0
    gt = gt_mask > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        # Both masks are empty, which is a perfect match
        return 1.0

    return intersection / union


def calc_map_score(pred_mask, gt_mask, thresholds=None):
    """
    Calculates the mean Average Precision (mAP) for a single image pair.
    The score is the mean precision across the specified IoU thresholds.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        gt_mask (np.ndarray): Ground truth binary mask.
        thresholds (list, optional): List of IoU thresholds. Defaults to Config.IOU_THRESHOLDS.

    Returns:
        float: mAP score for this image.
    """
    if thresholds is None:
        thresholds = Config.IOU_THRESHOLDS

    iou = calc_iou(pred_mask, gt_mask)

    # For a single image, precision at threshold t is 1 if IoU > t, else 0.
    # The metric is the average of these precisions.
    matches = np.array([iou > t for t in thresholds], dtype=float)
    return np.mean(matches)


def optimize_threshold(pred_probs, gt_masks):
    """
    Finds the optimal binarization threshold that maximizes mAP on the validation set.

    Args:
        pred_probs (list or np.ndarray): List of predicted probability maps.
        gt_masks (list or np.ndarray): List of ground truth masks.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Search range from 0.3 to 0.7
    thresholds = np.linspace(0.3, 0.7, 9)
    best_score = -1.0
    best_thresh = 0.5

    print(f"Optimizing threshold over {len(thresholds)} values (0.3 to 0.7)...")

    for th in thresholds:
        scores = []
        for prob, gt in zip(pred_probs, gt_masks):
            # Binarize prediction
            pred_mask = (prob > th).astype(np.uint8)
            score = calc_map_score(pred_mask, gt)
            scores.append(score)

        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_thresh = th

    print(f"Best Threshold: {best_thresh:.4f} with mAP: {best_score:.10f}")
    return best_thresh, best_score
