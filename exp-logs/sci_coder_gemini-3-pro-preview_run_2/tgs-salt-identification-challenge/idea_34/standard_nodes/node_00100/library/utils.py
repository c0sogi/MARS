import numpy as np
import cv2
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask into a Run-Length Encoded (RLE) string.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           1 represents the object, 0 represents background.

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style) as required by the competition
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs easily
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are starts, runs[1::2] are ends
    # Lengths are ends - starts
    runs[1::2] -= runs[0::2]

    return " ".join(str(x) for x in runs)


def rle_decode(rle_string, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        rle_string (str): Space-delimited RLE string.
        shape (tuple): Target shape of the mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(rle_string, str) or rle_string == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # One-indexed to Zero-indexed
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise (Fortran-style)
    return img.reshape(shape, order="F")


def pad_image(image, target_size=128):
    """
    Pads an image to the target size using reflection padding.
    Assumes the input image is square or handles H/W independently if needed,
    but here we assume 101x101 input and 128x128 target based on Config.

    Args:
        image (np.ndarray): Input image (H, W) or (H, W, C).
        target_size (int): Target spatial dimension (default 128).

    Returns:
        np.ndarray: Padded image.
    """
    h, w = image.shape[:2]

    delta_h = target_size - h
    delta_w = target_size - w

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Use reflection padding to reduce boundary artifacts
    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
    else:
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
        # Ensure channels are preserved if opencv messes up (usually fine for HWC)

    return padded


def crop_image(image, original_size=101):
    """
    Crops the center of the image to return to original dimensions.

    Args:
        image (np.ndarray): Padded image (H, W) or (H, W, C).
        original_size (int): Original spatial dimension (default 101).

    Returns:
        np.ndarray: Cropped image.
    """
    h, w = image.shape[:2]

    delta_h = h - original_size
    delta_w = w - original_size

    if delta_h < 0 or delta_w < 0:
        # If image is smaller than original size, return as is or pad (should not happen in this pipeline)
        return image

    top = delta_h // 2
    left = delta_w // 2

    return image[top : top + original_size, left : left + original_size]


def calculate_iou(pred_mask, true_mask):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.

    Args:
        pred_mask (np.ndarray): Predicted binary mask.
        true_mask (np.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    # Flatten to ensure 1D arrays
    p = pred_mask.flatten() > 0
    t = true_mask.flatten() > 0

    intersection = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()

    if union == 0:
        # Both masks are empty -> Perfect match
        return 1.0
    else:
        return intersection / union


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at IoU thresholds [0.5, 0.95, 0.05].

    Args:
        predict (np.ndarray): Predicted probabilities or binary masks.
                              If float, will be thresholded by `threshold`.
        truth (np.ndarray): Ground truth binary masks.
        threshold (float): Binarization threshold for predictions (default 0.5).

    Returns:
        float: The mean average precision score.
    """
    # Ensure inputs are numpy arrays
    predict = np.array(predict)
    truth = np.array(truth)

    # Binarize predictions if they are probabilities
    if predict.dtype == float or np.issubdtype(predict.dtype, np.floating):
        predict = (predict > threshold).astype(np.uint8)
    else:
        predict = predict.astype(np.uint8)

    truth = truth.astype(np.uint8)

    # IoU thresholds defined by the metric
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    total_score = 0.0
    n_samples = len(predict)

    if n_samples == 0:
        return 0.0

    for i in range(n_samples):
        iou = calculate_iou(predict[i], truth[i])

        # Calculate precision for this image across all thresholds
        # For a single object task:
        # If IoU > t: TP=1, FP=0, FN=0 -> Precision = 1
        # If IoU <= t: Precision = 0 (either FP=1 or FN=1 or both)
        # So average precision for one image is simply the proportion of thresholds passed.

        matches = iou > iou_thresholds
        image_score = np.mean(matches)
        total_score += image_score

    return total_score / n_samples


def optimize_threshold(preds, truths, num_steps=100):
    """
    Finds the optimal binarization threshold that maximizes the Kaggle metric.

    Args:
        preds (np.ndarray): Predicted probabilities.
        truths (np.ndarray): Ground truth binary masks.
        num_steps (int): Number of steps for linear search.

    Returns:
        float: Optimal threshold.
    """
    thresholds = np.linspace(0, 1, num_steps)
    best_threshold = 0.5
    best_score = -1.0

    # We can optimize this by vectorizing, but a simple loop is robust and clear
    for t in thresholds:
        # Skip extreme edges to avoid numerical instability or empty predictions if not desired
        if t < 0.01 or t > 0.99:
            continue

        score = do_kaggle_metric(preds, truths, threshold=t)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold
