import numpy as np
import cv2
import torch
from library.config import Config


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - salt, 0 - background.

    Returns:
        str: Space-delimited RLE string.
    """
    # Flatten column-wise (Fortran-style)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_string, shape=(101, 101)):
    """
    Decodes an RLE string into a binary mask.

    Args:
        rle_string (str): Space-delimited RLE string.
        shape (tuple): Target shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if str(rle_string) == "nan" or rle_string is None or rle_string == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.split()
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

    Args:
        image (np.ndarray): Image of shape (H, W) or (H, W, C).

    Returns:
        np.ndarray: Padded image.
    """
    height, width = image.shape[:2]
    target_h, target_w = Config.IMG_SIZE, Config.IMG_SIZE

    if height == target_h and width == target_w:
        return image

    diff_h = target_h - height
    diff_w = target_w - width

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # Check if image is grayscale (2D) or multichannel (3D)
    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
    else:
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        # Ensure channel dimension is preserved if it was lost or modified (cv2 handles this usually)
        if len(padded.shape) == 2 and len(image.shape) == 3:
            padded = np.expand_dims(padded, axis=2)

    return padded


def unpad_image(image):
    """
    Crops an image from Config.IMG_SIZE back to Config.ORIG_SIZE (center crop).

    Args:
        image (np.ndarray): Image of shape (H, W) or (H, W, C).

    Returns:
        np.ndarray: Unpadded (cropped) image.
    """
    height, width = image.shape[:2]
    target_h, target_w = Config.ORIG_SIZE, Config.ORIG_SIZE

    if height == target_h and width == target_w:
        return image

    diff_h = height - target_h
    diff_w = width - target_w

    pad_top = diff_h // 2
    pad_left = diff_w // 2

    return image[pad_top : pad_top + target_h, pad_left : pad_left + target_w]


def calculate_iou_map(y_pred, y_true, threshold_range=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted binary masks.
        y_true (np.ndarray or torch.Tensor): Ground truth binary masks.
        threshold_range (np.ndarray): Array of IoU thresholds to evaluate.

    Returns:
        float: The average precision score averaged over all thresholds.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Binarize predictions if they are probabilities
    # Assuming y_pred is already thresholded (0 or 1) or boolean
    # If float, threshold at 0.5 for the base mask calculation
    if (
        y_pred.dtype == float
        or y_pred.dtype == np.float32
        or y_pred.dtype == np.float64
    ):
        y_pred = (y_pred > 0.5).astype(np.uint8)
    else:
        y_pred = y_pred.astype(np.uint8)

    y_true = y_true.astype(np.uint8)

    # Flatten for IoU calculation
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Calculate Intersection and Union
    intersection = np.sum(y_pred * y_true)
    union = np.sum(y_pred) + np.sum(y_true) - intersection

    # Handle empty union (both empty)
    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    # Calculate score across thresholds
    # Metric: For each threshold t, if IoU > t, it's a "hit" (Precision=1), else "miss" (Precision=0)
    # This simplifies to the mean of boolean comparisons
    matches = iou > threshold_range
    score = np.mean(matches)

    return float(score)


def get_batch_iou_score(y_preds, y_trues):
    """
    Calculates the average mAP score for a batch of images.

    Args:
        y_preds (np.ndarray or torch.Tensor): Batch of predicted masks (B, H, W).
        y_trues (np.ndarray or torch.Tensor): Batch of ground truth masks (B, H, W).

    Returns:
        float: Mean score for the batch.
    """
    if isinstance(y_preds, torch.Tensor):
        y_preds = y_preds.detach().cpu().numpy()
    if isinstance(y_trues, torch.Tensor):
        y_trues = y_trues.detach().cpu().numpy()

    scores = []
    for i in range(len(y_preds)):
        score = calculate_iou_map(y_preds[i], y_trues[i])
        scores.append(score)

    return np.mean(scores)
