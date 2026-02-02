import numpy as np
import cv2
import os
import random
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
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


def pad_image(image, target_size=128):
    """
    Pads the input image to the target size using reflection padding.
    Works for (H, W) or (H, W, C) images.
    """
    h, w = image.shape[:2]
    delta_h = target_size - h
    delta_w = target_size - w

    if delta_h <= 0 and delta_w <= 0:
        return image

    top = max(0, delta_h // 2)
    bottom = max(0, delta_h - top)
    left = max(0, delta_w // 2)
    right = max(0, delta_w - left)

    # Use reflection padding to avoid boundary artifacts
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_REFLECT_101
    )
    return padded_image


def unpad_image(image, original_size=101):
    """
    Crops the image back to the original size (center crop).
    """
    h, w = image.shape[:2]
    delta_h = h - original_size
    delta_w = w - original_size

    if delta_h <= 0 and delta_w <= 0:
        return image

    top = delta_h // 2
    left = delta_w // 2

    return image[top : top + original_size, left : left + original_size]


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) string.
    The mask is expected to be 2D (H, W) or 3D (H, W, 1).
    """
    # Flatten column-wise
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_string, shape=(101, 101)):
    """
    Decodes an RLE string into a binary mask.
    """
    if not isinstance(rle_string, str) or rle_string == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the competition metric: Mean Average Precision at different IoU thresholds.

    Args:
        predict: Predicted probabilities or binary mask (N, H, W) or (H, W).
        truth: Ground truth mask (N, H, W) or (H, W).
        threshold: Threshold to binarize the predictions (if probabilities provided).

    Returns:
        float: The mean score across the batch.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Handle single image input
    if predict.ndim == 2:
        predict = predict[np.newaxis, ...]
        truth = truth[np.newaxis, ...]

    # Binarize predictions
    pred_mask = (predict > threshold).astype(np.uint8)
    true_mask = (truth > 0.5).astype(np.uint8)

    ious = []
    # Thresholds from 0.5 to 0.95 with step 0.05
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for i in range(len(pred_mask)):
        p = pred_mask[i]
        t = true_mask[i]

        intersection = np.sum((p == 1) & (t == 1))
        union = np.sum((p == 1) | (t == 1))

        if union == 0:
            # Both empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate precision for this image:
        # For each threshold, if IoU > threshold, it's a hit (1), else miss (0).
        # Average over all thresholds.
        matches = iou > iou_thresholds
        score = np.mean(matches)
        ious.append(score)

    return np.mean(ious)
