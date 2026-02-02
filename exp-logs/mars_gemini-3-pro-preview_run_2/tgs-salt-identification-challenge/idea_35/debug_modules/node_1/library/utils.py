import numpy as np
import torch
import cv2
import os
import random
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).

    Args:
        img (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

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
    return img.reshape(shape, order="F")


def pad_image_128(image):
    """
    Pads an image from 101x101 to 128x128 using reflection padding.
    Handles both (H, W) and (H, W, C) inputs.
    """
    h, w = image.shape[:2]
    target_h, target_w = Config.IMG_TARGET_SIZE, Config.IMG_TARGET_SIZE

    if h == target_h and w == target_w:
        return image

    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # cv2.copyMakeBorder handles multi-channel images correctly
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    return padded


def unpad_image_128(image):
    """
    Crops an image from 128x128 back to 101x101 (center crop).
    Handles both (H, W) and (H, W, C) inputs.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = Config.IMG_ORIG_SIZE, Config.IMG_ORIG_SIZE

    if h == orig_h and w == orig_w:
        return image

    pad_h = h - orig_h
    pad_w = w - orig_w

    top = pad_h // 2
    left = pad_w // 2

    if len(image.shape) == 3:
        return image[top : top + orig_h, left : left + orig_w, :]
    else:
        return image[top : top + orig_h, left : left + orig_w]


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at IoU thresholds [0.5, 0.95, 0.05].

    Args:
        predict (np.ndarray or torch.Tensor): Predictions (probabilities or binary).
        truth (np.ndarray or torch.Tensor): Ground truth masks.
        threshold (float): Threshold to binarize predictions if probabilities.

    Returns:
        float: The mean average precision score.
    """
    # Convert tensors to numpy
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize
    predict = (predict > threshold).astype(np.uint8)
    truth = (truth > 0.5).astype(np.uint8)

    batch_size = predict.shape[0]
    scores = []

    for i in range(batch_size):
        p = predict[i].flatten()
        t = truth[i].flatten()

        sum_p = np.sum(p)
        sum_t = np.sum(t)

        # Handle empty masks cases
        if sum_t == 0 and sum_p == 0:
            scores.append(1.0)
            continue
        if sum_t == 0 and sum_p > 0:
            scores.append(0.0)
            continue
        if sum_t > 0 and sum_p == 0:
            scores.append(0.0)
            continue

        intersection = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()

        iou = intersection / union if union > 0 else 0.0

        # Calculate score over thresholds
        thresholds = np.arange(0.5, 1.0, 0.05)
        matches = iou > thresholds
        scores.append(np.mean(matches))

    return np.mean(scores)
