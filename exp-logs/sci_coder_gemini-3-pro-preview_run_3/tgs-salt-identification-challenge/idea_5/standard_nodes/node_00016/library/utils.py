import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def pad_image(image, target_size=Config.IMG_SIZE):
    """
    Pads the input image (H, W) or (H, W, C) to the target size using reflection padding.
    """
    h, w = image.shape[:2]
    diff_h = target_size - h
    diff_w = target_size - w

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
    )
    return padded_image


def unpad_image(image, original_size=Config.ORIG_SIZE):
    """
    Crops the center of the image to return to the original size.
    """
    h, w = image.shape[:2]
    diff_h = h - original_size
    diff_w = w - original_size

    pad_top = diff_h // 2
    pad_left = diff_w // 2

    if len(image.shape) == 3:
        return image[
            pad_top : pad_top + original_size, pad_left : pad_left + original_size, :
        ]
    else:
        return image[
            pad_top : pad_top + original_size, pad_left : pad_left + original_size
        ]


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding string.
    img: numpy array, 1 - mask, 0 - background
    Returns: string formatted as 'start length start length ...'
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def calc_map_score(preds, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the Mean Average Precision at IoU thresholds (0.5 to 0.95).
    preds: Binary predictions (N, H, W) or (H, W)
    targets: Binary ground truth (N, H, W) or (H, W)
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are binary
    preds = (preds > 0).astype(np.uint8)
    targets = (targets > 0).astype(np.uint8)

    # Handle single image case
    if preds.ndim == 2:
        preds = preds[np.newaxis, ...]
        targets = targets[np.newaxis, ...]

    ious = []
    for pred, target in zip(preds, targets):
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection

        if union == 0:
            # Both empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union
        ious.append(iou)

    ious = np.array(ious)

    # Calculate precision at each threshold
    # Precision is 1 if IoU > threshold, else 0 (for single class per image)
    # Shape: (N_images, N_thresholds)
    matches = ious[:, None] > thresholds[None, :]

    # Average over thresholds per image
    ap_per_image = np.mean(matches, axis=1)

    # Average over batch
    map_score = np.mean(ap_per_image)

    return map_score
