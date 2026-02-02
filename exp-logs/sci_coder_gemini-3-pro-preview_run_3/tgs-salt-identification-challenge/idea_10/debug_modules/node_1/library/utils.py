import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library.config import Config


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=42):
    """Sets the random seed for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_image(image, target_size=128):
    """
    Pads an image to target_size x target_size using reflection padding.
    Handles both (H, W) and (H, W, C) images.
    Assumes input size is smaller than or equal to target_size.
    """
    h, w = image.shape[:2]

    if h == target_size and w == target_size:
        return image

    diff_h = target_size - h
    diff_w = target_size - w

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    # cv2.copyMakeBorder works for multi-channel images too
    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )
    return padded_image


def unpad_image(image, original_size=101):
    """
    Crops the center of the image to original_size x original_size.
    """
    h, w = image.shape[:2]

    if h == original_size and w == original_size:
        return image

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


def rle_encode(mask):
    """
    Encodes a binary mask to RLE string.
    Pixels are numbered from top to bottom, then left to right (F-order).
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes an RLE string to a binary mask.
    """
    if pd.isna(mask_rle) or str(mask_rle).strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape[1], shape[0]).T


def calc_map(preds, targets, threshold=0.5):
    """
    Calculates the Mean Average Precision at IoU thresholds [0.5, 0.55, ..., 0.95].

    Args:
        preds: Predicted probabilities or binary masks (N, H, W).
               If float, will be thresholded.
        targets: Ground truth binary masks (N, H, W).
        threshold: Threshold to convert probabilities to binary masks.

    Returns:
        float: The mean average precision score.
    """
    # Convert to numpy if tensor
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions if they are probabilities
    if preds.dtype == float or np.issubdtype(preds.dtype, np.floating):
        preds_bin = (preds > threshold).astype(np.uint8)
    else:
        preds_bin = preds.astype(np.uint8)

    targets_bin = targets.astype(np.uint8)

    # Handle single sample case
    if preds_bin.ndim == 2:
        preds_bin = preds_bin[np.newaxis, ...]
        targets_bin = targets_bin[np.newaxis, ...]

    batch_size = preds_bin.shape[0]
    precisions = []

    # IoU thresholds: 0.5 to 0.95 with step 0.05
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(batch_size):
        pred_mask = preds_bin[i]
        true_mask = targets_bin[i]

        # Check for empty masks
        pred_empty = not np.any(pred_mask)
        true_empty = not np.any(true_mask)

        if true_empty:
            if pred_empty:
                precisions.append(1.0)
            else:
                precisions.append(0.0)
        else:
            if pred_empty:
                precisions.append(0.0)
            else:
                # Calculate IoU
                intersection = np.sum(pred_mask & true_mask)
                union = np.sum(pred_mask | true_mask)
                iou = intersection / (union + 1e-7)

                # Calculate score for this image across all thresholds
                # Score is 1 if IoU > t, else 0
                # Average Precision = mean(scores across thresholds)
                matches = iou > iou_thresholds
                score = np.mean(matches)
                precisions.append(score)

    return np.mean(precisions)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the checkpoint
    torch.save(state, filename)

    # If this is the best model, save a copy with a distinct name
    if is_best:
        base_dir = os.path.dirname(filename)
        base_name = os.path.basename(filename)

        # Determine best filename
        # If filename is 'fold_0_checkpoint.pth', save as 'fold_0_best.pth'
        if "checkpoint" in base_name:
            best_name = base_name.replace("checkpoint", "best")
        else:
            best_name = "best_" + base_name

        best_path = os.path.join(base_dir, best_name)
        torch.save(state, best_path)
