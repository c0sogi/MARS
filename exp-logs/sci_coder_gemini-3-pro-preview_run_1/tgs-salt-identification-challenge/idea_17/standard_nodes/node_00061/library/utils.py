import numpy as np
import torch
import os
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility using the Config class.

    Args:
        seed (int): The seed value to set.
    """
    Config.set_seed(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask (0s and 1s).

    Returns:
        str: Space-delimited string of RLE pairs (start length).
    """
    # Flatten in column-major order (Fortran-style)
    pixels = img.flatten(order="F")
    # Pad to detect transitions at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    # Find transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_H, Config.ORIG_W)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Target shape (height, width).

    Returns:
        np.ndarray: Binary mask.
    """
    if not isinstance(mask_rle, str) or mask_rle == "" or str(mask_rle) == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculate the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict (np.ndarray or torch.Tensor): Predicted masks or probabilities. Shape (N, H, W).
        truth (np.ndarray or torch.Tensor): Ground truth masks. Shape (N, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The mean average precision over the batch.
    """
    # Ensure inputs are numpy arrays
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    N = len(predict)
    if N == 0:
        return 0.0

    scores = []

    # Binarize predictions
    predict_bin = (predict > threshold).astype(np.uint8)
    truth_bin = (truth > 0.5).astype(np.uint8)

    # IoU Thresholds: 0.5, 0.55, ..., 0.95 (10 steps)
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for i in range(N):
        p = predict_bin[i]
        t = truth_bin[i]

        sum_p = np.sum(p)
        sum_t = np.sum(t)

        # Handle Empty Mask Cases
        if sum_p == 0 and sum_t == 0:
            # Both empty: Perfect match
            scores.append(1.0)
        elif sum_p > 0 and sum_t == 0:
            # Predicted salt where there is none: Fail
            scores.append(0.0)
        elif sum_p == 0 and sum_t > 0:
            # Missed salt: Fail
            scores.append(0.0)
        else:
            intersection = np.sum(p * t)
            union = sum_p + sum_t - intersection

            if union == 0:
                iou = 1.0
            else:
                iou = intersection / union

            # Calculate score for this image: Fraction of thresholds passed
            # "greater than" check as per metric definition
            matches = np.sum(iou > iou_thresholds)
            scores.append(matches / 10.0)

    return np.mean(scores)
