import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W), where 1 indicates the object.

    Returns:
        str: Space-delimited string of start positions and run lengths.
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    # Flatten column-major (Fortran style)
    pixels = mask.T.flatten()
    # Prepend and append 0 to detect transitions at the edges
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[0] is the start of the first run, runs[1] is the end, etc.
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

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


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at IoU thresholds from 0.5 to 0.95.

    Args:
        predict (np.ndarray or torch.Tensor): Predicted masks or probabilities. Shape (N, H, W).
        truth (np.ndarray or torch.Tensor): Ground truth masks. Shape (N, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The mean average precision score over the batch.
    """
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions and ground truth
    predict = (predict > threshold).astype(np.uint8)
    truth = (truth > 0.5).astype(np.uint8)

    precisions = []

    for i in range(len(predict)):
        p = predict[i]
        t = truth[i]

        intersection = np.sum((p == 1) & (t == 1))
        union = np.sum((p == 1) | (t == 1))

        if union == 0:
            # Both prediction and truth are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # The metric sweeps over a range of IoU thresholds from 0.5 to 0.95 with a step size of 0.05
        # (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        thresholds = np.arange(0.5, 1.0, 0.05)

        # A "hit" (TP) is when IoU > threshold.
        # Since we have binary masks (one object per image), Precision is 1 if Hit, 0 if Miss.
        matches = iou > thresholds
        score = np.mean(matches)
        precisions.append(score)

    return np.mean(precisions)
