import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: RLE string 'start length start length ...'.
    """
    # Flatten column-major (Fortran style) as per competition requirement
    pixels = mask.T.flatten()

    # Handle empty mask case
    if np.sum(pixels) == 0:
        return ""

    # Pad with zeros to detect runs at the start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes an RLE string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W). Defaults to (101, 101).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    # Create flat array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape and transpose to recover original orientation
    return img.reshape(shape[::-1]).T


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean precision for a batch of images at a specific IoU threshold.

    Args:
        predict (torch.Tensor or np.ndarray): Predicted probabilities or binary masks.
        truth (torch.Tensor or np.ndarray): Ground truth masks.
        threshold (float): The IoU threshold required to count a prediction as a hit (TP).

    Returns:
        float: The mean precision score for the batch.
    """
    # Convert tensors to numpy
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Threshold probabilities to binary mask (standard 0.5 prob threshold)
    # If input is already binary (int/uint), use as is
    if predict.dtype in [np.float32, np.float64]:
        predict_binary = (predict > 0.5).astype(np.uint8)
    else:
        predict_binary = predict.astype(np.uint8)

    truth_binary = (truth > 0.5).astype(np.uint8)

    batch_size = predict.shape[0]
    precisions = []

    for i in range(batch_size):
        p = predict_binary[i]
        t = truth_binary[i]

        # Flatten to 1D for set operations
        p_flat = p.flatten()
        t_flat = t.flatten()

        intersection = np.sum(p_flat * t_flat)
        union = np.sum(p_flat) + np.sum(t_flat) - intersection

        # Calculate IoU
        if union == 0:
            # Both prediction and truth are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Determine Hit/Miss based on IoU threshold
        # Precision = 1 if IoU > threshold else 0
        if iou > threshold:
            precisions.append(1.0)
        else:
            precisions.append(0.0)

    return np.mean(precisions)


def calc_map(predict, truth):
    """
    Calculates the Mean Average Precision (mAP) over the range of IoU thresholds defined in Config.

    Args:
        predict (torch.Tensor or np.ndarray): Predicted probabilities.
        truth (torch.Tensor or np.ndarray): Ground truth masks.

    Returns:
        float: The mAP score averaged over all thresholds.
    """
    thresholds = Config.IOU_THRESHOLDS
    scores = []

    for t in thresholds:
        score = do_kaggle_metric(predict, truth, threshold=t)
        scores.append(score)

    return np.mean(scores)
