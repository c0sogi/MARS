import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited string of start positions and run lengths.
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    # Flatten column-major (Fortran style)
    pixels = img.flatten(order="F")

    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): Target shape of the mask (H, W).

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


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at IoU thresholds [0.5, ..., 0.95].

    Args:
        predict (torch.Tensor or np.ndarray): Predicted probabilities or masks (N, H, W).
        truth (torch.Tensor or np.ndarray): Ground truth masks (N, H, W).
        threshold (float): Threshold to binarize predicted probabilities.

    Returns:
        float: The mean average precision across the batch.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions
    pred_mask = (predict > threshold).astype(np.uint8)
    truth_mask = (truth > 0.5).astype(np.uint8)

    # Flatten spatial dimensions to calculate intersection and union per image
    N = pred_mask.shape[0]
    pred_mask = pred_mask.reshape(N, -1)
    truth_mask = truth_mask.reshape(N, -1)

    intersection = (pred_mask & truth_mask).sum(axis=1)
    union = (pred_mask | truth_mask).sum(axis=1)

    # Calculate IoU
    # If union is 0 (both empty), IoU is defined as 1
    iou = np.ones(N)
    mask = union > 0
    iou[mask] = intersection[mask] / union[mask]

    # Define thresholds: 0.5, 0.55, ..., 0.95
    thresholds = np.arange(0.5, 0.96, 0.05)

    # Calculate precision for each image at each threshold
    # matches shape: (N, 10)
    matches = iou[:, None] > thresholds[None, :]

    # Average precision per image (mean over thresholds)
    image_scores = matches.mean(axis=1)

    # Return mean score over the batch
    return image_scores.mean()
