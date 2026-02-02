import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for submission.

    The pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           1 indicates a masked pixel, 0 indicates background.

    Returns:
        str: A space-delimited list of pairs (start, length).
             Returns '-' if the mask is empty.
    """
    # Flatten column-wise (Fortran-style) to match "top to bottom, then left to right"
    pixels = mask.flatten(order="F")

    # If no pixels are masked, return '-'
    if np.sum(pixels) == 0:
        return "-"

    # Prepend and append 0 to detect transitions between 0 and 1
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # np.where returns indices in the padded array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array contains [start1, end1, start2, end2, ...]
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    # Format as a space-separated string
    return " ".join(str(x) for x in runs)


def dice_coef_metric(y_pred, y_true, threshold=0.5, epsilon=1e-6):
    """
    Computes the Dice Coefficient for binary segmentation.

    Formula: 2 * |X n Y| / (|X| + |Y|)

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits.
        y_true (torch.Tensor): Ground truth binary masks (0 or 1).
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Ensure inputs are float tensors
    y_pred = y_pred.float()
    y_true = y_true.float()

    # Binarize predictions based on the threshold
    y_pred_bin = (y_pred > threshold).float()

    # Flatten the tensors to compute the metric over the entire input batch/volume
    y_pred_f = y_pred_bin.view(-1)
    y_true_f = y_true.view(-1)

    # Calculate intersection and union
    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum()

    # Compute Dice score
    dice = (2.0 * intersection) / (union + epsilon)

    return dice.item()
