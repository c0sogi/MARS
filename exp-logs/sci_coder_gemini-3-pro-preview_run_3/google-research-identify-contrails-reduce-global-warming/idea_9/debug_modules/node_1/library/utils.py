import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
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
    Encodes a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W) where 1 indicates mask and 0 indicates background.

    Returns:
        str: RLE string (start length start length ...) or '-' if the mask is empty.
    """
    # Flatten in column-major order (top to bottom, then left to right)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return "-"

    return " ".join(str(x) for x in runs)


def dice_score(preds, targets, threshold=0.5, smooth=Config.SMOOTH):
    """
    Computes the Dice Coefficient for binary segmentation.

    Args:
        preds (torch.Tensor): Predicted probabilities of shape (B, ...) or (H, W).
        targets (torch.Tensor): Ground truth binary masks of same shape as preds.
        threshold (float): Threshold to binarize predictions.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The Dice score.
    """
    # Binarize predictions
    preds = (preds > threshold).float()
    targets = targets.float()

    # Flatten to compute global intersection/union for the batch
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

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
