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
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_coef_metric(y_pred, y_true, threshold=0.5, epsilon=1e-6):
    """
    Calculates the Dice Coefficient for monitoring performance.
    Computes the global Dice score for the provided batch (treating the batch as a single volume),
    which aligns with the competition metric strategy.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits.
                               Assumes values are probabilities (0-1) for thresholding.
                               Shape: (B, C, H, W) or (B, H, W).
        y_true (torch.Tensor): Ground truth masks. Shape: same as y_pred.
        threshold (float): Threshold to convert predictions to binary mask.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The Dice coefficient.
    """
    # Detach from graph to prevent memory leaks during validation
    y_pred = y_pred.detach()
    y_true = y_true.detach()

    # Apply threshold to get binary predictions
    y_pred_bin = (y_pred > threshold).float()

    # Flatten the tensors to compute global intersection and union over the batch
    y_pred_flat = y_pred_bin.view(-1)
    y_true_flat = y_true.view(-1)

    intersection = (y_pred_flat * y_true_flat).sum()

    # Dice formula: 2 * |X n Y| / (|X| + |Y|)
    union = y_pred_flat.sum() + y_true_flat.sum()

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice.item()


def rle_encode(mask):
    """
    Run-length encoding for a binary mask.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).
                           1 indicates mask, 0 indicates background.

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten the mask column-wise (top to bottom, then left to right)
    # The competition specifies: "pixels are numbered from top to bottom, then left to right"
    # This corresponds to Fortran-style flattening in numpy.
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run (value 1), runs[1] is end of first run (value 0), etc.
    # We want starts and lengths.
    # Starts are at even indices, Ends are at odd indices.
    # Lengths = Ends - Starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)
