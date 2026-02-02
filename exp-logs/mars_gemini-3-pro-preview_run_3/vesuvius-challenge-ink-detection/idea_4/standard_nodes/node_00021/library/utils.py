import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).
    The pixels are numbered from left to right, then top to bottom (row-major).
    Indices are 1-based.

    Args:
        img (np.ndarray): Binary mask of shape (H, W) where 1 indicates ink.

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten the image in row-major order
    pixels = img.flatten()

    # Prepend and append 0 to detect starts and ends of runs efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The 'runs' array contains start indices of 1s and start indices of 0s (which are ends of 1s)
    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, threshold=0.5, beta=0.5, smooth=1e-6):
    """
    Calculates the F-beta score for binary classification.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or binary map.
        targets (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to binarize predictions.
        beta (float): The beta value for the F-score (default 0.5 weights precision higher).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure inputs are on the same device
    if preds.device != targets.device:
        targets = targets.to(preds.device)

    # Binarize predictions
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    # Flatten tensors
    preds_bin = preds_bin.reshape(-1)
    targets_bin = targets_bin.reshape(-1)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    # Calculate F-beta Score
    # Formula: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    beta_sq = beta**2
    fbeta = (
        (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + smooth)
    )

    return fbeta.item()
