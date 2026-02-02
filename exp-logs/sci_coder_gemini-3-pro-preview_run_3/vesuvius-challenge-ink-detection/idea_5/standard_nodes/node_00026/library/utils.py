import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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
    Run-length encoding for a binary mask.
    The mask is flattened in row-major order (left-to-right, then top-to-bottom).

    Args:
        mask (np.ndarray or torch.Tensor): Binary mask (0 or 1).

    Returns:
        str: Space-delimited string of 'start length' pairs.
             Pixels are 1-indexed.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Ensure mask is strictly binary
    mask = (mask > 0).astype(np.uint8)

    pixels = mask.flatten()
    # Prepend and append 0 to detect transitions at the start and end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions (probabilities or binary).
        targets (np.ndarray or torch.Tensor): Ground truth labels.
        beta (float): The beta parameter (default 0.5 weights precision higher).
        threshold (float): Threshold to binarize predictions if they are probabilities.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions and targets
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = (targets > 0.5).astype(np.uint8)

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)
    return float(score)


def get_optimal_threshold(preds, targets, search_range=None):
    """
    Finds the optimal binarization threshold that maximizes the F0.5 score.

    Args:
        preds (np.ndarray or torch.Tensor): Predicted probabilities.
        targets (np.ndarray or torch.Tensor): Ground truth labels.
        search_range (tuple, optional): Tuple (start, end, step) for threshold search.
                                        If None, uses values from Config.

    Returns:
        tuple: (best_threshold, best_score)
    """
    if search_range is None:
        start = Config.THRESHOLD_SEARCH_START
        end = Config.THRESHOLD_SEARCH_END
        step = Config.THRESHOLD_SEARCH_STEP
    else:
        start, end, step = search_range

    # Use a small epsilon to ensure the end value is included if it aligns with step
    thresholds = np.arange(start, end + 1e-9, step)
    best_threshold = 0.5
    best_score = -1.0

    # Ensure inputs are numpy arrays once to avoid repeated transfer overhead
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    for thresh in thresholds:
        score = fbeta_score(preds, targets, beta=0.5, threshold=thresh)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
