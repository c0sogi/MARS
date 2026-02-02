import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encoding(mask: np.ndarray) -> str:
    """
    Converts a binary mask into Run-Length Encoding (RLE) format for submission.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start_index run_length).
    """
    # Ensure mask is binary and flattened in row-major order
    pixels = mask.flatten()

    # We prepend and append 0 to detect runs that start at the first pixel or end at the last
    # This allows us to find transitions from 0 to 1 (start of run) and 1 to 0 (end of run)
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # In the 'runs' array:
    # Even indices (0, 2, 4...) correspond to transitions from 0 -> 1 (Start of a run)
    # Odd indices (1, 3, 5...) correspond to transitions from 1 -> 0 (End of a run)

    # The submission format requires (Start Index, Length)
    # Current 'runs' contains (Start Index, End Index) approximately

    # Calculate lengths: End Index - Start Index
    runs[1::2] -= runs[::2]

    # Convert to space-separated string
    return " ".join(str(x) for x in runs)


def fbeta_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.5,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> float:
    """
    Computes the F-beta score for binary segmentation.

    The F0.5 score weights precision higher than recall.
    Formula: ((1 + beta^2) * p * r) / (beta^2 * p + r)
    Equivalent to: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)

    Args:
        preds (torch.Tensor): Predicted probabilities or logits.
        targets (torch.Tensor): Ground truth binary masks.
        beta (float): The beta value for the F-score. Defaults to 0.5.
        threshold (float): Threshold to convert probabilities to binary predictions.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The computed F-beta score.
    """
    # Apply threshold to get binary predictions
    y_pred = (preds > threshold).float()
    y_true = targets.float()

    # Flatten tensors to calculate metrics over the entire batch/image
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    beta_sq = beta**2

    # Calculate F-beta score
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()
