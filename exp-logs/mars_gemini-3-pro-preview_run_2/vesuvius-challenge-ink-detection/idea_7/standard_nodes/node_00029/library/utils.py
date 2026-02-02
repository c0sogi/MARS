import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


def fbeta_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.5,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> float:
    """
    Calculates the F-beta score for binary segmentation.

    The F-beta score is a weighted harmonic mean of precision and recall.
    For this task, beta=0.5 is used, which weights precision higher than recall.

    Args:
        preds (torch.Tensor): Predicted probabilities or logits.
        targets (torch.Tensor): Ground truth binary masks.
        beta (float): The beta parameter for the F-score. Defaults to 0.5.
        threshold (float): Threshold to convert probabilities to binary predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Ensure inputs are flattened and on the same device
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Convert probabilities to binary predictions
    y_pred = (preds > threshold).float()
    y_true = targets.float()

    # Calculate True Positives, False Positives, False Negatives
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    beta_sq = beta**2

    # Calculate F-beta score
    # Formula: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
    numerator = (1 + beta_sq) * tp
    denominator = ((1 + beta_sq) * tp) + (beta_sq * fn) + fp

    score = numerator / (denominator + epsilon)

    return score.item()


def rle_encoding(mask: np.ndarray) -> str:
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom (Row-Major).
    1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten the mask in row-major order (left to right, then top to bottom)
    pixels = mask.flatten()

    # We prepend and append 0 to detect transitions at the start and end of the array
    # This simplifies finding runs of 1s
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # In the padded array, runs of 1s start where value changes from 0 to 1
    # and end where value changes from 1 to 0.
    # Because we padded with 0 at the start, the first change (if any) must be 0->1 (start of run)
    # The indices in `runs` correspond to the starts and ends of segments.
    # Even indices (0, 2, ...) are starts of 1s
    # Odd indices (1, 3, ...) are ends of 1s

    # However, the submission format requires (start, length).
    # `runs` currently holds indices.
    # runs[0] is start of first run
    # runs[1] is start of next gap (end of first run)

    runs[1::2] -= runs[::2]  # Calculate lengths: end - start

    # Convert to string space-delimited
    return " ".join(str(x) for x in runs)
