import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def rle_encoding(mask):
    """
    Converts a binary mask into a Run-Length Encoded (RLE) string.

    The metric checks that the pairs are sorted, positive, and the decoded pixel
    values are not duplicated. The pixels are numbered from left to right,
    then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

    Args:
        mask (numpy.ndarray): Binary mask (0 or 1) of shape (height, width).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    # Flatten the mask in row-major order (C-style) as per the task description
    # "left to right, then top to bottom"
    pixels = mask.flatten()

    # Prepend and append 0 to detect transitions at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run
    # runs[1] is the end of the first run (start of next gap)
    # The length is runs[1] - runs[0]
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def fbeta_score(predictions, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score for binary classification.

    The F-beta score is a weighted harmonic mean of precision and recall.
    F0.5 weights precision higher than recall.

    Args:
        predictions (torch.Tensor): Predicted probabilities or logits.
        targets (torch.Tensor): Ground truth binary labels.
        beta (float): The beta parameter for the F-score. Defaults to 0.5.
        threshold (float): Threshold to convert probabilities to binary predictions.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # Ensure inputs are tensors
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)

    # Binarize predictions
    preds_binary = (predictions > threshold).float()
    targets_binary = targets.float()

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (preds_binary * targets_binary).sum()
    fp = (preds_binary * (1 - targets_binary)).sum()
    fn = ((1 - preds_binary) * targets_binary).sum()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F-beta score
    # Formula: (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    # Equivalent using TP, FP, FN:
    # ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()
