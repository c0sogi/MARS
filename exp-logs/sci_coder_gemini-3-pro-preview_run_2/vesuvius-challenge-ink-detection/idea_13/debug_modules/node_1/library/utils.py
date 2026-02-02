import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the Config class method.
    """
    Config.set_seed(seed)


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    The competition specifies: "The pixels are numbered from left to right,
    then top to bottom". This corresponds to standard C-style (row-major) flattening.

    Args:
        img (np.ndarray): Binary mask image (0 for background, 1 for ink).

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Ensure image is binary
    pixels = img.flatten()

    # We prepend and append 0 to detect starts and ends of runs of 1s
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The indices in 'runs' alternate between start of 1s and end of 1s (start of 0s)
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def f05_score(preds, labels, threshold=0.5, epsilon=1e-7):
    """
    Calculates the F0.5 score (modified Sørensen–Dice coefficient) for PyTorch tensors.

    F0.5 weights precision higher than recall (beta=0.5).
    Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)

    Args:
        preds (torch.Tensor): Predicted probabilities or logits.
        labels (torch.Tensor): Ground truth binary labels.
        threshold (float): Threshold to convert probabilities to binary.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The F0.5 score.
    """
    # Apply threshold to get binary predictions
    # Assuming preds are probabilities (sigmoid applied if needed before calling or handled here)
    # If preds are raw logits, sigmoid should be applied.
    # Here we assume inputs are compatible with direct comparison or are probabilities.
    # For safety, if max val > 1, implies logits, but usually metrics take probs.
    # We will assume preds are probabilities [0, 1].

    pred_mask = (preds > threshold).float()
    true_mask = labels.float()

    # Flatten tensors
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)

    tp = (pred_mask * true_mask).sum()
    fp = (pred_mask * (1 - true_mask)).sum()
    fn = ((1 - pred_mask) * true_mask).sum()

    beta = 0.5
    beta_sq = beta**2

    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + epsilon) / (denominator + epsilon)

    return score.item()
