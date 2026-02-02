import numpy as np
import torch
from library.config import Config


def fbeta_score(y_true, y_pred, beta=0.5, epsilon=1e-7):
    """
    Calculates the F-beta score, weighting precision higher than recall when beta < 1.

    Args:
        y_true: Ground truth binary labels (numpy array or torch tensor).
        y_pred: Predicted binary labels (numpy array or torch tensor).
        beta: The beta parameter for F-score (default 0.5).
        epsilon: Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to 1D for global metric calculation
    y_true = y_true.flatten().astype(np.float32)
    y_pred = y_pred.flatten().astype(np.float32)

    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    # Calculate F-beta score
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)
    return float(score)


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.
    The pixels are numbered from left to right, then top to bottom.

    Args:
        mask: Binary mask (numpy array), where 1 indicates ink.

    Returns:
        str: Space-delimited string of start positions and run lengths.
    """
    # Flatten the mask (row-major order)
    pixels = mask.flatten()

    # Ensure binary (0 or 1)
    pixels = (pixels > 0.5).astype(np.uint8)

    # Add sentinel values at the beginning and end to detect runs at boundaries
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_pos - start_pos
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def find_best_threshold(y_true, y_probs):
    """
    Finds the optimal probability threshold to maximize the F0.5 score
    on the validation set.

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted probabilities.

    Returns:
        tuple: (best_threshold, best_score)
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_probs = y_probs.flatten()

    best_score = 0.0
    best_threshold = 0.5

    # Generate thresholds range including the end value
    thresholds = np.arange(
        Config.THRESHOLD_START, Config.THRESHOLD_END + 1e-5, Config.THRESHOLD_STEP
    )

    for thresh in thresholds:
        # Binarize predictions based on current threshold
        y_pred = (y_probs >= thresh).astype(np.uint8)

        # Calculate score
        score = fbeta_score(y_true, y_pred, beta=0.5)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
