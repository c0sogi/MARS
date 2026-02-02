import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility by delegating to the Config class.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    Config.set_seed(seed)


def rle_encoding(mask):
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The metric uses run-length encoding on the pixel values.
    Pairs of values that contain a start position and a run length.
    Pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1).

    Args:
        mask (numpy.ndarray): Binary mask (0 or 1) of shape (height, width).

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    # Flatten the mask in row-major order (C-style)
    pixels = mask.flatten()

    # Concatenate [0] at the beginning and end to detect transitions
    # This ensures we catch runs starting at index 0 or ending at the last index
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    # runs[0] is the start of the first run (0->1)
    # runs[1] is the end of the first run (1->0), which is the start of the next gap
    # Adding 1 because indices from np.where are 0-based relative to the sliced array
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_pos - start_pos
    # runs[1::2] are the end positions (where 1 changes to 0)
    # runs[::2] are the start positions (where 0 changes to 1)
    runs[1::2] -= runs[::2]

    # Format as a space-separated string
    return " ".join(str(x) for x in runs)


def calculate_f05(y_true, y_pred, threshold=0.5, beta=0.5, epsilon=1e-7):
    """
    Calculates the F0.5 score (beta=0.5) for binary segmentation.

    The F0.5 score weights precision higher than recall.

    Args:
        y_true (torch.Tensor or numpy.ndarray): Ground truth binary mask.
        y_pred (torch.Tensor or numpy.ndarray): Predicted probabilities or logits.
        threshold (float): Threshold to binarize predictions.
        beta (float): Beta value for the F-score. Defaults to 0.5.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The calculated F0.5 score.
    """
    # Convert inputs to torch tensors if they are numpy arrays
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    # Ensure both tensors are on the same device
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)

    # Binarize predictions based on the threshold
    # Note: Assumes y_pred are probabilities or values where > threshold implies positive class
    y_pred_bin = (y_pred > threshold).float()
    y_true_bin = y_true.float()

    # Flatten the tensors to compute global metrics over the batch/image
    y_pred_bin = y_pred_bin.view(-1)
    y_true_bin = y_true_bin.view(-1)

    # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    tp = (y_pred_bin * y_true_bin).sum()
    fp = (y_pred_bin * (1 - y_true_bin)).sum()
    fn = ((1 - y_pred_bin) * y_true_bin).sum()

    # Calculate F-beta score
    # Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)

    return score.item()
