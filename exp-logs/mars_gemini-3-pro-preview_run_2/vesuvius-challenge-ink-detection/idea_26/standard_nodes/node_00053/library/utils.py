import numpy as np
import torch


def sigmoid(x):
    """
    Applies the sigmoid function to the input array or scalar.

    Args:
        x (numpy.ndarray or float): Input data.

    Returns:
        numpy.ndarray or float: Sigmoid transformed data.
    """
    return 1 / (1 + np.exp(-x))


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (numpy.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
             Pixels are numbered from left to right, then top to bottom, starting at 1.
    """
    pixels = mask.flatten()
    # Prepend and append 0 to detect runs that start at index 0 or end at the last index
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[1::2] holds the end indices, runs[::2] holds the start indices
    # We want lengths, so subtract start from end
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def dice_coefficient(preds, targets, threshold=0.5, beta=0.5, smooth=1e-6):
    """
    Calculates the F-beta score (Sørensen-Dice coefficient variant) for binary segmentation.
    This metric is used to evaluate how well the output matches the reference.

    Args:
        preds (torch.Tensor): Model predictions (logits or probabilities).
        targets (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary mask.
        beta (float): Weighting factor for precision vs recall. Default 0.5.
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        float: The calculated F-beta score.
    """
    # If predictions are logits (contain values outside [0, 1]), apply sigmoid
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)

    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Binarize predictions
    pred_mask = (preds > threshold).float()

    # Calculate True Positives, False Positives, False Negatives
    tp = (pred_mask * targets).sum()
    fp = (pred_mask * (1 - targets)).sum()
    fn = ((1 - pred_mask) * targets).sum()

    # Calculate F-beta score
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)

    return score.item()
