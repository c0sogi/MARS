import numpy as np
import torch
import torch.nn.functional as F
from library.config import Config


def dice_loss(pred, target, smooth=1e-6):
    """
    Computes the combined Binary Cross Entropy (BCE) and Dice Loss.

    This loss function is designed to handle class imbalance by combining
    pixel-wise classification accuracy (BCE) with overlap similarity (Dice).

    Args:
        pred (torch.Tensor): Predicted logits from the model of shape (B, 1, H, W).
        target (torch.Tensor): Ground truth binary masks of shape (B, 1, H, W).
        smooth (float): Smoothing factor to prevent division by zero in Dice calculation.

    Returns:
        torch.Tensor: The scalar loss value (BCE + Dice Loss).
    """
    # Binary Cross Entropy with Logits
    # We use BCEWithLogitsLoss implicitly via functional interface for numerical stability
    bce = F.binary_cross_entropy_with_logits(pred, target)

    # Dice Loss
    # Apply sigmoid to convert logits to probabilities
    pred_probs = torch.sigmoid(pred)

    # Flatten the tensors to compute the Dice score over the batch or image
    # Flattening (B, 1, H, W) -> (-1) treats the batch as a single continuous stream
    pred_flat = pred_probs.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Dice Coefficient: 2 * |A n B| / (|A| + |B|)
    dice_score = (2.0 * intersection + smooth) / (union + smooth)

    # Dice Loss = 1 - Dice Coefficient
    dice_loss_val = 1.0 - dice_score

    return bce + dice_loss_val


def fbeta_score(pred, target, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score, used as the primary evaluation metric (F0.5).

    The F0.5 score weights precision higher than recall (beta=0.5).

    Args:
        pred (torch.Tensor): Predicted logits of shape (B, 1, H, W).
        target (torch.Tensor): Ground truth binary masks of shape (B, 1, H, W).
        beta (float): The beta parameter for the F-score (default 0.5).
        threshold (float): The threshold to convert probabilities to binary predictions.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The computed F-beta score.
    """
    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred)

    # Binarize predictions based on the threshold
    pred_bin = (pred_probs > threshold).float()
    target_bin = target.float()

    # Flatten tensors
    pred_flat = pred_bin.view(-1)
    target_flat = target_bin.view(-1)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1.0 - target_flat)).sum()
    fn = ((1.0 - pred_flat) * target_flat).sum()

    # Calculate F-beta score
    # Formula: (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)
    beta_sq = beta**2
    numerator = (1.0 + beta_sq) * tp
    denominator = ((1.0 + beta_sq) * tp) + (beta_sq * fn) + fp

    score = numerator / (denominator + epsilon)

    return score.item()


def rle_encoding(mask):
    """
    Converts a binary mask into Run-Length Encoding (RLE) format.

    The format is a space-delimited list of pairs: start_position run_length.
    Pixels are numbered from left to right, then top to bottom, starting at 1.

    Args:
        mask (numpy.ndarray): Binary mask (0s and 1s) of shape (H, W).

    Returns:
        str: The run-length encoded string.
    """
    # Flatten the mask (row-major order)
    pixels = mask.flatten()

    # Add guard values (0) at the start and end to detect transitions cleanly
    # We concatenate [0] at the beginning and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the pixel value changes (0->1 or 1->0)
    # pixels[1:] != pixels[:-1] returns a boolean array of changes
    # np.where returns the indices where changes occur
    # We add 1 to adjust for the 0-padding at the start and to match 1-based indexing
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # 'runs' currently holds the start index of every flip.
    # Because we padded with 0, the first flip (if any) must be 0->1 (Start of run).
    # The next flip must be 1->0 (End of run), and so on.
    # runs[0::2] are the Start indices.
    # runs[1::2] are the End indices (exclusive).

    # We need to store Lengths instead of End indices.
    # Length = End - Start
    # We modify the odd indices (Ends) to store Lengths
    if len(runs) > 0:
        runs[1::2] -= runs[0::2]

    # Convert to string separated by spaces
    return " ".join(str(x) for x in runs)
