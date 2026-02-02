import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Useful for segmentation tasks to balance pixel-wise accuracy and region overlap.
    """

    def __init__(self, weight=None, size_average=True, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(
            weight=weight, reduction="mean" if size_average else "sum"
        )
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (N, C, H, W) logits
        # targets: (N, C, H, W) binary mask (0 or 1)

        # BCE Loss
        bce_loss = self.bce(inputs, targets)

        # Dice Loss
        inputs_prob = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        return bce_loss + dice_loss


def fbeta_score(outputs, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score for binary segmentation.

    Args:
        outputs (torch.Tensor): Model outputs (logits).
        targets (torch.Tensor): Ground truth binary masks.
        beta (float): Weight of precision in harmonic mean. Default 0.5.
        threshold (float): Threshold for converting probabilities to binary.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(outputs)

    # Binarize predictions
    preds = (probs > threshold).float()

    # Flatten
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    # Calculate True Positives, False Positives, False Negatives
    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    # Calculate Precision and Recall
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F-beta
    beta_sq = beta**2
    fbeta = ((1 + beta_sq) * precision * recall) / (
        beta_sq * precision + recall + epsilon
    )

    return fbeta.item()


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE).
    The pixels are numbered from left to right, then top to bottom.

    Args:
        mask (numpy.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start_position, run_length).
    """
    # Flatten the mask row-major (left to right, then top to bottom)
    pixels = mask.flatten()

    # We prepend and append 0 to detect starts and ends of runs efficiently
    # This handles cases where the run starts at index 0 or ends at the last index
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # The runs array now contains start indices of value changes.
    # Because we padded with 0, the first change must be 0->1 (start of run),
    # the second 1->0 (end of run), etc.
    # runs[0] is start of first run
    # runs[1] is start of first gap (which is end of first run + 1)

    # Calculate lengths: length = end_index - start_index
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def sigmoid(x):
    """
    Applies sigmoid function to a numpy array.
    """
    return 1 / (1 + np.exp(-x))


def min_max_normalize(image, min_val=None, max_val=None):
    """
    Normalizes an image array to the range [0, 1].

    Args:
        image (numpy.ndarray): Input image.
        min_val (float, optional): Min value for normalization. If None, computed from image.
        max_val (float, optional): Max value for normalization. If None, computed from image.

    Returns:
        numpy.ndarray: Normalized image.
    """
    if min_val is None:
        min_val = image.min()
    if max_val is None:
        max_val = image.max()

    # Avoid division by zero if image is constant
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_val) / (max_val - min_val)
