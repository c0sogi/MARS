import numpy as np
import torch
from library.config import Config


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility.
    Delegates to the Config class to ensure consistency across all libraries.

    Args:
        seed (int, optional): The seed value. If None, uses Config.SEED.
    """
    if seed is None:
        seed = Config.SEED
    Config.set_seed(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 0 for background, 1 for salt.

    Returns:
        str: RLE string 'start length start length ...'
    """
    # Flatten column-wise (Fortran-style) to match the requirement:
    # "numbered from top to bottom, then left to right"
    pixels = mask.T.flatten()

    # Pad with 0s to detect runs at the very start or end of the array
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes (0->1 or 1->0)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is the start of the first run of 1s
    # runs[1] is the start of the first run of 0s (which marks the end of the first run of 1s)
    # Calculate lengths: length = end - start
    # We update the even indices (lengths) by subtracting the odd indices (starts)
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict (np.ndarray or torch.Tensor): Predicted probabilities or binary masks.
                                              Shape (N, H, W) or (N, 1, H, W).
        truth (np.ndarray or torch.Tensor): Ground truth binary masks.
                                            Shape (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize the predicted probabilities.
                           Defaults to 0.5.

    Returns:
        float: The mean average precision score across the batch.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Remove channel dimension if present (e.g., N, 1, H, W -> N, H, W)
    if predict.ndim == 4:
        predict = predict.squeeze(1)
    if truth.ndim == 4:
        truth = truth.squeeze(1)

    # Binarize predictions based on the supplied threshold
    predict = (predict > threshold).astype(np.uint8)
    truth = (truth > 0.5).astype(np.uint8)

    batch_size = predict.shape[0]
    precisions = []

    # Define IoU thresholds: 0.5, 0.55, 0.6, ..., 0.95
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    for i in range(batch_size):
        p = predict[i]
        t = truth[i]

        intersection = np.sum(p * t)
        union = np.sum(p) + np.sum(t) - intersection

        # Calculate IoU
        if union == 0:
            # Both prediction and truth are empty -> Perfect match
            iou = 1.0
        else:
            iou = intersection / union

        # Calculate Average Precision for this image
        # A match is counted if IoU > threshold
        matches = iou > iou_thresholds
        avg_precision = np.mean(matches)
        precisions.append(avg_precision)

    return np.mean(precisions)
