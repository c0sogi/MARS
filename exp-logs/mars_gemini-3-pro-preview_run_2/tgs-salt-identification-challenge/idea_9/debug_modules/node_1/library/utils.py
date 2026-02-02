import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 represents salt, 0 represents background.

    Returns:
        str: Space-delimited RLE string (e.g., '1 3 10 5').
    """
    # Flatten in column-major order (Fortran-style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect changes at the start and end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def do_kaggle_metric(predict, truth, threshold=0.5):
    """
    Calculates the mean average precision at different IoU thresholds (0.5 to 0.95).

    The metric sweeps over IoU thresholds [0.5, 0.55, ..., 0.95].
    At each threshold, a precision score is calculated (1 if IoU > t, else 0).
    The final score is the average of these precisions.

    Args:
        predict (torch.Tensor or np.ndarray): Predicted probabilities or binary mask.
        truth (torch.Tensor or np.ndarray): Ground truth binary mask.
        threshold (float): Threshold to binarize predicted probabilities (default 0.5).

    Returns:
        float: The mean average precision score averaged over the batch.
    """
    # Convert tensors to numpy if necessary
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()

    # Binarize predictions
    predict = (predict > threshold).astype(np.uint8)
    truth = (truth > 0.5).astype(np.uint8)

    # Reshape to (N, -1) for vectorized calculation
    # If input is (H, W), treat as batch size 1
    if predict.ndim == 2:
        predict = predict.reshape(1, -1)
        truth = truth.reshape(1, -1)
    else:
        predict = predict.reshape(predict.shape[0], -1)
        truth = truth.reshape(truth.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (predict & truth).sum(axis=1)
    union = (predict | truth).sum(axis=1)

    # Calculate IoU
    # Initialize IoU to 1.0 for cases where Union is 0 (Empty Prediction matching Empty Truth)
    iou = np.ones_like(intersection, dtype=np.float32)
    mask_union = union > 0
    iou[mask_union] = intersection[mask_union] / union[mask_union]

    # Calculate Metric over thresholds
    # Thresholds: 0.5, 0.55, 0.6, ..., 0.95
    thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare IoU against all thresholds: Result shape (Batch_Size, Num_Thresholds)
    # A hit is counted if IoU > threshold
    matches = iou[:, None] > thresholds[None, :]

    # Average precision for each image (mean over thresholds)
    image_scores = matches.mean(axis=1)

    # Return mean score over the batch
    return image_scores.mean()
