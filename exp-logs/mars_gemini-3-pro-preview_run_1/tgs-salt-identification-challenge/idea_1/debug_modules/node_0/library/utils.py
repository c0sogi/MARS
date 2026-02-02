import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(img):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask image of shape (H, W).
                          0 - background, 1 - foreground/salt.

    Returns:
        str: Space-delimited string of pairs (start_position, run_length).
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    # Flatten column-wise (Fortran-style) as per competition requirement
    pixels = img.flatten(order="F")

    # Prepend and append 0 to detect transitions efficiently
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0::2] are the start indices (1-based because of the prepended 0)
    # runs[1::2] are the end indices
    # Calculate lengths
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): Space-delimited RLE string.
        shape (tuple): Target shape of the mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "" or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flat array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions (Fortran-style)
    return img.reshape(shape, order="F")


def calculate_iou_batch(y_pred, y_true, threshold=0.5, smooth=1e-6):
    """
    Calculates the mean Intersection over Union (IoU) for a batch of predictions.

    Args:
        y_pred (torch.Tensor): Predicted probabilities or logits.
        y_true (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to convert predictions to binary.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: Mean IoU for the batch.
    """
    # Ensure inputs are tensors
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)

    # Binarize predictions based on threshold
    pred_mask = (y_pred > threshold).float()
    true_mask = y_true.float()

    # Flatten to (N, -1) to compute IoU per sample independently
    pred_mask = pred_mask.view(pred_mask.shape[0], -1)
    true_mask = true_mask.view(true_mask.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (pred_mask * true_mask).sum(dim=1)
    total = pred_mask.sum(dim=1) + true_mask.sum(dim=1)
    union = total - intersection

    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()
