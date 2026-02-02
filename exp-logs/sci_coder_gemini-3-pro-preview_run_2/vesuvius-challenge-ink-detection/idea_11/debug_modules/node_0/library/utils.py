import os
import cv2
import numpy as np
import torch
import random
from library.config import Config


def seed_everything(seed: int):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encoding(x):
    """
    Run-Length Encoding for the submission.
    Args:
        x: numpy array of shape (height, width), binary (0/1).
    Returns:
        str: Space-delimited list of pairs (start_index, length).
             Pixels are 1-indexed, numbered left-to-right, top-to-bottom.
    """
    # Flatten the array (row-major)
    pixels = x.flatten()

    # Pad with 0 at start and end to detect all transitions
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: end_index - start_index
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def sigmoid(x):
    """
    Applies sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-Beta score (default beta=0.5).

    The F0.5 score weights precision higher than recall.
    Formula: ((1 + beta^2) * TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)

    Args:
        preds: torch.Tensor (logits or probabilities) or numpy array.
        targets: torch.Tensor or numpy array (binary 0/1).
        beta: float, the beta parameter for F-score.
        threshold: float, threshold to binarize predictions.
        epsilon: float, smoothing factor for numerical stability.

    Returns:
        float: The computed F-Beta score.
    """
    # Convert to tensor if numpy
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Move to CPU for calculation
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # Apply sigmoid if input appears to be logits (heuristically)
    if preds.min() < 0 or preds.max() > 1:
        preds = torch.sigmoid(preds)

    # Binarize
    y_pred = (preds > threshold).float()
    y_true = targets.float()

    # Calculate TP, FP, FN
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    # F-Beta calculation
    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    # Handle edge case: if denominator is 0
    if denominator == 0:
        # If both empty, score is 1.0, otherwise 0.0
        return 1.0 if (tp + fp + fn) == 0 else 0.0

    return (numerator / denominator).item()


def load_volume_slice(volume_dir, z_index):
    """
    Loads a specific Z-slice from the fragment's surface volume directory.

    Args:
        volume_dir: str, path to the 'surface_volume' directory.
        z_index: int, the index of the slice to load (0-64).

    Returns:
        numpy.ndarray: The image data (height, width) or None if missing.
    """
    filename = f"{z_index:02d}.tif"
    filepath = os.path.join(volume_dir, filename)

    if not os.path.exists(filepath):
        return None

    # Load image as-is (likely uint16)
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return image


def load_mask(mask_path):
    """
    Loads the binary mask for a fragment.

    Args:
        mask_path: str, path to the mask file.

    Returns:
        numpy.ndarray: Binary mask (0/1).
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Ensure binary
    return (mask > 0).astype(np.uint8)


def normalize_image(image):
    """
    Normalizes a raw image (uint16) to [0, 1] range using config constants.

    Args:
        image: numpy array.

    Returns:
        numpy array: Normalized float32 image.
    """
    image = image.astype(np.float32)
    image = (image - Config.PIXEL_MIN) / (Config.PIXEL_MAX - Config.PIXEL_MIN)
    image = np.clip(image, 0, 1)
    return image
