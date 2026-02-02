import os
import random
import numpy as np
import torch
import cv2


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true (np.array or torch.Tensor): Ground truth binary labels (0 or 1).
        y_pred (np.array or torch.Tensor): Predicted probabilities for the positive class.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: The pF1 score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Calculate pTP (probabilistic True Positives)
    # Sum of predicted probabilities for actual positive cases
    p_tp = np.sum(y_true * y_pred)

    # Calculate pFP (probabilistic False Positives)
    # Sum of predicted probabilities for actual negative cases
    p_fp = np.sum((1.0 - y_true) * y_pred)

    # TP + FN (Total actual positives)
    total_positives = np.sum(y_true)

    # pPrecision = pTP / (pTP + pFP)
    # Note: pTP + pFP = sum(y_pred)
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # pRecall = pTP / (TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return float(pf1)


def load_image(path):
    """
    Loads an image from the specified path using OpenCV.
    Fails loudly if the file does not exist or cannot be loaded.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be loaded/decoded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    # Attempt to load the image
    # cv2.IMREAD_UNCHANGED loads the image as is (including alpha channel if present, or 16-bit depth)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(
            f"Failed to decode image at: {path}. The file might be corrupt or format unsupported."
        )

    return img
