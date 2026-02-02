import os
import random
import numpy as np
import torch
import cv2


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def probabilistic_f1(y_true, y_pred, beta=1):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP) = Sum(y_true * y_pred) / Sum(y_pred)
        pRecall = pTP / (TP + FN) = Sum(y_true * y_pred) / Sum(y_true)

    Args:
        y_true: Ground truth labels (binary). Numpy array or Torch tensor.
        y_pred: Predicted probabilities (0-1). Numpy array or Torch tensor.
        beta: Beta value for F-score (default 1 for F1).

    Returns:
        float: The pF1 score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays to 1D
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Clip probabilities to valid range [0, 1]
    y_pred = np.clip(y_pred, 0, 1)

    # Calculate Probabilistic True Positives (pTP)
    # pTP = Sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # Calculate denominators
    # Sum(y_pred) represents (pTP + pFP)
    sum_pred = np.sum(y_pred)

    # Sum(y_true) represents (TP + FN), i.e., total actual positives
    total_positives = np.sum(y_true)

    # Calculate pPrecision
    # Handle division by zero if model predicts 0 probability for everything
    p_precision = p_tp / sum_pred if sum_pred > 1e-7 else 0.0

    # Calculate pRecall
    # Handle division by zero if there are no positive samples in ground truth
    p_recall = p_tp / total_positives if total_positives > 1e-7 else 0.0

    # Calculate pF1
    if (p_precision + p_recall) < 1e-7:
        return 0.0

    f1 = (1 + beta**2) * (p_precision * p_recall) / ((beta**2 * p_precision) + p_recall)

    return f1


def load_image_robust(path):
    """
    Loads an image from the given path using OpenCV.
    Strictly adheres to 'Fail Loudly': Raises exceptions if file is missing or corrupt.

    Args:
        path (str): Full or relative path to the image file.

    Returns:
        np.ndarray: The loaded image.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file exists but cannot be decoded by OpenCV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    # 1. Attempt Standard Loading
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # 2. Fallback: Decode from Raw Bytes
    # Cite debug_lesson_8: Implement Cascading Fallbacks for Robust DICOM Ingestion.
    if img is None:
        try:
            with open(path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        raise IOError(
            f"OpenCV failed to decode image at {path}. File may be corrupt or format unsupported."
        )

    return img
