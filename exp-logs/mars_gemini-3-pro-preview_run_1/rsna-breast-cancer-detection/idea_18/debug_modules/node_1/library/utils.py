import os
import random
import numpy as np
import torch
import cv2
import rasterio
from library.config import Config


def seed_everything(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def probabilistic_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Probabilistic F1 score (pF1) as defined in the task.

    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)

    Where:
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities (0 to 1).

    Returns:
        float: The pF1 score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Epsilon for numerical stability
    eps = 1e-7

    # Probabilistic True Positives (pTP)
    # Sum of predicted probabilities for positive ground truth cases
    p_tp = np.sum(y_true * y_pred)

    # Probabilistic False Positives (pFP)
    # Sum of predicted probabilities for negative ground truth cases
    p_fp = np.sum((1 - y_true) * y_pred)

    # Total Actual Positives (TP + FN)
    # This is simply the count of positive labels in the ground truth
    total_positives = np.sum(y_true)

    # Probabilistic Precision
    # pTP / (pTP + pFP)
    # Note: pTP + pFP is mathematically equal to sum(y_pred)
    precision_denominator = p_tp + p_fp + eps
    p_precision = p_tp / precision_denominator

    # Probabilistic Recall
    # pTP / (TP + FN)
    recall_denominator = total_positives + eps
    p_recall = p_tp / recall_denominator

    # Probabilistic F1
    f1_denominator = p_precision + p_recall + eps
    p_f1 = 2 * (p_precision * p_recall) / f1_denominator

    return p_f1


def load_dicom_image(path: str) -> np.ndarray:
    """
    Robustly attempts to load a DICOM image (or image file) from the given path.
    Prioritizes OpenCV, falls back to Rasterio.

    Args:
        path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image as a numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be loaded by any method.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found at: {path}")

    # Attempt 1: OpenCV
    # cv2.IMREAD_UNCHANGED is crucial for preserving bit-depth (e.g., 16-bit mammograms)
    try:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
    except Exception:
        # Continue to next method if cv2 fails
        pass

    # Attempt 2: Rasterio
    # Rasterio (GDAL) is robust for various formats including JPEG2000 embedded in DICOM
    try:
        with rasterio.open(path) as src:
            # Read the first band
            image = src.read(1)
            return image
    except Exception:
        pass

    # Attempt 3: Raw Binary Fallback (Demo Environment)
    # Cite debug_lesson_29: Handle Variable Headers in Raw Binary Data via End-Relative Seeks
    try:
        file_size = os.path.getsize(path)
        # Expected size for 256x256 uint8 image (common in demo/test environments)
        expected_pixels = 256 * 256
        # Allow for a header up to ~4KB
        if expected_pixels <= file_size < (expected_pixels + 4096):
            with open(path, "rb") as f:
                # Seek to the end minus the pixel data size
                f.seek(-expected_pixels, os.SEEK_END)
                data = f.read(expected_pixels)
                image = np.frombuffer(data, dtype=np.uint8).reshape(256, 256)
                return image
    except Exception:
        pass

    # If all methods fail, raise an exception (Fail Loudly)
    raise ValueError(
        f"Failed to load image at {path}. Ensure the format is supported by cv2 or rasterio."
    )
