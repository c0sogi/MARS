import os
import sys
import random
import logging
import numpy as np
import torch
import cv2


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_file, level=logging.INFO):
    """
    Sets up a logger that writes to both console and a file.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not set up file logging to {log_file}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def probabilistic_f1(y_true, y_pred_probs, beta=1, eps=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred_probs: Array-like of predicted probabilities [0, 1].
        beta: Weight of recall in the harmonic mean.
        eps: Small constant to avoid division by zero.

    Returns:
        float: The pF1 score.
    """
    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)

    # Probabilistic True Positives (pTP)
    # The expected number of true positives: sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred_probs)

    # Probabilistic Precision
    # pPrecision = pTP / (pTP + pFP)
    # The denominator is the total expected number of positives predicted: sum(y_pred)
    p_precision = p_tp / (np.sum(y_pred_probs) + eps)

    # Probabilistic Recall
    # pRecall = pTP / (TP + FN)
    # The denominator is the total number of actual positives: sum(y_true)
    p_recall = p_tp / (np.sum(y_true) + eps)

    # F1 Score
    f1 = (
        (1 + beta**2)
        * (p_precision * p_recall)
        / ((beta**2 * p_precision) + p_recall + eps)
    )

    return f1


def load_image(path, size=None):
    """
    Loads an image from a file path using OpenCV, ensuring grayscale.

    Args:
        path (str): Path to the image file.
        size (tuple, optional): Target size (width, height) for resizing.

    Returns:
        np.ndarray: The loaded grayscale image.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the image cannot be loaded (corrupt or unsupported format).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    # Attempt to load the image
    # IMREAD_UNCHANGED is used to preserve bit-depth (e.g., 16-bit mammograms)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        # Fallback for raw binary (DICOM/Raw) if cv2 fails
        try:
            file_size = os.path.getsize(path)
            # Check for 256x256 (approx 65KB)
            if 65536 <= file_size < 70000:
                with open(path, "rb") as f:
                    f.seek(-65536, os.SEEK_END)
                    buf = f.read(65536)
                    img = np.frombuffer(buf, dtype=np.uint8).reshape(256, 256)
            # Check for 768x768 (approx 590KB)
            elif 589824 <= file_size < 600000:
                with open(path, "rb") as f:
                    f.seek(-589824, os.SEEK_END)
                    buf = f.read(589824)
                    img = np.frombuffer(buf, dtype=np.uint8).reshape(768, 768)
        except Exception:
            pass

    if img is None:
        # Fail loudly as per requirements
        raise ValueError(
            f"Failed to load image: {path}. File may be corrupt or format unsupported."
        )

    # Check dimensions and convert to grayscale if necessary
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if requested
    if size is not None:
        # cv2.resize expects (width, height)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    return img
