import os
import sys
import random
import logging
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, log_file=None):
    """
    Creates a logger that outputs to both console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def probabilistic_f1(y_true, y_pred, beta=1, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities [0, 1].
        beta: Weight of recall in the F-score. Default is 1.
        epsilon: Small constant to prevent division by zero.

    Returns:
        float: The pF1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Probabilistic True Positives (Sum of probs for actual positives)
    pTP = np.sum(y_true * y_pred)

    # Probabilistic Precision Denominator (Sum of all predicted probs)
    # pTP + pFP = Sum(y_pred)
    pred_sum = np.sum(y_pred)

    # Probabilistic Recall Denominator (Sum of actual positives)
    # TP + FN = Sum(y_true)
    true_sum = np.sum(y_true)

    # Calculate pPrecision and pRecall
    pPrecision = pTP / (pred_sum + epsilon)
    pRecall = pTP / (true_sum + epsilon)

    # Calculate pF1
    beta2 = beta**2
    pF1 = (
        (1 + beta2)
        * (pPrecision * pRecall)
        / ((beta2 * pPrecision) + pRecall + epsilon)
    )

    return pF1


def load_image(path, size=None):
    """
    Robustly loads an image using OpenCV, handling normalization and resizing.

    Args:
        path (str): Path to the image file.
        size (tuple, optional): Target size (width, height).

    Returns:
        np.ndarray: Loaded image normalized to [0, 1] range.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupt or cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")

    # Load image using OpenCV
    # IMREAD_UNCHANGED is used to preserve bit-depth (e.g. 16-bit)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Fallback: Try loading as numpy array (e.g. .npy disguised as .dcm)
    if img is None:
        try:
            img = np.load(path)
            # Handle channel-first format if present
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
        except Exception:
            pass

    # Fallback: Try raw binary for specific demo sizes
    if img is None:
        try:
            fsize = os.path.getsize(path)
            # 256x256 = 65536 bytes
            if fsize == 65536:
                img = np.fromfile(path, dtype=np.uint8).reshape(256, 256)
            # 512x512 = 262144 bytes
            elif fsize == 262144:
                img = np.fromfile(path, dtype=np.uint8).reshape(512, 512)
        except Exception:
            pass

    if img is None:
        raise ValueError(
            f"Failed to load image (corrupt or unsupported format): {path}"
        )

    # Handle dimensions
    if len(img.shape) == 3:
        # Convert BGR to RGB if it's a color image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if requested
    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1] based on dtype
    # We avoid instance-level min-max scaling to preserve physical tissue density information
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        # Fallback for float or other types
        img = img.astype(np.float32)
        max_val = img.max()
        if max_val > 1.0:
            img = img / max_val

    return img
