import os
import random
import numpy as np
import torch
import cv2
from library import config


def seed_everything(seed=config.SEED):
    """
    Sets the random seed for reproducibility across standard libraries and torch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pf1_score(labels, preds):
    """
    Computes the Probabilistic F1 score (pF1) as defined in the task.

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP) = pTP / sum(preds)
        pRecall = pTP / (TP + FN) = pTP / sum(labels)
        pTP = sum(labels * preds)

    Args:
        labels (array-like): Binary ground truth labels (0 or 1).
        preds (array-like): Predicted probabilities (0.0 to 1.0).

    Returns:
        float: The probabilistic F1 score.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculate Probabilistic True Positives (pTP)
    # Contribution is pred_prob where label is 1
    p_tp = np.sum(labels * preds)

    # Calculate Probabilistic Precision
    # Denominator is pTP + pFP, which equals the sum of all predicted probabilities
    precision_denom = np.sum(preds)
    if precision_denom > 0:
        p_precision = p_tp / precision_denom
    else:
        p_precision = 0.0

    # Calculate Probabilistic Recall
    # Denominator is TP + FN, which equals the sum of all positive labels (Total Positives)
    recall_denom = np.sum(labels)
    if recall_denom > 0:
        p_recall = p_tp / recall_denom
    else:
        p_recall = 0.0

    # Calculate Probabilistic F1
    if (p_precision + p_recall) > 0:
        pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall)
    else:
        pf1 = 0.0

    return pf1


def read_image_from_bytes(file_path):
    """
    Reads an image directly from bytes using OpenCV, bypassing pydicom.
    This is effective for DICOMs that wrap standard image streams (e.g., JPEG2000).

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.ndarray: The image array, or None if reading fails.
    """
    if not os.path.exists(file_path):
        return None

    try:
        # Read file as a byte stream
        with open(file_path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

        # Decode using OpenCV
        # IMREAD_UNCHANGED is used to preserve bit-depth (e.g., 16-bit mammograms)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        return img
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return None
