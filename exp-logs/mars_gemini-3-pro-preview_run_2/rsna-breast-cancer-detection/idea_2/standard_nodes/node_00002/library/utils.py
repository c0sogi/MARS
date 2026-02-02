import os
import cv2
import numpy as np
import torch
import random
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to SEED from config.
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


def read_dicom_manual(file_path):
    """
    Reads a DICOM file by scanning for JPEG/JPEG2000 headers and decoding the payload.
    This bypasses the need for pydicom or gdcm/pylibjpeg when data is encapsulated.

    Args:
        file_path (str): Path to the .dcm file.

    Returns:
        np.ndarray: The decoded image array (grayscale), or None if failure.
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # Strategy 1: JPEG 2000 (Start of Codestream: FF 4F FF 51)
        # This is common in mammography datasets (e.g. RSNA/VinDr)
        j2k_start = content.find(b"\xff\x4f\xff\x51")
        if j2k_start != -1:
            # Attempt to decode from the found header
            img_array = np.frombuffer(content[j2k_start:], np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img

        # Strategy 2: Standard JPEG (Start of Image: FF D8)
        # DICOM files might contain thumbnails (small) and the main image (large).
        # We search for all FF D8 markers and return the largest valid image.
        jpeg_starts = []
        start_idx = 0
        while True:
            idx = content.find(b"\xff\xd8", start_idx)
            if idx == -1:
                break
            jpeg_starts.append(idx)
            start_idx = idx + 2

        best_img = None
        max_pixels = 0

        for start in jpeg_starts:
            img_array = np.frombuffer(content[start:], np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixels = img.shape[0] * img.shape[1]
                # Filter out likely thumbnails (e.g. < 64x64)
                if pixels > max_pixels and pixels > 4096:
                    max_pixels = pixels
                    best_img = img

        if best_img is not None:
            return best_img

        return None

    except Exception:
        # Return None on any read/decode error to allow the pipeline to handle it (e.g. blank image)
        return None


def probabilistic_f1(y_true, y_pred_probs):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
    pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
    pPrecision = pTP / (pTP + pFP)
    pRecall = pTP / (TP + FN)

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred_probs (array-like): Predicted probabilities [0, 1].

    Returns:
        float: The pF1 score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_probs)

    # Avoid division by zero
    epsilon = 1e-7

    # Probabilistic True Positives: Sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # Probabilistic False Positives: Sum((1 - y_true) * y_pred)
    # Note: Denominator for Precision is pTP + pFP = Sum(y_pred)
    predicted_positives = np.sum(y_pred)

    # Probabilistic False Negatives: Sum(y_true * (1 - y_pred))
    # Note: Denominator for Recall is TP + FN = Sum(y_true) (Total actual positives)
    actual_positives = np.sum(y_true)

    p_precision = p_tp / (predicted_positives + epsilon)
    p_recall = p_tp / (actual_positives + epsilon)

    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1
