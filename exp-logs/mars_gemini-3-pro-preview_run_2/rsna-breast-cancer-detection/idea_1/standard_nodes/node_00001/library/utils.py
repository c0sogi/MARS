import os
import cv2
import numpy as np
import torch
from library.config import IMG_SIZE


def read_dicom_from_bytes(file_path: str) -> np.ndarray:
    """
    Reads a DICOM file by scanning for image signatures and decoding the payload
    using OpenCV, bypassing the need for pydicom. Handles JPEG and JPEG 2000.

    Args:
        file_path (str): Path to the .dcm file.

    Returns:
        np.ndarray: The decoded image in RGB format resized to IMG_SIZE.
                    Returns a black image if decoding fails.
    """
    if not os.path.exists(file_path):
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    try:
        with open(file_path, "rb") as f:
            data = f.read()

        # Signatures based on task description and standard formats
        # JPEG 2000 signature box: 00 00 00 0C 6A 50
        jp2_sig = b"\x00\x00\x00\x0c\x6a\x50"
        # JPEG SOI: FF D8
        jpeg_sig = b"\xff\xd8"

        offset = -1

        # Attempt to find JPEG 2000 signature first (more specific)
        idx_jp2 = data.find(jp2_sig)

        # Attempt to find JPEG signature
        idx_jpeg = data.find(jpeg_sig)

        # Logic to pick the valid start index
        if idx_jp2 != -1:
            offset = idx_jp2
        elif idx_jpeg != -1:
            offset = idx_jpeg

        if offset == -1:
            # No recognizable image header found
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Extract payload
        img_data = data[offset:]

        # Decode from memory
        # cv2.imdecode returns BGR
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target dimension
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        return img

    except Exception:
        # Fallback for any IO or processing errors
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def probabilistic_f1(
    y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-7
) -> torch.Tensor:
    """
    Calculates the Probabilistic F1 score (pF1).

    Args:
        y_true (torch.Tensor): Ground truth binary labels (0 or 1).
        y_pred (torch.Tensor): Predicted probabilities (0 to 1).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: The scalar pF1 score.
    """
    # Ensure inputs are float for calculation
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Probabilistic True Positives: sum(y_true * y_pred)
    p_tp = (y_true * y_pred).sum()

    # Probabilistic False Positives: sum((1 - y_true) * y_pred)
    p_fp = ((1 - y_true) * y_pred).sum()

    # True Positives + False Negatives is simply the count of actual positives
    total_positives = y_true.sum()

    # Probabilistic Precision
    p_precision = p_tp / (p_tp + p_fp + epsilon)

    # Probabilistic Recall (Denominator is standard TP + FN)
    p_recall = p_tp / (total_positives + epsilon)

    # Probabilistic F1
    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1
