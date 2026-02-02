import os
import random
import numpy as np
import torch
import cv2
import pydicom
from library.config import Config


def seed_everything(seed=Config.SEED):
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


def probabilistic_f1(y_true, y_pred, epsilon=1e-7):
    """
    Calculates the Probabilistic F1 score (pF1).

    Formula:
        pF1 = 2 * (pPrecision * pRecall) / (pPrecision + pRecall)
        pPrecision = pTP / (pTP + pFP)
        pRecall = pTP / (TP + FN)

    Args:
        y_true: Array-like of ground truth labels (0 or 1).
        y_pred: Array-like of predicted probabilities [0, 1].
        epsilon: Small constant to prevent division by zero.

    Returns:
        float: The pF1 score.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    # pTP = Sum(y_true * y_pred)
    p_tp = np.sum(y_true * y_pred)

    # pFP = Sum((1 - y_true) * y_pred)
    # Note: pTP + pFP = Sum(y_pred)
    precision_denom = np.sum(y_pred)

    # TP + FN = Sum(y_true) (Total actual positives)
    recall_denom = np.sum(y_true)

    p_precision = p_tp / (precision_denom + epsilon)
    p_recall = p_tp / (recall_denom + epsilon)

    pf1 = 2 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    return pf1


def load_dicom_and_process(
    file_path,
    img_size=Config.IMG_SIZE,
    norm_min_percentile=Config.NORM_MIN_PERCENTILE,
    norm_max_percentile=Config.NORM_MAX_PERCENTILE,
):
    """
    Loads a DICOM file, applies percentile windowing, normalizes to [0, 1], and resizes.

    Args:
        file_path: Relative path to the DICOM file (e.g., 'train_images/pid/iid.dcm').
        img_size: Tuple (H, W) for resizing.
        norm_min_percentile: Lower percentile for clipping (e.g., 1).
        norm_max_percentile: Upper percentile for clipping (e.g., 99).

    Returns:
        np.ndarray: Processed image array of shape (H, W, 1) with values in [0, 1].
                    Returns a zero array if loading fails.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    # Return blank image if file does not exist
    if not os.path.exists(full_path):
        return np.zeros((img_size[0], img_size[1], 1), dtype=np.float32)

    try:
        dicom = pydicom.dcmread(full_path)
        img = dicom.pixel_array

        # Handle MONOCHROME1 (where 0 is white/dense).
        # Invert so that high values always represent dense tissue/cancer.
        if (
            hasattr(dicom, "PhotometricInterpretation")
            and dicom.PhotometricInterpretation == "MONOCHROME1"
        ):
            img = np.max(img) - img

        img = img.astype(np.float32)

        # Percentile Windowing
        # Clip outliers to stabilize contrast
        if norm_min_percentile is not None and norm_max_percentile is not None:
            p_min = np.percentile(img, norm_min_percentile)
            p_max = np.percentile(img, norm_max_percentile)

            if p_max > p_min:
                img = np.clip(img, p_min, p_max)
                # Normalize to [0, 1]
                img = (img - p_min) / (p_max - p_min)
            else:
                # If image is constant, return zeros
                img = np.zeros_like(img)

        # Resize
        if img_size is not None:
            # cv2.resize expects (width, height)
            img = cv2.resize(img, (img_size[1], img_size[0]))

        # Add channel dimension if missing
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        return img

    except Exception:
        # Fallback for corrupt files or decoding errors
        return np.zeros((img_size[0], img_size[1], 1), dtype=np.float32)
