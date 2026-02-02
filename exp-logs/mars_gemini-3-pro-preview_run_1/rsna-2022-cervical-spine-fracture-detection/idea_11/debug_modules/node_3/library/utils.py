import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pydicom
import cv2
import glob
from library.config import Config

# =============================================================================
# Device & Hardware
# =============================================================================


def get_device():
    """
    Returns the appropriate torch device (CUDA or CPU).
    """
    return torch.device(Config.DEVICE)


# =============================================================================
# DICOM & Image Processing
# =============================================================================


def window_image(img, window_center, window_width):
    """
    Applies a specific windowing to a DICOM image.
    Formula: (img - (center - width/2)) / width
    Then clipped to [0, 1].
    """
    # Placeholder if needed, but usually passed
    # In this function, we assume img is already rescaled to HU if slope/intercept were applied
    # or we apply the windowing logic on raw values if they are consistent.
    # For RSNA, standard windowing is applied on HU values.

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    # Normalize to [0, 1]
    # Avoid division by zero
    if window_width == 0:
        return np.zeros_like(img)

    img = (img - img_min) / window_width
    return img


# Global counter to prevent log flooding
_DICOM_ERROR_COUNT = 0
_DICOM_ERROR_LIMIT = 10


def load_dicom(
    path,
    window_center=Config.WINDOW_CENTER,
    window_width=Config.WINDOW_WIDTH,
    resize_to=None,
):
    """
    Reads a DICOM file, applies slope/intercept to get HU, applies windowing,
    and optionally resizes.

    Args:
        path (str): Path to the .dcm file.
        window_center (int): Window center (level).
        window_width (int): Window width.
        resize_to (int, optional): Size to resize the image (square).

    Returns:
        np.ndarray: Processed image (float32) in range [0, 1].
    """
    global _DICOM_ERROR_COUNT
    try:
        dicom = pydicom.dcmread(path)

        # Extract pixel array
        img = dicom.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if present to convert to Hounsfield Units (HU)
        slope = getattr(dicom, "RescaleSlope", 1.0)
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Apply Windowing
        img = window_image(img, window_center, window_width)

        # Resize if requested
        if resize_to is not None:
            img = cv2.resize(
                img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR
            )

        return img

    except Exception as e:
        # Fallback for corrupt images or read errors
        # Create a black image of appropriate size
        if _DICOM_ERROR_COUNT < _DICOM_ERROR_LIMIT:
            print(f"Error loading DICOM {path}: {e}")
            if _DICOM_ERROR_COUNT == _DICOM_ERROR_LIMIT - 1:
                print("Stopping further DICOM error logging to prevent overflow.")
            _DICOM_ERROR_COUNT += 1

        sz = resize_to if resize_to is not None else Config.IMAGE_SIZE_ORIGINAL
        return np.zeros((sz, sz), dtype=np.float32)


def load_dicom_stack(study_dir, plane="axial", resize_to=None):
    """
    Loads all DICOMs in a directory, sorts them by instance number (Z-axis),
    and returns a 3D volume.
    """
    files = glob.glob(os.path.join(study_dir, "*.dcm"))

    # Sort by Instance Number
    # We read just the header for sorting to be fast, or parse filename if reliable.
    # Parsing filename is risky. Let's read InstanceNumber.
    dicoms = []
    for f in files:
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            dicoms.append((f, int(d.InstanceNumber)))
        except:
            continue

    dicoms.sort(key=lambda x: x[1])

    volume = []
    for f_path, _ in dicoms:
        img = load_dicom(f_path, resize_to=resize_to)
        volume.append(img)

    if len(volume) == 0:
        return None

    return np.stack(volume, axis=0)  # (Depth, Height, Width)


# =============================================================================
# Metrics & Loss
# =============================================================================


class RSNALogLoss(nn.Module):
    """
    Weighted Multi-label Logarithmic Loss.
    """

    def __init__(self, use_logits=True, weights=None):
        super().__init__()
        self.use_logits = use_logits
        # Default weights: 1 for C1-C7, 7 for patient_overall (heuristic based on competition)
        if weights is None:
            # Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
            self.weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
        else:
            self.weights = torch.tensor(weights)

        self.weights = self.weights.to(Config.DEVICE)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Predictions (Logits if use_logits=True, else Probabilities). Shape (B, 8)
            y_true: Targets (0 or 1). Shape (B, 8)
        """
        # Move weights to same device as input
        if self.weights.device != y_pred.device:
            self.weights = self.weights.to(y_pred.device)

        if self.use_logits:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        else:
            # Clamp for stability
            y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")

        # Apply weights
        # loss is (B, 8), weights is (8,)
        weighted_loss = loss * self.weights

        # Average over all elements (conceptually flattening the batch and classes)
        # The prompt says "loss is averaged across all rows".
        # In submission, each class for each patient is a row.
        # So we take the mean of the weighted values.
        return weighted_loss.mean()


def get_score(y_true, y_pred):
    """
    Calculates the competition metric for validation.
    y_true: numpy array (N, 8)
    y_pred: numpy array (N, 8) - Probabilities
    """
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Clip predictions
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # Binary Cross Entropy
    # L = -w * [y * log(p) + (1-y) * log(1-p)]
    loss = -weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss.mean()


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves model state and training info.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, path)
    # print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path, optimizer=None):
    """
    Loads model state.
    """
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return None

    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


# =============================================================================
# Submission Helper
# =============================================================================


def format_submission(study_ids, predictions, output_path=Config.SUBMISSION_PATH):
    """
    Formats predictions into the competition submission CSV format.

    Args:
        study_ids (list): List of StudyInstanceUIDs.
        predictions (np.ndarray): Array of shape (N, 8) with probabilities.
                                  Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
        output_path (str): Path to save the CSV.
    """
    row_ids = []
    probs = []

    col_names = (
        Config.TARGET_COLS
    )  # ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    for i, study_uid in enumerate(study_ids):
        preds = predictions[i]
        for j, col in enumerate(col_names):
            row_id = f"{study_uid}_{col}"
            row_ids.append(row_id)
            probs.append(preds[j])

    df_sub = pd.DataFrame({"row_id": row_ids, "fractured": probs})

    df_sub.to_csv(output_path, index=False)
    # print(f"Submission saved to {output_path} with {len(df_sub)} rows.")
    return df_sub


# =============================================================================
# Caching Utility
# =============================================================================


def cache_data(filename, data=None, load_cached_data=True):
    """
    Handles caching of deterministic data (e.g., DataFrames, features).
    Uses .parquet for DataFrames and .npy for numpy arrays/other objects.

    Args:
        filename (str): Name of the file (e.g. 'train_features.npy').
        data (object): Data to save if cache is invalid/missing.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        The loaded or saved data.
    """
    cache_path = os.path.join(Config.CACHE_DIR, filename)

    # Determine type based on extension
    is_parquet = filename.endswith(".parquet")
    is_npy = filename.endswith(".npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            if is_parquet:
                return pd.read_parquet(cache_path)
            elif is_npy:
                return np.load(cache_path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # If we are here, we need to save 'data'
    if data is None:
        return None  # Nothing to save or load

    if is_parquet:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(cache_path, index=False)
    elif is_npy:
        np.save(cache_path, data)

    return data
