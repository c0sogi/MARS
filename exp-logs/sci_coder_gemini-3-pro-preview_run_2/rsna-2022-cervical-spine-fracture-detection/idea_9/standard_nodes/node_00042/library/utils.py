import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import log_loss
from library.config import Config

# Attempt to import pydicom.
# While not in the strict installed list, it is required for DICOM processing.
try:
    import pydicom
except ImportError:
    pydicom = None


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom_slice(path: str, size: int = None):
    """
    Reads a DICOM file, applies bone windowing, and normalizes to [0, 1].

    Args:
        path (str): Path to the .dcm file.
        size (int, optional): Target size (size, size) for resizing.
                              If None, returns original size.

    Returns:
        np.ndarray: Preprocessed image slice (H, W) with values in [0, 1].
    """
    if pydicom is None:
        raise ImportError(
            "pydicom is required to read DICOM files but is not installed."
        )

    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        slope = getattr(dcm, "RescaleSlope", 1.0)
        img = img * slope + intercept

        # Apply Bone Windowing
        # Standard Bone Window: Center (Level) = 500, Width = 2000
        # (Range: -500 to 1500)
        window_center = 500
        window_width = 2000

        min_value = window_center - (window_width / 2)
        max_value = window_center + (window_width / 2)

        img = np.clip(img, min_value, max_value)

        # Normalize to [0, 1]
        img = (img - min_value) / window_width

        # Handle inversion if PhotometricInterpretation is MONOCHROME1
        if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = 1.0 - img

    except Exception as e:
        # Fallback for corrupt files or missing pixel data
        # Return a black image of target size or default 512x512
        print(f"Error loading DICOM {path}: {e}")
        default_size = size if size else 512
        return np.zeros((default_size, default_size), dtype=np.float32)

    # Resize if requested
    if size is not None:
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    return img


def calculate_weighted_log_loss(y_true, y_pred):
    """
    Calculates the weighted multi-label logarithmic loss.

    Weights:
        patient_overall: 7.0
        C1 - C7: 1.0

    Args:
        y_true (np.ndarray or pd.DataFrame): Ground truth labels (N, 8).
                                             Order: C1..C7, patient_overall
        y_pred (np.ndarray or pd.DataFrame): Predicted probabilities (N, 8).

    Returns:
        float: The mean weighted log loss.
    """
    # Define weights matching the column order in Config.TARGET_COLS
    # Config.TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    # Weights: C1-C7 = 1.0, patient_overall = 7.0
    class_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])

    # Convert to numpy if DataFrame
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true[Config.TARGET_COLS].values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred[Config.TARGET_COLS].values

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate Log Loss for each class column
    # Formula: - (y * log(p) + (1-y) * log(1-p))
    loss_per_class = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Apply weights
    weighted_loss = loss_per_class * class_weights

    # The metric is averaged across all rows (and implicitly columns by the weighting scheme)
    # The competition description says "loss is averaged across all rows".
    # Assuming the weights are applied per row-label instance.
    return np.mean(weighted_loss)


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Saves the model state, optimizer state, and training metadata.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": float(loss),
    }
    torch.save(state, filename)
    # print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """
    Loads model weights and optimizer state from a checkpoint.
    """
    if not os.path.exists(filename):
        print(f"Checkpoint not found: {filename}")
        return None

    checkpoint = torch.load(filename, map_location=Config.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
