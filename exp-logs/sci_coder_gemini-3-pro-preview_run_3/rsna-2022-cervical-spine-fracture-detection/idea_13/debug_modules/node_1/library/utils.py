import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import random
import albumentations as A
from sklearn.metrics import log_loss
from library.config import Config


def seed_everything(seed=42):
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


def load_dicom_volume(study_id, image_dir):
    """
    Reads all DICOM files for a given study, sorts them by Z-position,
    and converts the pixel array to Hounsfield Units (HU).

    Args:
        study_id (str): The StudyInstanceUID.
        image_dir (str): The root directory containing study folders.

    Returns:
        np.ndarray: A 3D numpy array of shape (Depth, Height, Width) containing float32 HU values.
                    Returns an empty array if loading fails.
    """
    # Construct the full path to the study directory
    study_path = os.path.join(image_dir, study_id)

    # Check if the directory exists
    if not os.path.exists(study_path):
        return np.zeros((0, 512, 512), dtype=np.float32)

    # List all DICOM files
    dicom_files = glob.glob(os.path.join(study_path, "*.dcm"))
    if not dicom_files:
        return np.zeros((0, 512, 512), dtype=np.float32)

    # Read files
    slices = []
    for f in dicom_files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return np.zeros((0, 512, 512), dtype=np.float32)

    # Sort slices by Z-position (ImagePositionPatient[2])
    # Fallback to InstanceNumber if spatial coordinates are missing
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Extract pixel data and convert to HU
    images = []
    for s in slices:
        try:
            # Get pixel array
            img = s.pixel_array.astype(np.float32)

            # Apply Rescale Slope and Intercept to convert to HU
            slope = getattr(s, "RescaleSlope", 1.0)
            intercept = getattr(s, "RescaleIntercept", 0.0)

            if slope != 1.0:
                img = img * slope
            img = img + intercept

            images.append(img)
        except Exception:
            continue

    if not images:
        return np.zeros((0, 512, 512), dtype=np.float32)

    # Stack into a 3D volume (Depth, Height, Width)
    volume = np.stack(images)
    return volume


def window_image(volume, window_center, window_width):
    """
    Applies standard windowing to the volume to highlight specific tissues.
    Normalizes the output to the [0, 1] range.

    Args:
        volume (np.ndarray): Input volume in Hounsfield Units.
        window_center (float): The center of the window (e.g., 400 for bone).
        window_width (float): The width of the window (e.g., 1800 for bone).

    Returns:
        np.ndarray: Windowed volume normalized between 0 and 1.
    """
    min_value = window_center - window_width / 2.0
    max_value = window_center + window_width / 2.0

    # Clip values to the window range
    volume = np.clip(volume, min_value, max_value)

    # Normalize to [0, 1]
    volume = (volume - min_value) / window_width

    return volume


def save_cache(study_id, data):
    """
    Saves processed data to the cache directory as a .npy file.

    Args:
        study_id (str): The unique identifier for the study.
        data (np.ndarray): The data to cache.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")
    np.save(file_path, data)


def load_cache(study_id):
    """
    Attempts to load processed data from the cache directory.

    Args:
        study_id (str): The unique identifier for the study.

    Returns:
        np.ndarray or None: The loaded data if it exists, otherwise None.
    """
    file_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")
    if os.path.exists(file_path):
        try:
            return np.load(file_path)
        except Exception:
            return None
    return None


def competition_metric(y_true, y_pred):
    """
    Calculates the weighted multi-label logarithmic loss.

    Weights:
        - patient_overall: 1.0
        - C1-C7: 1/7 each

    Args:
        y_true (array-like): Ground truth labels (N, 8).
        y_pred (array-like): Predicted probabilities (N, 8).

    Returns:
        float: The weighted average log loss.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # Ensure inputs are float
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # Clip predictions to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Define weights corresponding to columns [C1, C2, C3, C4, C5, C6, C7, patient_overall]
    # The 'patient_overall' is weighted more highly (1.0) compared to individual vertebrae (1/7).
    weights = np.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1.0])

    total_weighted_loss = 0.0

    # Calculate Log Loss for each column independently
    for i in range(8):
        # Binary Cross Entropy
        bce = -(
            y_true[:, i] * np.log(y_pred[:, i])
            + (1 - y_true[:, i]) * np.log(1 - y_pred[:, i])
        )
        mean_bce = np.mean(bce)
        total_weighted_loss += weights[i] * mean_bce

    # Normalize by the sum of weights
    return total_weighted_loss / np.sum(weights)


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train' for augmentation, 'valid'/'test' for resizing only.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                # Volumetric-consistent affine transform (Shift, Scale, Rotate)
                # Fused into one operation to minimize interpolation artifacts
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Resize to model input size
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            ]
        )
    else:
        return A.Compose(
            [
                # Only resize for validation/inference
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            ]
        )
