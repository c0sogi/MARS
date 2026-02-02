import os
import glob
import random
import numpy as np
import torch
import pydicom
import cv2
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dicom(path: str, size: int = 256):
    """
    Reads a DICOM file, applies bone windowing, resizes, and returns a uint8 image.

    Args:
        path (str): Path to the .dcm file.
        size (int): Target size for resizing (square).

    Returns:
        np.ndarray: Preprocessed image of shape (size, size) with dtype uint8.
    """
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array.astype(np.float32)

        # Handle Photometric Interpretation (Invert if MONOCHROME1)
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = np.max(img) - img

        # Apply Rescale Slope and Intercept to get Hounsfield Units
        intercept = getattr(dicom, "RescaleIntercept", 0)
        slope = getattr(dicom, "RescaleSlope", 1)
        img = img * slope + intercept

        # Apply Bone Windowing
        # Standard Bone Window: Center (Level) = 400, Width = 1800
        center = 400
        width = 1800
        low = center - width / 2
        high = center + width / 2

        img = np.clip(img, low, high)

        # Normalize to [0, 255]
        img = (img - low) / (high - low)
        img = (img * 255.0).astype(np.uint8)

        # Resize
        if size is not None:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

        return img

    except Exception as e:
        # Return black image on failure to prevent pipeline crash
        if size is not None:
            return np.zeros((size, size), dtype=np.uint8)
        return np.zeros((512, 512), dtype=np.uint8)


def load_case_data(study_id: str, image_folder: str, load_cached_data: bool = True):
    """
    Loads and preprocesses a full case (study) into a 2.5D sequence.
    Implements caching mechanism to store preprocessed volumes as .npy files.

    Args:
        study_id (str): The StudyInstanceUID.
        image_folder (str): Path to the folder containing DICOM files for this study.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: 2.5D volume stack of shape (SEQ_LEN, IMAGE_SIZE, IMAGE_SIZE, 3).
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cached_volume = np.load(cache_path)
            # Verify shape matches current Config to prevent stale cache issues
            expected_shape = (Config.SEQ_LEN, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)
            if cached_volume.shape == expected_shape:
                return cached_volume
        except Exception:
            # If load fails or shape mismatch, proceed to recompute
            pass

    # 2. Compute from scratch
    if not os.path.exists(image_folder):
        # Return zeros if folder missing
        return np.zeros(
            (Config.SEQ_LEN, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
        )

    # Get all DICOM files
    files = glob.glob(os.path.join(image_folder, "*.dcm"))

    # Sort files numerically by slice number (filename)
    # Assumes filenames are like '1.dcm', '10.dcm'
    try:
        files = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))
    except ValueError:
        # Fallback for non-integer filenames
        files = sorted(files)

    num_files = len(files)
    if num_files == 0:
        return np.zeros(
            (Config.SEQ_LEN, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
        )

    # Uniform sampling of indices
    indices = np.linspace(0, num_files - 1, Config.SEQ_LEN).astype(int)

    volume = []

    for idx in indices:
        # 2.5D Stacking: (z-1, z, z+1)
        # Handle boundary conditions by clamping
        idx_prev = max(0, idx - 1)
        idx_curr = idx
        idx_next = min(num_files - 1, idx + 1)

        # Load slices
        img_prev = load_dicom(files[idx_prev], size=Config.IMAGE_SIZE)
        img_curr = load_dicom(files[idx_curr], size=Config.IMAGE_SIZE)
        img_next = load_dicom(files[idx_next], size=Config.IMAGE_SIZE)

        # Stack channels -> (H, W, 3)
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)
        volume.append(img_stack)

    # Convert to array -> (SEQ_LEN, H, W, 3)
    volume_array = np.array(volume, dtype=np.uint8)

    # 3. Save to cache
    np.save(cache_path, volume_array)

    return volume_array


def weighted_loss_metric(y_pred, y_true):
    """
    Calculates the weighted multi-label logarithmic loss.

    Args:
        y_pred: (N, 8) array-like of predicted probabilities.
        y_true: (N, 8) array-like of ground truth labels (0 or 1).

    Column Order Assumption:
        [C1, C2, C3, C4, C5, C6, C7, patient_overall]

    Weights:
        C1-C7: 1.0
        patient_overall: 7.0

    Returns:
        float: The weighted average log loss.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Define weights
    # Patient overall is weighted 7x higher than individual vertebrae
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    total_weight = np.sum(weights)

    loss_sum = 0.0

    # Calculate log loss for each column and aggregate
    for i in range(8):
        y_p = y_pred[:, i]
        y_t = y_true[:, i]

        # Binary Cross Entropy
        bce = -(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))

        # Average over the batch
        col_loss = np.mean(bce)

        loss_sum += col_loss * weights[i]

    # Return weighted average
    return loss_sum / total_weight
