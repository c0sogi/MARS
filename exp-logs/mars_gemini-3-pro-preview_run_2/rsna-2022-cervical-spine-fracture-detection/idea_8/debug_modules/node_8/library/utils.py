import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
import cv2
import pandas as pd
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def window_image(img: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
    """
    Applies a specific windowing to a CT image to highlight specific structures (e.g., bone).
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def load_dicom(
    path: str,
    size: tuple = (512, 512),
    window_center: int = 400,
    window_width: int = 1800,
) -> np.ndarray:
    """
    Loads a DICOM file, applies rescale slope/intercept, windows it for bone, and resizes.

    Args:
        path: Path to the .dcm file.
        size: Tuple (height, width) for resizing.
        window_center: Center of the window (default 400 for bone).
        window_width: Width of the window (default 1800 for bone).

    Returns:
        Processed image as a numpy array (H, W) normalized to [0, 1].
    """
    try:
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array

        # Apply Rescale Slope and Intercept to get HU
        slope = getattr(dicom, "RescaleSlope", 1)
        intercept = getattr(dicom, "RescaleIntercept", 0)
        img = img * slope + intercept

        # Apply Windowing
        img = window_image(img, window_center, window_width)

        # Normalize to [0, 1]
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = (img - img_min) / (img_max - img_min)

        # Resize
        if size:
            img = cv2.resize(img, (size[1], size[0]))

        return img.astype(np.float32)

    except Exception as e:
        # Fallback for missing or corrupt files: return black image
        # print(f"Warning: Failed to load {path}: {e}")
        if size:
            return np.zeros(size, dtype=np.float32)
        return np.zeros((512, 512), dtype=np.float32)


def create_spatial_mask(
    bbox_df: pd.DataFrame, study_uid: str, slice_num: int, shape: tuple = (384, 384)
) -> np.ndarray:
    """
    Creates a binary spatial mask for a specific slice based on bounding boxes.
    Used for the Spatial Supervision task.

    Args:
        bbox_df: DataFrame containing bounding boxes.
        study_uid: Study Instance UID.
        slice_num: The slice number.
        shape: Output shape of the mask (H, W).

    Returns:
        Binary mask (H, W) where 1 indicates fracture area.
    """
    mask = np.zeros(shape, dtype=np.float32)

    # Filter boxes for this study and slice
    # Assuming bbox_df has columns: StudyInstanceUID, slice_number, x, y, width, height
    # Note: slice_number in bbox_df matches the file index

    if bbox_df is None or bbox_df.empty:
        return mask

    boxes = bbox_df[
        (bbox_df["StudyInstanceUID"] == study_uid)
        & (bbox_df["slice_number"] == slice_num)
    ]

    if boxes.empty:
        return mask

    # Original image dimensions are typically 512x512, need to scale coordinates
    # We assume original DICOMs are 512x512 for coordinate scaling unless specified otherwise
    orig_h, orig_w = 512, 512
    scale_y = shape[0] / orig_h
    scale_x = shape[1] / orig_w

    for _, row in boxes.iterrows():
        x, y, w, h = row["x"], row["y"], row["width"], row["height"]

        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        # Clip to boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(shape[1], x2)
        y2 = min(shape[0], y2)

        mask[y1:y2, x1:x2] = 1.0

    return mask


def weighted_log_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculates the weighted multi-label logarithmic loss.

    L_ij = -w_j * [y_ij * log(p_ij) + (1-y_ij) * log(1-p_ij)]

    Args:
        y_true: Ground truth tensor (batch_size, num_classes).
        y_pred: Predicted probabilities (batch_size, num_classes).
        weights: Tensor of weights for each class (num_classes,).
                 If None, defaults to competition heuristic (7 for overall, 1 for others).

    Returns:
        Scalar loss (mean over batch and classes).
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    # Default weights: C1-C7 = 1.0, patient_overall = 7.0
    if weights is None:
        # Assuming last column is patient_overall based on Config.TARGET_COLS
        num_classes = y_true.shape[1]
        w = torch.ones(num_classes, device=y_true.device)
        if num_classes == 8:
            w[-1] = 7.0  # patient_overall
        weights = w

    # Calculate binary cross entropy per element (no reduction)
    # BCE = -(y * log(p) + (1-y) * log(1-p))
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    # Apply weights
    weighted_loss = loss * weights

    # Average over all elements
    return weighted_loss.mean()


def load_or_generate_cache(
    file_name: str, generation_func, load_cached_data: bool = True, **kwargs
):
    """
    Generic caching mechanism for deterministic data processing.

    Args:
        file_name: Name of the cache file (e.g., 'processed_metadata.parquet').
        generation_func: Function to call if cache is missing. Must return the data.
        load_cached_data: Boolean flag to enable/disable loading from cache.
        kwargs: Arguments passed to generation_func.

    Returns:
        The loaded or generated data.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, file_name)

    # Determine file type
    is_parquet = file_name.endswith(".parquet")
    is_npy = file_name.endswith(".npy")

    # 1. Try to load
    if load_cached_data and os.path.exists(file_path):
        try:
            if is_parquet:
                return pd.read_parquet(file_path)
            elif is_npy:
                return np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load cache {file_path}: {e}. Regenerating...")

    # 2. Generate
    data = generation_func(**kwargs)

    # 3. Save
    try:
        if is_parquet:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, index=False)
        elif is_npy:
            if isinstance(data, np.ndarray):
                np.save(file_path, data)
    except Exception as e:
        print(f"Warning: Failed to save cache {file_path}: {e}")

    return data


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
