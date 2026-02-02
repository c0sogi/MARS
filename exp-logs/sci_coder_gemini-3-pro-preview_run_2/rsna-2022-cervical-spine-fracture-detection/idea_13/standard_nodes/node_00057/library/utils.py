import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
from library.config import Config


def load_dicom_slice(path, size=Config.IMAGE_SIZE):
    """
    Reads a DICOM file, applies bone windowing, and resizes the image.

    Args:
        path (str): Path to the DICOM file.
        size (tuple): Target size (height, width).

    Returns:
        np.ndarray: Preprocessed image array normalized to [0, 1].
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(float)

        # Apply Rescale Slope and Intercept if present
        slope = getattr(dcm, "RescaleSlope", 1.0)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        img = img * slope + intercept

        # Bone Windowing
        # Center (WL) = 400, Width (WW) = 1800
        center = 400
        width = 1800
        low = center - width / 2
        high = center + width / 2

        # Clip and Normalize
        img = np.clip(img, low, high)
        img = (img - low) / width

        # Resize
        if size:
            img = cv2.resize(img, size)

        return img

    except Exception as e:
        # Return zero array in case of error
        if size:
            return np.zeros(size)
        return np.zeros((512, 512))


def get_all_study_paths(root_dir, cache_key="train", load_cached_data=True):
    """
    Retrieves a dictionary mapping StudyInstanceUID to a sorted list of file paths.
    Implements caching logic to avoid re-scanning directories.

    Args:
        root_dir (str): Directory containing study folders (e.g., train_images).
        cache_key (str): Identifier for the cache file (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {StudyInstanceUID: [list of sorted file paths]}
    """
    cache_file = os.path.join(Config.CACHE_DIR, f"{cache_key}_paths_cache.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            df = pd.read_parquet(cache_file)
            # Convert DataFrame back to dict
            return pd.Series(df.file_paths.values, index=df.StudyInstanceUID).to_dict()
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from scratch
    study_dirs = glob.glob(os.path.join(root_dir, "*"))
    data = []

    for d in study_dirs:
        uid = os.path.basename(d)
        # Get all DICOM files
        files = glob.glob(os.path.join(d, "*.dcm"))
        # Sort by instance number (filename integer)
        # Assumes filenames are like '1.dcm', '10.dcm'
        try:
            files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
        except ValueError:
            # Fallback for non-integer filenames, though dataset spec implies integers
            files.sort()

        data.append({"StudyInstanceUID": uid, "file_paths": files})

    df = pd.DataFrame(data)

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    df.to_parquet(cache_file)

    return pd.Series(df.file_paths.values, index=df.StudyInstanceUID).to_dict()


def competition_metric(y_true, y_pred):
    """
    Calculates the Weighted Multi-Label Logarithmic Loss.

    Args:
        y_true (pd.DataFrame): DataFrame with ['row_id', 'fractured'].
        y_pred (pd.DataFrame): DataFrame with ['row_id', 'fractured'].

    Returns:
        float: The weighted log loss.
    """
    # Ensure DataFrames are merged on row_id to align predictions with targets
    # Suffixes handle potential column name collisions
    combined = y_true.merge(y_pred, on="row_id", suffixes=("_true", "_pred"))

    if combined.empty:
        return 0.0

    # Determine weights based on row_id
    # 'patient_overall' is weighted 1.0
    # 'C1' through 'C7' are weighted 1/7 (~0.142)
    # This ensures the overall label has equal weight to the sum of all vertebrae labels
    def get_weight(row_id):
        if "patient_overall" in row_id:
            return 1.0
        return 1.0 / 7.0

    weights = combined["row_id"].apply(get_weight).values

    # Extract arrays
    y_t = combined["fractured_true"].values.astype(float)
    y_p = combined["fractured_pred"].values.astype(float)

    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_p = np.clip(y_p, epsilon, 1 - epsilon)

    # Calculate Log Loss per row
    # L = - [y * log(p) + (1-y) * log(1-p)]
    log_loss = -(y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))

    # Apply weights
    weighted_log_loss = log_loss * weights

    # Average across all rows
    return np.mean(weighted_log_loss)
