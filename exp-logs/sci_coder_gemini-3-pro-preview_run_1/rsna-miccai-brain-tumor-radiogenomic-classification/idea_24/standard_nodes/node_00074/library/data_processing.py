import os
import glob
import numpy as np
import pandas as pd
import cv2
from library.config import Config


def read_dicom(path, image_size=Config.IMAGE_SIZE):
    """
    Reads a DICOM file using OpenCV, converts to float32, and resizes.
    Returns raw pixel values (no normalization applied here).

    Args:
        path (str): Path to the DICOM file.
        image_size (int): Target spatial dimension (H=W).

    Returns:
        np.ndarray: Float32 image array of shape (image_size, image_size).
    """
    # Attempt to read with OpenCV (IMREAD_UNCHANGED preserves depth if possible)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        # Return black image if read fails
        return np.zeros((image_size, image_size), dtype=np.float32)

    img = img.astype(np.float32)

    # Resize if dimensions differ from target
    if img.shape[:2] != (image_size, image_size):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    return img


def compute_centroid(file_paths):
    """
    Computes the Z-axis center of mass (index) for a modality based on brain tissue presence.
    Uses a threshold > 0 to define the ROI, as per the CA-WIV strategy.

    Args:
        file_paths (list): List of file paths for the modality.

    Returns:
        int: The index of the centroid slice.
    """
    if not file_paths:
        return 0

    # Sort files by instance number (filename structure: Image-X.dcm)
    try:
        sorted_files = sorted(
            file_paths, key=lambda x: int(x.split("-")[-1].split(".")[0])
        )
    except:
        # Fallback sorting
        sorted_files = sorted(file_paths)

    total_mass = 0.0
    weighted_sum = 0.0

    # Optimization: Sample every 2nd slice to balance speed and anatomical precision.
    # We avoid checking every single slice to keep runtime low, but check frequently enough
    # to capture the brain shape accurately.
    step = 2

    for i in range(0, len(sorted_files), step):
        path = sorted_files[i]
        # Read raw image (no resize needed for mass calc)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            # Threshold > 0 to define Brain ROI
            # Mass is defined as the count of non-zero pixels (volume of tissue in slice)
            mass = np.count_nonzero(img > 0)

            # Filter background noise (empty or near-empty slices)
            if mass > 50:
                weighted_sum += i * mass
                total_mass += mass

    if total_mass > 0:
        centroid = int(weighted_sum / total_mass)
    else:
        # Fallback: Middle slice if no tissue detected
        centroid = len(sorted_files) // 2

    # Ensure centroid is within valid bounds
    centroid = max(0, min(centroid, len(sorted_files) - 1))

    return centroid


def load_volumetric_slab(
    file_paths, centroid_idx, delta=Config.STRIDE, image_size=Config.IMAGE_SIZE
):
    """
    Extracts a 3-channel slab (center-delta, center, center+delta) for a modality.
    Applies Independent Channel Min-Max scaling to [0, 1].

    Args:
        file_paths (list): List of file paths.
        centroid_idx (int): The index of the center slice.
        delta (int): The stride for the slab.
        image_size (int): Spatial dimension.

    Returns:
        np.ndarray: Stacked array of shape (image_size, image_size, 3).
    """
    if not file_paths:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

    # Sort files
    try:
        sorted_files = sorted(
            file_paths, key=lambda x: int(x.split("-")[-1].split(".")[0])
        )
    except:
        sorted_files = sorted(file_paths)

    num_files = len(sorted_files)

    # Define indices: z-delta, z, z+delta
    indices = [centroid_idx - delta, centroid_idx, centroid_idx + delta]

    # Clip indices to valid file range
    indices = [max(0, min(i, num_files - 1)) for i in indices]

    slices = []
    for i in indices:
        path = sorted_files[i]
        img = read_dicom(path, image_size)

        # Independent Channel Min-Max Scaling
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            # Scale to [0, 1]
            img = (img - img_min) / (img_max - img_min)
        else:
            # Handle flat/empty images
            img = np.zeros_like(img)

        slices.append(img)

    # Stack along last dimension -> (H, W, 3)
    slab = np.stack(slices, axis=-1)

    return slab


def get_centroids_with_caching(
    df, input_dir, cache_name="centroids", load_cached_data=True
):
    """
    Computes or loads cached centroids for the given dataframe.
    Ensures deterministic processing and saves results to parquet.

    Args:
        df (pd.DataFrame): Metadata dataframe containing BraTS21ID and paths.
        input_dir (str): Base input directory.
        cache_name (str): Identifier for the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe with BraTS21ID and {modality}_centroid columns.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute Centroids
    results = []
    modalities = Config.MODALITIES

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        res = {"BraTS21ID": sid}

        for mod in modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(input_dir, rel_path)

            if os.path.exists(full_path):
                files = glob.glob(os.path.join(full_path, "*.dcm"))
                centroid = compute_centroid(files)
            else:
                centroid = 0

            res[f"{mod}_centroid"] = centroid

        results.append(res)

    df_res = pd.DataFrame(results)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_res.to_parquet(cache_path)

    return df_res
