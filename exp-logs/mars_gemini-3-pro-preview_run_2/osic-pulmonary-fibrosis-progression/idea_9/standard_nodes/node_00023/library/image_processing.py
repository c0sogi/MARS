import os
import numpy as np
import cv2
from library.config import Config


def load_scan(path):
    """
    Reads DICOM files from a directory, sorts them by instance number,
    and converts them to a 3D numpy array of Hounsfield Units.
    """
    if not os.path.exists(path):
        return None

    # List .dcm files
    files = [f for f in os.listdir(path) if f.lower().endswith(".dcm")]
    if not files:
        return None

    # Sort by instance number (assuming filename is the instance number, e.g., '1.dcm')
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()

    slices = []
    for f in files:
        file_path = os.path.join(path, f)
        # Attempt to read with OpenCV (IMREAD_UNCHANGED to keep depth)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        slices.append(img)

    if not slices:
        return None

    # Stack slices into a 3D volume (Depth, Height, Width)
    volume = np.stack(slices)

    # Convert to Hounsfield Units (HU)
    # Heuristic: CT scans are typically shifted.
    # If loaded as uint16, Air (approx -1000 HU) is often 0.
    # Standard intercept is -1024.
    if volume.min() >= 0:
        volume = volume.astype(np.int16) - 1024
    else:
        volume = volume.astype(np.int16)

    return volume


def compute_density_histogram(volume):
    """
    Computes the lung volume and a density histogram based on clinical HU ranges.
    Returns a feature vector: [Total_Volume, Bin1_Prob, Bin2_Prob, Bin3_Prob, Bin4_Prob]
    """
    # 1. Compute Total Lung Volume
    # Count voxels within the broad lung window
    lung_mask = (volume >= Config.HU_MIN) & (volume <= Config.HU_MAX)
    total_volume = np.sum(lung_mask)

    # 2. Compute Density Histogram
    # We compute the histogram over voxels that are roughly within the body (e.g. > -1000)
    # to avoid counting the large amount of background air.
    valid_voxels = volume[volume > -1000]

    if len(valid_voxels) == 0:
        hist_norm = np.zeros(len(Config.DENSITY_BINS) - 1, dtype=np.float32)
    else:
        hist, _ = np.histogram(valid_voxels, bins=Config.DENSITY_BINS)

        # Normalize to get probabilities/proportions
        total_counts = hist.sum()
        if total_counts > 0:
            hist_norm = hist / total_counts
        else:
            hist_norm = np.zeros_like(hist, dtype=np.float32)

    # Concatenate Volume and Histogram
    return np.concatenate(([float(total_volume)], hist_norm.astype(np.float32)))


def get_stratified_slices(volume):
    """
    Selects 3 slices (Apex, Mid, Base) with the highest variance.
    Resizes and normalizes them for model input.
    """
    num_slices = volume.shape[0]
    zone_size = num_slices // Config.NUM_ZONES

    selected_slices = []

    for i in range(Config.NUM_ZONES):
        # Define zone boundaries
        start = i * zone_size
        end = (i + 1) * zone_size
        if i == Config.NUM_ZONES - 1:
            end = num_slices

        # Handle cases with very few slices
        if start >= end:
            zone_volume = volume[0:1]
        else:
            zone_volume = volume[start:end]

        # Calculate variance for each slice in the zone (across Height and Width)
        variances = np.var(zone_volume, axis=(1, 2))
        max_var_idx = np.argmax(variances)

        best_slice = zone_volume[max_var_idx]

        # Resize to target IMG_SIZE
        resized = cv2.resize(
            best_slice.astype(np.float32), (Config.IMG_SIZE, Config.IMG_SIZE)
        )

        # Normalize
        # Clip to defined HU range
        clipped = np.clip(resized, Config.HU_MIN, Config.HU_MAX)
        # Min-Max Scale to [0, 1]
        normalized = (clipped - Config.HU_MIN) / (Config.HU_MAX - Config.HU_MIN)

        selected_slices.append(normalized)

    return np.array(selected_slices, dtype=np.float32)


def process_patient(patient_id, dcm_rel_path, load_cached_data=True):
    """
    Orchestrates the processing for a single patient with caching.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_rel_path (str): Relative path to the DICOM directory (e.g., 'train/ID...').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {'slices': np.array, 'histogram': np.array}
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_file = os.path.join(Config.CACHE_DIR, f"{patient_id}.npz")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            with np.load(cache_file) as data:
                slices = data["slices"]
                histogram = data["histogram"]
                return {"slices": slices, "histogram": histogram}
        except Exception:
            # If load fails (corrupt file), proceed to recompute
            pass

    # 2. Compute Features
    full_path = os.path.join(Config.INPUT_DIR, dcm_rel_path)
    volume = load_scan(full_path)

    if volume is None:
        # Fallback for missing/unreadable scans: Zero arrays
        slices = np.zeros(
            (Config.NUM_ZONES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )
        # 1 volume scalar + 4 histogram bins = 5 features
        histogram = np.zeros(5, dtype=np.float32)
    else:
        histogram = compute_density_histogram(volume)
        slices = get_stratified_slices(volume)

    # 3. Save to Cache
    np.savez(cache_file, slices=slices, histogram=histogram)

    return {"slices": slices, "histogram": histogram}
