import os
import numpy as np
import pydicom
import cv2
from library.config import Config


def load_dicom_volume(dcm_path):
    """
    Loads DICOM files from a directory, sorts them by Z-position,
    and converts to Hounsfield Units (HU).

    Args:
        dcm_path (str): Path to the directory containing .dcm files.

    Returns:
        np.ndarray: 3D volume array (D, H, W) in Hounsfield Units.
    """
    if not os.path.exists(dcm_path):
        raise FileNotFoundError(f"Path not found: {dcm_path}")

    files = [f for f in os.listdir(dcm_path) if f.endswith(".dcm")]
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dcm_path}")

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dcm_path, f))
            # Sort by ImagePositionPatient[2] (Z-axis) if available, else InstanceNumber
            if hasattr(ds, "ImagePositionPatient"):
                pos = float(ds.ImagePositionPatient[2])
                slices.append((pos, ds))
            elif hasattr(ds, "InstanceNumber"):
                pos = float(ds.InstanceNumber)
                slices.append((pos, ds))
            else:
                # Fallback if no spatial info
                slices.append((0, ds))
        except Exception:
            # Skip corrupt files
            continue

    if not slices:
        raise ValueError(f"No valid DICOM slices found in {dcm_path}")

    # Sort by Z position
    slices.sort(key=lambda x: x[0])
    sorted_slices = [x[1] for x in slices]

    # Stack images and convert to HU
    images = []
    for s in sorted_slices:
        img = s.pixel_array.astype(np.float32)

        # Apply RescaleSlope and RescaleIntercept to get HU
        slope = getattr(s, "RescaleSlope", 1.0)
        intercept = getattr(s, "RescaleIntercept", 0.0)

        # Handle cases where slope/intercept might be stored as strings
        try:
            slope = float(slope)
            intercept = float(intercept)
        except:
            slope = 1.0
            intercept = 0.0

        img = img * slope + intercept
        images.append(img)

    volume = np.stack(images)  # (D, H, W)
    return volume


def get_lung_mask(volume, hu_min=Config.HU_MIN, hu_max=Config.HU_MAX):
    """
    Creates a binary mask based on HU thresholding.
    Used primarily for identifying lung regions for slice selection.

    Args:
        volume (np.ndarray): 3D HU volume.
        hu_min (int): Minimum HU threshold.
        hu_max (int): Maximum HU threshold.

    Returns:
        np.ndarray: Boolean mask.
    """
    mask = (volume >= hu_min) & (volume <= hu_max)
    return mask


def compute_density_histogram(volume):
    """
    Computes a 4-bin density histogram representing radiomic features.

    Bins:
    1. Emphysema: < -950 HU
    2. Healthy: -950 to -700 HU
    3. Fibrosis: -700 to -200 HU
    4. Consolidation: > -200 HU

    Filters out background air (< -1024 HU).

    Args:
        volume (np.ndarray): 3D HU volume.

    Returns:
        np.ndarray: Normalized histogram (4,).
    """
    # Flatten volume
    flat_vol = volume.flatten()

    # Filter background air (approx < -1000, using -1024 as safe lower bound)
    valid_pixels = flat_vol[flat_vol > -1024]

    if valid_pixels.size == 0:
        return np.zeros(4, dtype=np.float32)

    # Define bins edges: (-inf, -950), [-950, -700), [-700, -200), [-200, inf)
    bins = [-np.inf, -950, -700, -200, np.inf]

    hist, _ = np.histogram(valid_pixels, bins=bins)

    # Normalize to sum to 1
    total_pixels = hist.sum()
    if total_pixels > 0:
        hist = hist.astype(np.float32) / total_pixels
    else:
        hist = np.zeros(4, dtype=np.float32)

    return hist


def select_variance_slices(volume, mask, n_slices=Config.NUM_SLICES):
    """
    Selects indices of axial slices with the highest variance within the lung mask.

    Args:
        volume (np.ndarray): 3D HU volume.
        mask (np.ndarray): Boolean mask indicating ROI.
        n_slices (int): Number of slices to select.

    Returns:
        list: Sorted list of selected slice indices.
    """
    variances = []
    D = volume.shape[0]

    for i in range(D):
        slice_mask = mask[i]
        # Only consider slices with some lung content
        if np.sum(slice_mask) > 0:
            # Compute variance of pixels within the mask
            var = np.var(volume[i][slice_mask])
            variances.append((i, var))
        else:
            variances.append((i, 0.0))

    # Sort by variance descending
    variances.sort(key=lambda x: x[1], reverse=True)

    # Pick top N indices
    top_indices = [x[0] for x in variances[:n_slices]]

    # If we don't have enough slices (e.g. small volume), pad with middle slice or 0
    while len(top_indices) < n_slices:
        if D > 0:
            top_indices.append(D // 2)
        else:
            top_indices.append(0)

    # Sort indices to maintain spatial order
    top_indices.sort()

    return top_indices


def construct_25d_slices(
    volume, slice_indices, img_size=Config.IMG_SIZE, context=Config.SLICE_CONTEXT
):
    """
    Constructs 2.5D slices (3 channels) for the given indices.
    Channels are formed by [z-1, z, z+1].

    Args:
        volume (np.ndarray): 3D HU volume.
        slice_indices (list): List of target slice indices.
        img_size (int): Target spatial dimension (H=W).
        context (int): Number of neighbor slices (default 1 for 3 channels).

    Returns:
        np.ndarray: Array of shape (N, img_size, img_size, 3) normalized to [0, 1].
    """
    output_slices = []
    D, H, W = volume.shape

    # Normalization parameters (Lung Window)
    min_hu = -1000.0
    max_hu = 400.0

    for idx in slice_indices:
        channels = []
        for offset in range(-context, context + 1):  # e.g., -1, 0, 1
            z = idx + offset
            # Clamp z to valid range
            z = max(0, min(D - 1, z))
            slice_img = volume[z]
            channels.append(slice_img)

        # Stack -> (H, W, 3)
        img_stack = np.stack(channels, axis=-1)

        # Clip and Normalize
        img_stack = np.clip(img_stack, min_hu, max_hu)
        img_stack = (img_stack - min_hu) / (max_hu - min_hu)

        # Resize
        # cv2.resize expects (W, H)
        img_resized = cv2.resize(
            img_stack, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )

        output_slices.append(img_resized)

    return np.array(output_slices, dtype=np.float32)


def process_patient(patient_id, dcm_path, load_cached_data=True):
    """
    Orchestrates the image processing pipeline for a single patient with caching.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_path (str): Path to DICOM directory.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (slices_25d, radiomics)
            slices_25d: np.ndarray (N, 224, 224, 3)
            radiomics: np.ndarray (4,)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = Config.get_cache_path(patient_id, ".npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data["slices"], data["radiomics"]
        except Exception as e:
            # If load fails, ignore and recompute
            pass

    # 2. Compute from scratch
    try:
        volume = load_dicom_volume(dcm_path)

        # Radiomics (Global)
        radiomics = compute_density_histogram(volume)

        # Mask for Slice Selection
        mask = get_lung_mask(volume)

        # Slice Selection
        indices = select_variance_slices(volume, mask)

        # 2.5D Construction
        slices_25d = construct_25d_slices(volume, indices)

        # 3. Save Cache
        data = {"slices": slices_25d, "radiomics": radiomics}
        np.save(cache_path, data)

        return slices_25d, radiomics

    except Exception as e:
        # Return safe defaults in case of error (e.g. empty directory, corrupt DICOMs)
        # This prevents the whole pipeline from crashing due to one bad patient
        print(f"Error processing patient {patient_id}: {e}")
        dummy_slices = np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
        )
        dummy_rad = np.zeros(4, dtype=np.float32)
        return dummy_slices, dummy_rad
