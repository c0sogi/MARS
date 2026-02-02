import os
import glob
import numpy as np
import cv2
import warnings
from library.config import (
    HU_MIN,
    HU_MAX,
    VOL_HU_MIN,
    VOL_HU_MAX,
    IMG_SIZE,
    CACHE_DIR,
    SEED,
    INPUT_DIR,
)

# Set random seed for reproducibility
np.random.seed(SEED)

# Attempt to import pydicom. If not available, we will handle it gracefully.
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    warnings.warn("pydicom module not found. Image processing will return dummy data.")


def load_scan(dcm_dir):
    """
    Loads DICOM files from a directory, sorts them by slice location,
    and converts pixel values to Hounsfield Units (HU).

    Args:
        dcm_dir (str): Path to the directory containing .dcm files.

    Returns:
        scan (np.ndarray): 3D numpy array of shape (Depth, Height, Width) in HU.
        spacing (tuple): (slice_thickness, pixel_spacing_x, pixel_spacing_y) in mm.
    """
    if not HAS_PYDICOM:
        # Fallback: Return dummy 3D array and default spacing
        return np.zeros((10, 512, 512), dtype=np.float32), (1.0, 0.7, 0.7)

    # List all .dcm files
    files = glob.glob(os.path.join(dcm_dir, "*.dcm"))
    if not files:
        # Return dummy if directory is empty
        return np.zeros((1, 512, 512), dtype=np.float32), (1.0, 1.0, 1.0)

    # Read DICOM files
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return np.zeros((1, 512, 512), dtype=np.float32), (1.0, 1.0, 1.0)

    # Sort slices by ImagePositionPatient Z-coordinate (reliable for spatial ordering)
    # Fallback to InstanceNumber if position is missing
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass  # Keep file system order if all metadata missing

    # Extract Physical Spacing Information
    try:
        # PixelSpacing is usually [RowSpacing, ColSpacing] -> [Y, X]
        ps = slices[0].PixelSpacing
        spacing_y, spacing_x = float(ps[0]), float(ps[1])
    except (AttributeError, IndexError):
        spacing_y, spacing_x = 0.7, 0.7  # Approximate default

    try:
        slice_thickness = float(slices[0].SliceThickness)
    except (AttributeError, IndexError):
        # Estimate thickness from Z-position difference of first two slices
        if len(slices) > 1:
            try:
                z0 = float(slices[0].ImagePositionPatient[2])
                z1 = float(slices[1].ImagePositionPatient[2])
                slice_thickness = abs(z1 - z0)
            except:
                slice_thickness = 1.0
        else:
            slice_thickness = 1.0

    # Convert raw pixel values to Hounsfield Units (HU)
    image_stack = []
    for s in slices:
        try:
            # Raw pixel data
            img = s.pixel_array.astype(np.float32)

            # Apply Rescale Slope and Intercept
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = slope * img + intercept

            image_stack.append(img)
        except Exception:
            # Skip slice if pixel data cannot be read/decompressed
            continue

    if not image_stack:
        # Fallback if no pixels could be read
        return np.zeros((1, 512, 512), dtype=np.float32), (1.0, 1.0, 1.0)

    # Stack into 3D array (Depth, Height, Width)
    scan = np.stack(image_stack, axis=0)
    return scan, (slice_thickness, spacing_x, spacing_y)


def compute_volumetrics(scan, spacing):
    """
    Computes physical lung volume and mean lung density based on HU thresholds.

    Args:
        scan (np.ndarray): 3D array in HU.
        spacing (tuple): (slice_thickness, spacing_x, spacing_y).

    Returns:
        volume_ml (float): Total lung volume in milliliters.
        mean_density (float): Mean HU of lung tissue.
    """
    if scan is None or scan.size == 0:
        return 0.0, -1000.0

    slice_thickness, sx, sy = spacing
    # Voxel volume in mm^3
    voxel_volume = slice_thickness * sx * sy
    # Convert to mL (1 cm^3 = 1000 mm^3)
    voxel_volume_ml = voxel_volume / 1000.0

    # Create Lung Mask based on HU thresholds
    # Lungs are typically between -1000 and -400 HU
    mask = (scan >= VOL_HU_MIN) & (scan <= VOL_HU_MAX)

    lung_voxels = np.sum(mask)
    volume_ml = lung_voxels * voxel_volume_ml

    if lung_voxels > 0:
        mean_density = np.mean(scan[mask])
    else:
        mean_density = -1000.0  # Default to air density if no lung found

    return float(volume_ml), float(mean_density)


def select_stratified_slices(scan):
    """
    Implements Stratified-Variance Sampling.
    Partitions the scan into Apex, Mid, and Base zones, and selects the slice
    with the highest variance from each zone.

    Args:
        scan (np.ndarray): 3D array (D, H, W).

    Returns:
        img_tensor (np.ndarray): Processed image tensor of shape (3, IMG_SIZE, IMG_SIZE).
                                 Values are normalized to [0, 1].
    """
    depth = scan.shape[0]

    # Handle cases with very few slices by duplicating
    if depth < 3:
        indices = [0] * 3
        if depth == 2:
            indices = [0, 1, 1]
    else:
        # Partition depth into 3 zones
        chunk_size = depth // 3
        ranges = [
            (0, chunk_size),  # Apex
            (chunk_size, 2 * chunk_size),  # Mid
            (2 * chunk_size, depth),  # Base
        ]

        indices = []
        for start, end in ranges:
            # Extract sub-volume for the zone
            sub_vol = scan[start:end]

            # Calculate variance for each slice (spatial complexity proxy)
            # Variance is computed over height and width axes
            variances = np.var(sub_vol, axis=(1, 2))

            # Select index of maximum variance relative to the sub-volume
            max_idx = np.argmax(variances)

            # Map back to global index
            indices.append(start + max_idx)

    # Process the selected slices
    selected_slices = []
    for idx in indices:
        slc = scan[idx]

        # 1. Windowing / Clipping
        slc = np.clip(slc, HU_MIN, HU_MAX)

        # 2. Normalization [0, 1]
        slc = (slc - HU_MIN) / (HU_MAX - HU_MIN)

        # 3. Resizing
        # cv2.resize expects (Width, Height)
        slc_resized = cv2.resize(
            slc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        selected_slices.append(slc_resized)

    # Stack into tensor (Channels, Height, Width) -> (3, 224, 224)
    img_tensor = np.stack(selected_slices, axis=0)
    return img_tensor.astype(np.float32)


def process_patient(patient_id, dcm_path_rel, load_cached_data=True):
    """
    Orchestrates the processing for a single patient.
    Checks cache first; if missing, loads DICOM, computes features, and caches result.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_path_rel (str): Relative path to DICOM directory (e.g., "train/ID...").
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        image (np.ndarray): (3, IMG_SIZE, IMG_SIZE)
        volumetrics (np.ndarray): (2,) containing [Volume, Density]
    """
    # Define cache file path
    cache_path = os.path.join(CACHE_DIR, f"{patient_id}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            return data["image"], data["volumetrics"]
        except Exception:
            # If load fails (corrupt file), proceed to re-compute
            pass

    # 2. Compute from scratch
    full_path = os.path.join(INPUT_DIR, dcm_path_rel)

    if not os.path.exists(full_path):
        # Handle case where directory is missing (robustness)
        img = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        vol = np.array([0.0, -1000.0], dtype=np.float32)
    else:
        # Load and Process
        scan, spacing = load_scan(full_path)
        volume, density = compute_volumetrics(scan, spacing)
        img = select_stratified_slices(scan)
        vol = np.array([volume, density], dtype=np.float32)

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Use savez to store multiple arrays (no pickle required for basic arrays)
    np.savez(cache_path, image=img, volumetrics=vol)

    return img, vol
