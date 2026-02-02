import os
import numpy as np
import cv2
import pydicom
import logging
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger()


def load_scan(path):
    """
    Loads DICOM files from a directory, sorts them by InstanceNumber,
    and returns a list of pydicom datasets.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        list: List of pydicom datasets sorted by Z-position.
    """
    if not os.path.exists(path):
        logger.error(f"DICOM path does not exist: {path}")
        return []

    slices = []
    try:
        files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        if not files:
            logger.warning(f"No .dcm files found in {path}")
            return []

        for fname in files:
            full_path = os.path.join(path, fname)
            try:
                ds = pydicom.dcmread(full_path)
                slices.append(ds)
            except Exception as e:
                logger.warning(f"Failed to read DICOM file {full_path}: {e}")
                continue

        # Sort slices by InstanceNumber (Z-position)
        # If InstanceNumber is missing, try ImagePositionPatient[2]
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            try:
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            except AttributeError:
                logger.warning(
                    f"Could not sort slices for {path}, using filename order."
                )
                slices.sort(key=lambda x: x.filename)

    except Exception as e:
        logger.error(f"Error loading scan from {path}: {e}")
        return []

    return slices


def get_pixels_hu(slices):
    """
    Converts a list of pydicom slices to a numpy array of Hounsfield Units (HU).
    Handles RescaleSlope and RescaleIntercept.

    Args:
        slices (list): List of pydicom datasets.

    Returns:
        np.ndarray: 3D numpy array of HU values (Z, Y, X).
    """
    try:
        image = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    except Exception as e:
        logger.error(f"Error stacking pixel arrays: {e}")
        return np.zeros((len(slices), 512, 512), dtype=np.float32)

    # Convert to Hounsfield Units (HU)
    # HU = pixel * slope + intercept
    if len(slices) > 0:
        intercept = getattr(slices[0], "RescaleIntercept", -1024)
        slope = getattr(slices[0], "RescaleSlope", 1)

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.float32)

        image += np.float32(intercept)

    # Some scans have circular padding with very low values (e.g. -2000),
    # but windowing will handle this naturally.
    return image


def normalize_volume(volume):
    """
    Applies Lung Windowing and normalizes values to [0, 1].
    Window: Level -650, Width 1500.

    Args:
        volume (np.ndarray): 3D array of HU values.

    Returns:
        np.ndarray: Normalized 3D array in range [0, 1].
    """
    level = Config.WINDOW_LEVEL
    width = Config.WINDOW_WIDTH

    lower = level - width / 2
    upper = level + width / 2

    # Clip and normalize
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)

    return volume


def get_slab_boundaries(length, num_slabs, overlap_frac):
    """
    Calculates the start and end indices for overlapping slabs.

    Args:
        length (int): Total length of the dimension.
        num_slabs (int): Number of slabs.
        overlap_frac (float): Fraction of total length to use as overlap.

    Returns:
        list of tuples: [(start, end), ...]
    """
    if length == 0:
        return []

    step = length / num_slabs
    margin = (length * overlap_frac) / 2.0

    boundaries = []
    for i in range(num_slabs):
        center_start = i * step
        center_end = (i + 1) * step

        # Expand boundaries by margin to create overlap
        start = max(0, center_start - margin)
        end = min(length, center_end + margin)

        boundaries.append((int(start), int(end)))

    return boundaries


def generate_orthogonal_tri_slabs(dicom_dir, load_cached_data=True):
    """
    Generates Axial and Coronal Tri-Slab images from a DICOM directory.
    Implements caching to ./working/idea_52/cache/.

    Args:
        dicom_dir (str): Full path to the patient's DICOM directory.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (axial_img, coronal_img)
            - axial_img: (224, 224, 3) float32 array, [0, 1]
            - coronal_img: (224, 224, 3) float32 array, [0, 1]
    """
    # Identify patient ID from path
    # Path format usually: .../train/ID00007637202177411956430
    patient_id = os.path.basename(os.path.normpath(dicom_dir))

    cache_dir = Config.CACHE_DIR
    axial_cache_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_cache_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(axial_cache_path) and os.path.exists(coronal_cache_path):
            try:
                axial_img = np.load(axial_cache_path)
                coronal_img = np.load(coronal_cache_path)
                return axial_img, coronal_img
            except Exception as e:
                logger.warning(
                    f"Cache load failed for {patient_id}, reprocessing. Error: {e}"
                )

    # 2. Process from Scratch
    slices = load_scan(dicom_dir)

    # Handle empty/failed load
    if not slices:
        logger.error(f"Failed to load slices for {patient_id}. Returning zeros.")
        empty_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return empty_img, empty_img

    try:
        # Convert to HU
        vol = get_pixels_hu(slices)
        # Normalize to [0, 1]
        vol = normalize_volume(vol)  # Shape: (Z, Y, X)
    except Exception as e:
        logger.error(f"Preprocessing failed for {patient_id}: {e}")
        empty_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return empty_img, empty_img

    # Resize slices to target size (224x224) immediately to save memory
    # Original slices are typically 512x512
    Z, H, W = vol.shape
    target_size = (Config.IMG_SIZE, Config.IMG_SIZE)

    # Pre-allocate resized volume
    vol_resized = np.zeros((Z, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
    for i in range(Z):
        vol_resized[i] = cv2.resize(vol[i], target_size, interpolation=cv2.INTER_AREA)

    vol = vol_resized
    Z, H, W = vol.shape  # Now (Z, 224, 224)

    # --- Generate Axial Tri-Slab (Projection along Z) ---
    axial_boundaries = get_slab_boundaries(Z, Config.NUM_SLABS, Config.SLAB_OVERLAP)
    axial_channels = []

    for start, end in axial_boundaries:
        if start >= end:
            # Handle edge case of very few slices
            mip = np.zeros((H, W), dtype=np.float32)
        else:
            slab = vol[start:end, :, :]
            mip = np.max(slab, axis=0)  # Max along Z
        axial_channels.append(mip)

    axial_img = np.stack(axial_channels, axis=-1)  # (224, 224, 3)

    # --- Generate Coronal Tri-Slab (Projection along Y) ---
    # Volume is (Z, Y, X) = (Z, 224, 224).
    # Coronal view corresponds to the (Z, X) plane, looking through Y.
    # We split the Y dimension (Anterior-Posterior) into slabs.
    coronal_boundaries = get_slab_boundaries(H, Config.NUM_SLABS, Config.SLAB_OVERLAP)
    coronal_channels = []

    for start, end in coronal_boundaries:
        if start >= end:
            mip_resized = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
        else:
            # Slice along Y (axis 1)
            slab = vol[:, start:end, :]
            # Max along Y -> Result is (Z, X)
            mip = np.max(slab, axis=1)

            # Resize to 224x224
            # cv2.resize expects (width, height) -> (X, Z)
            # We map Z (depth) to image height, X (width) to image width.
            # Depending on patient, Z can be < 224 or > 224.
            mip_resized = cv2.resize(mip, target_size, interpolation=cv2.INTER_LINEAR)

        coronal_channels.append(mip_resized)

    coronal_img = np.stack(coronal_channels, axis=-1)  # (224, 224, 3)

    # 3. Save to Cache
    try:
        np.save(axial_cache_path, axial_img)
        np.save(coronal_cache_path, coronal_img)
    except Exception as e:
        logger.warning(f"Failed to save cache for {patient_id}: {e}")

    return axial_img, coronal_img
