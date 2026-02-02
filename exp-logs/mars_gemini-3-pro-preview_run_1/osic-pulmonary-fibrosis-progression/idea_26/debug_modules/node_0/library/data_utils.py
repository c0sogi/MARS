import os
import numpy as np
import cv2
import pydicom
import torch
from library.config import Config


def load_scan(path):
    """
    Loads a CT scan from a directory of DICOM files.
    Returns a 3D numpy array (Z, Y, X) in Hounsfield Units.
    """
    try:
        if not os.path.exists(path):
            # Return empty volume if path doesn't exist
            return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        if not files:
            return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Read all dicom files
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, f))
                slices.append(ds)
            except Exception:
                continue

        if not slices:
            return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Sort slices by ImagePositionPatient Z coordinate (index 2)
        # If not available, fall back to InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Convert to Hounsfield Units (HU)
        images = []
        for s in slices:
            # Convert to float32
            img_2d = s.pixel_array.astype(np.float32)

            # Apply Slope and Intercept
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)

            # Handle potential array scaling if slope != 1
            if slope != 1:
                img_2d = slope * img_2d.astype(np.float64)
                img_2d = img_2d.astype(np.float32)

            img_2d += intercept
            images.append(img_2d)

        # Stack into (Z, Y, X) volume
        volume = np.stack(images, axis=0)
        return volume

    except Exception as e:
        print(f"Error loading scan from {path}: {e}")
        return np.zeros((10, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def get_tri_slab(volume, view="axial"):
    """
    Generates a 3-channel image using Fixed Overlapping Tri-Slabs.

    Args:
        volume: 3D numpy array (Z, Y, X)
        view: 'axial' or 'coronal'

    Returns:
        3-channel numpy array (H, W, 3) representing the MIPs of the slabs.
    """
    # Handle empty or malformed volumes
    if volume.ndim != 3 or volume.shape[0] == 0:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Orient volume based on view
    if view == "axial":
        # Axial view looks along Z. Slabs are chunks of Z.
        # Volume is already (Z, Y, X).
        processing_vol = volume
    elif view == "coronal":
        # Coronal view looks along Y. Slabs are chunks of Y.
        # Transpose (Z, Y, X) -> (Y, Z, X) so Y becomes the depth dimension.
        processing_vol = volume.transpose(1, 0, 2)
    else:
        raise ValueError(f"Unsupported view: {view}")

    D = processing_vol.shape[0]

    # If volume is too thin, just replicate MIP
    if D < 3:
        mip = np.max(processing_vol, axis=0)
        return np.stack([mip, mip, mip], axis=-1)

    # Calculate slab boundaries with overlap
    # Core regions: [0, 1/3), [1/3, 2/3), [2/3, 1]
    # Overlap: Config.SLAB_OVERLAP (fraction of total depth D)

    # Note: Config.SLAB_OVERLAP is 0.15 (15%)
    overlap_pixels = int(D * Config.SLAB_OVERLAP)

    idx_1_3 = int(D / 3)
    idx_2_3 = int(2 * D / 3)

    # Slab 1: 0 to 33% + overlap
    s1_start = 0
    s1_end = min(D, idx_1_3 + overlap_pixels)

    # Slab 2: 33% - overlap to 66% + overlap
    s2_start = max(0, idx_1_3 - overlap_pixels)
    s2_end = min(D, idx_2_3 + overlap_pixels)

    # Slab 3: 66% - overlap to 100%
    s3_start = max(0, idx_2_3 - overlap_pixels)
    s3_end = D

    # Compute Maximum Intensity Projection (MIP) for each slab
    # Axis 0 is the depth dimension after orientation
    slab1 = np.max(processing_vol[s1_start:s1_end, :, :], axis=0)
    slab2 = np.max(processing_vol[s2_start:s2_end, :, :], axis=0)
    slab3 = np.max(processing_vol[s3_start:s3_end, :, :], axis=0)

    # Stack into RGB-like image (H, W, 3)
    img = np.stack([slab1, slab2, slab3], axis=-1)
    return img


def normalize_and_resize(img):
    """
    Applies Lung Windowing, Normalization, and Resizing.

    Args:
        img: (H, W, 3) numpy array

    Returns:
        (Config.IMG_SIZE, Config.IMG_SIZE, 3) numpy array, normalized to [0, 1]
    """
    # Lung Window Settings
    # Level (L) = -600, Width (W) = 1500
    # Range: [-1350, 150]
    L = -600
    W = 1500
    lower_bound = L - W // 2
    upper_bound = L + W // 2

    # Clip to window
    img = np.clip(img, lower_bound, upper_bound)

    # Normalize to [0, 1]
    img = (img - lower_bound) / (upper_bound - lower_bound)

    # Resize to target size
    # cv2.resize expects (Width, Height)
    img_resized = cv2.resize(
        img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
    )

    return img_resized.astype(np.float32)


def load_patient_images(patient_id, dicom_dir_rel, load_cached_data=True):
    """
    Loads and processes images for a specific patient.
    Implements caching mechanism.

    Args:
        patient_id: Unique patient ID (str)
        dicom_dir_rel: Relative path to DICOM directory (from Config.DICOM_ROOT)
        load_cached_data: Whether to try loading from cache

    Returns:
        tuple: (axial_image, coronal_image)
    """
    # Ensure cache directory exists (redundant with Config but safe)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_ax_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
    cache_cor_path = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_ax_path) and os.path.exists(cache_cor_path):
            try:
                ax = np.load(cache_ax_path)
                cor = np.load(cache_cor_path)
                return ax, cor
            except Exception:
                # If load fails, proceed to recompute
                pass

    # 2. Compute from scratch
    full_dicom_path = os.path.join(Config.DICOM_ROOT, dicom_dir_rel)

    # Load 3D Volume
    volume = load_scan(full_dicom_path)

    # Process Axial View
    ax_raw = get_tri_slab(volume, view="axial")
    ax = normalize_and_resize(ax_raw)

    # Process Coronal View
    cor_raw = get_tri_slab(volume, view="coronal")
    cor = normalize_and_resize(cor_raw)

    # 3. Save to cache
    try:
        np.save(cache_ax_path, ax)
        np.save(cache_cor_path, cor)
    except Exception as e:
        print(f"Warning: Failed to save cache for {patient_id}: {e}")

    return ax, cor
