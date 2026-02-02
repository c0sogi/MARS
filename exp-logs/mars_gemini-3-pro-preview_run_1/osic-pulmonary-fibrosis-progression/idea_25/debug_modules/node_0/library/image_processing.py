import os
import numpy as np
import pydicom
import cv2
from library.config import Config


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by slice location.

    Args:
        path (str): Path to the directory containing DICOM files.

    Returns:
        list: A list of pydicom datasets sorted by ImagePositionPatient Z-coordinate.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DICOM directory not found: {path}")

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception:
                # Skip corrupt files
                continue

    if not slices:
        raise ValueError(f"No valid DICOM files found in {path}")

    # Sort by ImagePositionPatient[2] (Z-axis) if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).

    Args:
        scans (list): List of sorted pydicom datasets.

    Returns:
        np.ndarray: 3D numpy array of HU values (Z, Y, X).
    """
    image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (standard for HU)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield Units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def normalize_hu(image, min_hu=-1000, max_hu=400):
    """
    Normalizes HU values to range [0, 255] based on a lung window.

    Args:
        image (np.ndarray): Input HU image.
        min_hu (int): Minimum HU value (Air).
        max_hu (int): Maximum HU value (Tissue/Bone).

    Returns:
        np.ndarray: Normalized uint8 image.
    """
    image = (image - min_hu) / (max_hu - min_hu)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def generate_tri_slab_mip(volume, axis, target_size=(224, 224), overlap_ratio=0.15):
    """
    Generates a 3-channel RGB image using Maximum Intensity Projection (MIP)
    over 3 overlapping slabs along the specified axis.

    Args:
        volume (np.ndarray): 3D HU volume (Z, Y, X).
        axis (int): Axis to slice along. 0 for Axial (Z), 1 for Coronal (Y).
        target_size (tuple): Output resolution (H, W).
        overlap_ratio (float): Ratio of total depth to use for overlap.

    Returns:
        np.ndarray: Processed RGB image (H, W, 3).
    """
    # If generating Coronal view (slicing along Y), transpose to make Y the primary axis (Depth)
    # Original: (Z, Y, X)
    # Target for processing: (Depth, Height, Width)
    if axis == 0:
        # Axial: Depth=Z, H=Y, W=X
        vol_processed = volume
    elif axis == 1:
        # Coronal: Depth=Y, H=Z, W=X
        # Transpose (Z, Y, X) -> (Y, Z, X)
        vol_processed = np.transpose(volume, (1, 0, 2))
    else:
        raise ValueError("Axis must be 0 (Axial) or 1 (Coronal)")

    depth = vol_processed.shape[0]

    # Define slab boundaries
    # Core splits: 0 - 1/3 - 2/3 - 1
    # Overlap buffer
    overlap_pixels = int(depth * overlap_ratio)
    p1 = int(depth / 3)
    p2 = int(2 * depth / 3)

    # Slab 1: 0 to 1/3 + overlap/2
    s1_start = 0
    s1_end = min(depth, p1 + overlap_pixels // 2)

    # Slab 2: 1/3 - overlap/2 to 2/3 + overlap/2
    s2_start = max(0, p1 - overlap_pixels // 2)
    s2_end = min(depth, p2 + overlap_pixels // 2)

    # Slab 3: 2/3 - overlap/2 to End
    s3_start = max(0, p2 - overlap_pixels // 2)
    s3_end = depth

    # Extract slabs
    slab1 = vol_processed[s1_start:s1_end, :, :]
    slab2 = vol_processed[s2_start:s2_end, :, :]
    slab3 = vol_processed[s3_start:s3_end, :, :]

    # Handle edge case where volume is too thin (depth < 3)
    if slab1.size == 0:
        slab1 = vol_processed
    if slab2.size == 0:
        slab2 = vol_processed
    if slab3.size == 0:
        slab3 = vol_processed

    # Compute MIP (Maximum Intensity Projection) along depth
    # If slab is empty (shouldn't happen with logic above), use zeros
    mip1 = np.max(slab1, axis=0) if slab1.size > 0 else np.zeros_like(vol_processed[0])
    mip2 = np.max(slab2, axis=0) if slab2.size > 0 else np.zeros_like(vol_processed[0])
    mip3 = np.max(slab3, axis=0) if slab3.size > 0 else np.zeros_like(vol_processed[0])

    # Normalize HU to uint8
    mip1 = normalize_hu(mip1)
    mip2 = normalize_hu(mip2)
    mip3 = normalize_hu(mip3)

    # Stack into RGB
    img_rgb = np.stack([mip1, mip2, mip3], axis=-1)

    # Resize to target size
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    return img_resized


def process_patient(patient_id, dicom_rel_dir, load_cached_data=True):
    """
    Main processing function for a single patient.
    Loads DICOMs, generates Axial and Coronal Tri-Slab inputs, and handles caching.

    Args:
        patient_id (str): Unique patient identifier.
        dicom_rel_dir (str): Relative path to DICOM directory (e.g. "train/ID...").
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (axial_image, coronal_image) as numpy arrays.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path_ax = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
    cache_path_cor = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path_ax) and os.path.exists(cache_path_cor):
            try:
                ax_img = np.load(cache_path_ax)
                cor_img = np.load(cache_path_cor)
                return ax_img, cor_img
            except Exception:
                pass  # Fallback to processing if cache is corrupt

    # 2. Process from scratch
    full_path = os.path.join(Config.INPUT_ROOT, dicom_rel_dir)

    try:
        scans = load_scan(full_path)
        vol_hu = get_pixels_hu(scans)

        # Generate Axial View (Axis 0)
        ax_img = generate_tri_slab_mip(
            vol_hu,
            axis=0,
            target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            overlap_ratio=Config.SLAB_OVERLAP,
        )

        # Generate Coronal View (Axis 1)
        cor_img = generate_tri_slab_mip(
            vol_hu,
            axis=1,
            target_size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            overlap_ratio=Config.SLAB_OVERLAP,
        )

        # Save to cache
        np.save(cache_path_ax, ax_img)
        np.save(cache_path_cor, cor_img)

        return ax_img, cor_img

    except Exception as e:
        # Fallback for completely failed patients (e.g., empty dir)
        # Return black images to prevent pipeline crash
        print(f"Error processing patient {patient_id}: {e}")
        blank = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        return blank, blank
