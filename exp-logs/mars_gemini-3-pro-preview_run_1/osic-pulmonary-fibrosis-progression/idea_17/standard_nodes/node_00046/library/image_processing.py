import os
import glob
import numpy as np
import pydicom
import cv2
from library.config import Config


def load_scan(path):
    """
    Loads DICOM files from a directory, sorts them by instance number or position,
    and returns a list of pydicom datasets.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DICOM directory not found: {path}")

    files = [os.path.join(path, s) for s in os.listdir(path) if s.endswith(".dcm")]
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {path}")

    slices = [pydicom.dcmread(s) for s in files]

    # Sort by ImagePositionPatient Z coordinate if available, otherwise InstanceNumber
    # ImagePositionPatient is [x, y, z]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts a list of pydicom datasets to a 3D numpy array of Hounsfield Units.
    Handles slope, intercept, and padding.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately -1000
    image[image == -2000] = 0

    # Convert to Hounsfield Units (HU)
    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else -1024
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)

        image[i] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def window_image(img, window_center=-600, window_width=1500):
    """
    Applies a standard lung window to the HU values and normalizes to [0, 255].
    Lung Window: Center=-600, Width=1500 -> Range [-1350, 150]
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    # Normalize to 0-1 then 0-255
    # Avoid division by zero
    if img_max != img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min  # Should be 0

    img = (img * 255).astype(np.uint8)
    return img


def resize_image(img, size):
    """
    Resizes an image to the specified square size using Area interpolation
    (better for downsampling).
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def generate_tri_slab_mip(volume, axis_idx):
    """
    Generates a Tri-Slab Maximum Intensity Projection (MIP) image.

    Args:
        volume (np.ndarray): 3D volume (D, H, W) in HU.
        axis_idx (int): 0 for Axial (Z-axis), 1 for Coronal (Y-axis).

    Returns:
        np.ndarray: (Size, Size, 3) RGB image.
    """
    # Volume shape is (Z, Y, X)
    # If Coronal (axis 1), we want to slice along Y.
    # To reuse logic, we transpose Y to the 0-th dimension: (Y, Z, X)
    if axis_idx == 1:
        working_volume = volume.transpose(1, 0, 2)
    else:
        working_volume = volume

    n_slices = working_volume.shape[0]

    # Handle cases with very few slices by duplicating
    if n_slices < 3:
        mip = np.max(working_volume, axis=0)
        mip = window_image(mip)
        mip = resize_image(mip, Config.IMG_SIZE)
        return np.stack([mip, mip, mip], axis=-1)

    # Calculate slab boundaries
    # We divide the depth into 3 equal segments
    seg_len = n_slices / 3.0

    # Calculate overlap pixels
    overlap_px = int(seg_len * Config.SLAB_OVERLAP)

    # Define indices
    # Slab 1: 0 -> 1/3 + overlap
    s1_start = 0
    s1_end = int(seg_len + overlap_px)

    # Slab 2: 1/3 - overlap -> 2/3 + overlap
    s2_start = int(seg_len - overlap_px)
    s2_end = int(2 * seg_len + overlap_px)

    # Slab 3: 2/3 - overlap -> end
    s3_start = int(2 * seg_len - overlap_px)
    s3_end = n_slices

    # Safety clamping
    s1_end = min(s1_end, n_slices)
    s2_start = max(0, s2_start)
    s2_end = min(s2_end, n_slices)
    s3_start = max(0, s3_start)

    # Extract slabs
    slab1 = working_volume[s1_start:s1_end, :, :]
    slab2 = working_volume[s2_start:s2_end, :, :]
    slab3 = working_volume[s3_start:s3_end, :, :]

    # Compute MIPs
    # If a slab ends up empty (unlikely given math), use zeros
    mip1 = (
        np.max(slab1, axis=0)
        if slab1.shape[0] > 0
        else np.zeros_like(working_volume[0])
    )
    mip2 = (
        np.max(slab2, axis=0)
        if slab2.shape[0] > 0
        else np.zeros_like(working_volume[0])
    )
    mip3 = (
        np.max(slab3, axis=0)
        if slab3.shape[0] > 0
        else np.zeros_like(working_volume[0])
    )

    # Apply Windowing
    mip1 = window_image(mip1)
    mip2 = window_image(mip2)
    mip3 = window_image(mip3)

    # Resize
    mip1 = resize_image(mip1, Config.IMG_SIZE)
    mip2 = resize_image(mip2, Config.IMG_SIZE)
    mip3 = resize_image(mip3, Config.IMG_SIZE)

    # Stack channels
    img = np.stack([mip1, mip2, mip3], axis=-1)

    return img


def process_patient(patient_id, dicom_dir, load_cached_data=True):
    """
    Processes a patient's CT scan to generate Axial and Coronal Tri-Slab inputs.
    Implements caching to disk.

    Args:
        patient_id (str): Unique Patient ID.
        dicom_dir (str): Relative path to DICOM directory (e.g. "train/ID...").
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (axial_img, coronal_img) both as np.ndarray of shape (H, W, 3).
    """
    cache_dir = Config.CACHE_DIR
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Attempt Load
    if load_cached_data:
        if os.path.exists(axial_path) and os.path.exists(coronal_path):
            try:
                axial_img = np.load(axial_path)
                coronal_img = np.load(coronal_path)
                return axial_img, coronal_img
            except Exception:
                # Corrupt file, proceed to re-process
                pass

    # 2. Process
    full_dicom_path = os.path.join(Config.INPUT_DIR, dicom_dir)

    try:
        slices = load_scan(full_dicom_path)
        volume = get_pixels_hu(slices)

        # Generate Axial (Axis 0)
        axial_img = generate_tri_slab_mip(volume, axis_idx=0)

        # Generate Coronal (Axis 1)
        coronal_img = generate_tri_slab_mip(volume, axis_idx=1)

        # 3. Save
        os.makedirs(cache_dir, exist_ok=True)
        np.save(axial_path, axial_img)
        np.save(coronal_path, coronal_img)

        return axial_img, coronal_img

    except Exception as e:
        # Fallback for errors (e.g. empty directory, bad DICOMs)
        # Return black images to keep pipeline running
        print(f"Warning: Failed to process patient {patient_id}. Error: {e}")
        empty_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        return empty_img, empty_img
