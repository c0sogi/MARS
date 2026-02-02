import os
import numpy as np
import pydicom
import cv2
from library.config import Paths, Data


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by slice location.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        list: A sorted list of pydicom datasets.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DICOM directory not found: {path}")

    slices = [
        pydicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.endswith(".dcm")
    ]

    if not slices:
        raise FileNotFoundError(f"No .dcm files found in {path}")

    # Sort by ImagePositionPatient[2] (Z-coordinate)
    # If ImagePositionPatient is missing, fall back to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(scans):
    """
    Converts a list of pydicom datasets to a 3D numpy array of Hounsfield Units.

    Args:
        scans (list): List of pydicom datasets.

    Returns:
        np.ndarray: 3D array of HU values (Z, Y, X).
    """
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Convert to Hounsfield Units (HU)
    # The intercept is usually -1024, so air is approximately -1000
    image[image == -2000] = 0  # Fix for some scanners using -2000 as padding

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def apply_lung_window(image, level=-600, width=1500):
    """
    Applies a lung window to the image and normalizes to [0, 1].

    Args:
        image (np.ndarray): Input image in HU.
        level (int): Window level (center).
        width (int): Window width.

    Returns:
        np.ndarray: Windowed and normalized image (float32).
    """
    upper = level + width / 2
    lower = level - width / 2

    img_windowed = np.clip(image, lower, upper)
    img_windowed = (img_windowed - lower) / (upper - lower)

    return img_windowed.astype(np.float32)


def resize_image(image, size=(224, 224)):
    """
    Resizes an image to the target size.

    Args:
        image (np.ndarray): Input 2D image.
        size (tuple): Target size (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    # cv2.resize expects (width, height)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def generate_tri_slab(volume, axis_idx):
    """
    Generates a Tri-Slab RGB image from a 3D volume.

    Splits the volume into 3 overlapping slabs along the specified axis,
    computes the Maximum Intensity Projection (MIP) for each, applies
    lung windowing, resizes, and stacks them into channels.

    Args:
        volume (np.ndarray): 3D volume (Z, Y, X).
        axis_idx (int): Axis to split along (0 for Z/Axial, 1 for Y/Coronal).

    Returns:
        np.ndarray: Processed image of shape (224, 224, 3) normalized to [0, 1].
    """
    # Determine the size of the dimension we are splitting
    dim_size = volume.shape[axis_idx]

    # Define slab boundaries (0-33%, 33-66%, 66-100% with overlap)
    # Using 1/3 splits with overlap
    chunk_size = dim_size / 3.0
    overlap = dim_size * Data.SLAB_OVERLAP

    # Calculate start and end indices for 3 slabs
    # Slab 1: 0 to 1/3 + overlap
    s1_start = 0
    s1_end = int(chunk_size + overlap)

    # Slab 2: 1/3 - overlap to 2/3 + overlap
    s2_start = int(chunk_size - overlap)
    s2_end = int(2 * chunk_size + overlap)

    # Slab 3: 2/3 - overlap to end
    s3_start = int(2 * chunk_size - overlap)
    s3_end = dim_size

    # Ensure indices are within bounds
    s1_end = min(s1_end, dim_size)
    s2_start = max(0, s2_start)
    s2_end = min(s2_end, dim_size)
    s3_start = max(0, s3_start)

    slabs_indices = [(s1_start, s1_end), (s2_start, s2_end), (s3_start, s3_end)]
    channels = []

    for start, end in slabs_indices:
        # Slice the volume along the specified axis
        if axis_idx == 0:  # Axial split (Z)
            slab = volume[start:end, :, :]
            # MIP along Z (axis 0) -> Result (Y, X)
            if slab.shape[0] > 0:
                mip = np.max(slab, axis=0)
            else:
                mip = np.zeros((volume.shape[1], volume.shape[2]), dtype=volume.dtype)

        elif axis_idx == 1:  # Coronal split (Y)
            slab = volume[:, start:end, :]
            # MIP along Y (axis 1) -> Result (Z, X)
            if slab.shape[1] > 0:
                mip = np.max(slab, axis=1)
            else:
                mip = np.zeros((volume.shape[0], volume.shape[2]), dtype=volume.dtype)
        else:
            raise ValueError("Axis must be 0 (Axial) or 1 (Coronal)")

        # Apply Lung Window
        mip_windowed = apply_lung_window(mip)

        # Resize to target resolution
        mip_resized = resize_image(mip_windowed, size=(Data.IMG_SIZE, Data.IMG_SIZE))

        channels.append(mip_resized)

    # Stack into RGB (H, W, 3)
    img_rgb = np.stack(channels, axis=-1)

    return img_rgb


def process_patient(patient_id, dicom_dir_rel, load_cached_data=True):
    """
    Orchestrates the processing of a patient's CT scan into Axial and Coronal Tri-Slabs.
    Handles caching to disk.

    Args:
        patient_id (str): Unique patient identifier.
        dicom_dir_rel (str): Relative path to DICOM directory (e.g. "train/ID...").
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (axial_img, coronal_img)
               Each is a np.ndarray of shape (224, 224, 3), float32, range [0, 1].
    """
    # Ensure cache directory exists
    os.makedirs(Paths.CACHE_DIR, exist_ok=True)

    cache_path_axial = os.path.join(Paths.CACHE_DIR, f"{patient_id}_axial.npy")
    cache_path_coronal = os.path.join(Paths.CACHE_DIR, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(cache_path_axial) and os.path.exists(cache_path_coronal):
            try:
                axial_img = np.load(cache_path_axial)
                coronal_img = np.load(cache_path_coronal)
                return axial_img, coronal_img
            except Exception:
                # If load fails, proceed to recompute
                pass

    # 2. Compute from scratch
    full_dicom_path = os.path.join(Paths.INPUT_ROOT, dicom_dir_rel)

    try:
        scans = load_scan(full_dicom_path)
        volume_hu = get_pixels_hu(scans)

        # Generate Axial View (Split Z - Axis 0)
        axial_img = generate_tri_slab(volume_hu, axis_idx=0)

        # Generate Coronal View (Split Y - Axis 1)
        coronal_img = generate_tri_slab(volume_hu, axis_idx=1)

        # 3. Save to cache
        np.save(cache_path_axial, axial_img)
        np.save(cache_path_coronal, coronal_img)

        return axial_img, coronal_img

    except Exception as e:
        # In case of failure (e.g. corrupt DICOMs), return zero placeholders
        # This prevents the entire pipeline from crashing due to one bad patient
        # Cite debug_lesson_2: Handle Missing Codec Dependencies Gracefully
        # We truncate the error to avoid log flooding and cache the placeholder to avoid re-processing.
        print(
            f"Warning: Error processing patient {patient_id} (using placeholder): {str(e)[:150]}..."
        )
        placeholder = np.zeros((Data.IMG_SIZE, Data.IMG_SIZE, 3), dtype=np.float32)

        # Save to cache so subsequent visits for this patient load the placeholder immediately
        np.save(cache_path_axial, placeholder)
        np.save(cache_path_coronal, placeholder)

        return placeholder, placeholder
