import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
from joblib import Parallel, delayed
from library.config import Config


def get_pixel_hu(slices):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).
    Handles slope and intercept corrections.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    for i, s in enumerate(slices):
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else 0

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)

        image[i] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def load_scan(path):
    """
    Loads a DICOM scan from a directory, sorts by Z-position,
    and returns the volume in Hounsfield Units.

    Returns:
        np.ndarray: 3D volume (Z, Y, X)
    """
    if not os.path.exists(path):
        return None

    # List all DICOM files
    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return None

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except:
            continue

    if not slices:
        return None

    # Sort by ImagePositionPatient Z coordinate
    # ImagePositionPatient is [x, y, z]
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        # Fallback to InstanceNumber if position is missing
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Get volume in HU
    volume = get_pixel_hu(slices)
    return volume


def normalize_volume(volume):
    """
    Applies lung windowing [-1000, 400] and normalizes to [0, 255].
    """
    # Lung window
    min_hu = -1000
    max_hu = 400

    volume = np.clip(volume, min_hu, max_hu)

    # Normalize to 0-255
    volume = (volume - min_hu) / (max_hu - min_hu)
    volume = volume * 255.0
    volume = volume.astype(np.uint8)

    return volume


def generate_tri_slab(volume, axis_idx):
    """
    Generates a 3-channel image using Maximum Intensity Projection (MIP)
    over 3 overlapping slabs along the specified axis.

    Args:
        volume: 3D numpy array (Z, Y, X)
        axis_idx: 0 for Axial (Z), 1 for Coronal (Y)

    Returns:
        np.ndarray: 2D image with 3 channels (H, W, 3)
    """
    # Determine the size of the dimension we are collapsing
    depth = volume.shape[axis_idx]

    # If depth is too small, just repeat the max projection
    if depth < 3:
        mip = np.max(volume, axis=axis_idx)
        return np.stack([mip, mip, mip], axis=-1)

    # Define overlapping slab boundaries (approx 15% overlap logic)
    # Slab 1: 0% - 40%
    # Slab 2: 30% - 70%
    # Slab 3: 60% - 100%

    p1 = int(depth * 0.40)
    p2_start = int(depth * 0.30)
    p2_end = int(depth * 0.70)
    p3_start = int(depth * 0.60)

    # Handle small rounding issues
    if p1 == 0:
        p1 = 1
    if p2_end <= p2_start:
        p2_end = p2_start + 1

    # Extract slabs based on axis
    if axis_idx == 0:
        # Axial: Split Z (dim 0) -> Result (Y, X)
        slab1 = volume[0:p1, :, :]
        slab2 = volume[p2_start:p2_end, :, :]
        slab3 = volume[p3_start:, :, :]
    else:
        # Coronal: Split Y (dim 1) -> Result (Z, X)
        slab1 = volume[:, 0:p1, :]
        slab2 = volume[:, p2_start:p2_end, :]
        slab3 = volume[:, p3_start:, :]

    # Compute MIP
    c1 = np.max(slab1, axis=axis_idx)
    c2 = np.max(slab2, axis=axis_idx)
    c3 = np.max(slab3, axis=axis_idx)

    # Stack to RGB
    img = np.stack([c1, c2, c3], axis=-1)

    return img


def resize_image(image, size):
    """
    Resizes image to target resolution.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def process_single_patient(
    patient_id, dicom_rel_path, output_dir, load_cached, input_root
):
    """
    Worker function to process one patient.
    """
    axial_path = os.path.join(output_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(output_dir, f"{patient_id}_coronal.npy")

    # Check cache
    if load_cached and os.path.exists(axial_path) and os.path.exists(coronal_path):
        return True

    full_path = os.path.join(input_root, dicom_rel_path)

    try:
        # Load
        volume = load_scan(full_path)
        if volume is None:
            raise ValueError("Empty or invalid volume")

        # Normalize
        volume = normalize_volume(volume)

        # Generate Axial (Axis 0)
        axial_img = generate_tri_slab(volume, axis_idx=0)
        axial_img = resize_image(axial_img, Config.IMAGE_SIZE)

        # Generate Coronal (Axis 1)
        coronal_img = generate_tri_slab(volume, axis_idx=1)
        coronal_img = resize_image(coronal_img, Config.IMAGE_SIZE)

        # Save
        np.save(axial_path, axial_img)
        np.save(coronal_path, coronal_img)
        return True

    except Exception as e:
        # Create a blank image as fallback to prevent pipeline failure
        blank = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        np.save(axial_path, blank)
        np.save(coronal_path, blank)
        # print(f"Error processing {patient_id}: {e}")
        return False


def cache_images(metadata_path, load_cached_data=True):
    """
    Main entry point to process and cache images for a dataset.

    Args:
        metadata_path: Path to the csv file containing patient info.
        load_cached_data: If True, skips processing if files exist.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    df = pd.read_csv(metadata_path)

    # Get unique patients and their directories
    unique_patients = df[["Patient", "dicom_dir"]].drop_duplicates()

    # Process in parallel
    # We use the config number of workers
    Parallel(n_jobs=Config.NUM_WORKERS)(
        delayed(process_single_patient)(
            row["Patient"],
            row["dicom_dir"],
            Config.CACHE_DIR,
            load_cached_data,
            Config.INPUT_DIR,
        )
        for _, row in unique_patients.iterrows()
    )
