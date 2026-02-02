import os
import numpy as np
import cv2
import pydicom
from library.utils import get_logger

# Configure logger
logger = get_logger("dicom_processing")


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by slice location.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        list: List of pydicom datasets sorted by Z-position.
    """
    if not os.path.exists(path):
        logger.error(f"Directory not found: {path}")
        return []

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception as e:
                logger.warning(f"Failed to read DICOM {s}: {e}")

    if not slices:
        logger.warning(f"No DICOM files found in {path}")
        return []

    # Sort by ImagePositionPatient[2] (Z-coordinate) if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            logger.warning(
                "Could not sort slices by Z-pos or InstanceNumber. Using filename."
            )
            slices.sort(key=lambda x: x.filename)

    return slices


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).
    Handles slope, intercept, and padding.

    Args:
        scans (list): List of pydicom datasets.

    Returns:
        np.array: 3D numpy array of HU values (Z, Y, X).
    """
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately -1000
    # Common padding value in DICOM is -2000
    image[image == -2000] = 0

    # Convert to Hounsfield Units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def _get_tri_slab_mip(volume, axis, target_size=(224, 224)):
    """
    Internal helper to generate a Tri-Slab MIP image along a specific axis.

    Args:
        volume: 3D numpy array (Z, Y, X)
        axis: The axis to slice along (0 for Axial/Z, 1 for Coronal/Y)
        target_size: (H, W) for resizing

    Returns:
        np.array: (224, 224, 3) normalized RGB image
    """
    # Dimensions
    depth = volume.shape[axis]

    # Define slab boundaries (0-40%, 30-70%, 60-100%)
    # This provides overlap relative to simple thirds
    starts = [0.0, 0.30, 0.60]
    ends = [0.40, 0.70, 1.00]

    channels = []

    for s, e in zip(starts, ends):
        idx_start = int(depth * s)
        idx_end = int(depth * e)

        # Ensure at least one slice
        idx_end = max(idx_end, idx_start + 1)
        idx_end = min(idx_end, depth)

        # Slice volume
        if axis == 0:
            # Slicing along Z (Axial View)
            slab = volume[idx_start:idx_end, :, :]
            # MIP along Z -> Result is (Y, X)
            mip = np.max(slab, axis=0)
        elif axis == 1:
            # Slicing along Y (Coronal View)
            slab = volume[:, idx_start:idx_end, :]
            # MIP along Y -> Result is (Z, X)
            mip = np.max(slab, axis=1)
        else:
            raise ValueError("Axis must be 0 (Axial) or 1 (Coronal)")

        channels.append(mip)

    # Stack to RGB (3, H, W) -> Transpose to (H, W, 3)
    img = np.stack(channels, axis=-1)

    # Resize
    # cv2.resize expects (W, H)
    img_resized = cv2.resize(img.astype(np.float32), target_size)

    # Normalize to Lung Window [-1000, 400]
    min_hu = -1000.0
    max_hu = 400.0

    img_resized = (img_resized - min_hu) / (max_hu - min_hu)
    img_resized = np.clip(img_resized, 0.0, 1.0)

    return img_resized


def generate_orthogonal_tri_slabs(patient_id, patient_dir, load_cached_data=True):
    """
    Generates Axial and Coronal Tri-Slab MIP images for a patient.

    Args:
        patient_id (str): Unique Patient ID.
        patient_dir (str): Path to the directory containing DICOM files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {'axial': np.array, 'coronal': np.array}
              Images are (224, 224, 3) float32 in range [0, 1].
    """
    cache_dir = "./working/idea_20"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True).item()
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache for {patient_id}: {e}. Recomputing.")

    # 2. Compute from Scratch
    try:
        scans = load_scan(patient_dir)
        if not scans:
            # Return zeros if no scans found
            logger.error(f"No scans found for {patient_id}. Returning zeros.")
            zeros = np.zeros((224, 224, 3), dtype=np.float32)
            return {"axial": zeros, "coronal": zeros}

        vol_hu = get_pixels_hu(scans)

        # Generate Axial (Z-axis slice)
        axial_img = _get_tri_slab_mip(vol_hu, axis=0)

        # Generate Coronal (Y-axis slice)
        coronal_img = _get_tri_slab_mip(vol_hu, axis=1)

        data = {
            "axial": axial_img.astype(np.float32),
            "coronal": coronal_img.astype(np.float32),
        }

        # 3. Save to Cache
        np.save(cache_path, data)

        return data

    except Exception as e:
        logger.error(f"Error processing {patient_id}: {e}")
        # Return zeros on failure to allow pipeline to continue
        zeros = np.zeros((224, 224, 3), dtype=np.float32)
        return {"axial": zeros, "coronal": zeros}
