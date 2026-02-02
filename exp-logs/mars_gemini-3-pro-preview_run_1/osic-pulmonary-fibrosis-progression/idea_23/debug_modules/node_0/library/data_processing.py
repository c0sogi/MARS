import os
import numpy as np
import cv2
import pydicom
from library.config import Config


def read_dicom_volume(dicom_dir):
    """
    Reads a directory of DICOM files and constructs a 3D volume.
    Returns: numpy array of shape (Z, Y, X) in Hounsfield Units.
    """
    if not os.path.exists(dicom_dir):
        return None

    # List all .dcm files
    files = [
        os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")
    ]
    if not files:
        return None

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            # Ensure we have pixel data
            if hasattr(dcm, "pixel_array"):
                slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return None

    # Sort by ImagePositionPatient Z coordinate (index 2)
    # If ImagePositionPatient is missing, fall back to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Stack images and convert to Hounsfield Units
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if present
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)

        if slope != 1:
            img = slope * img.astype(np.float64)
            img = img.astype(np.float32)

        img += intercept
        images.append(img)

    # Stack to create (Z, Y, X) volume
    volume = np.stack(images)
    return volume


def generate_tri_slabs(volume, axis, img_size=224, overlap_ratio=0.15):
    """
    Generates a 3-channel image using Maximum Intensity Projection (MIP)
    over 3 overlapping slabs along the specified axis.

    Args:
        volume: 3D numpy array (Z, Y, X)
        axis: 0 for Axial (Z), 1 for Coronal (Y)
        img_size: Output spatial resolution
        overlap_ratio: Percentage of dimension to overlap between slabs

    Returns:
        (H, W, 3) numpy array, uint8, 0-255
    """
    if volume is None or volume.size == 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Determine the size of the dimension we are splitting
    dim_size = volume.shape[axis]

    # Define split points: 0, 1/3, 2/3, 1
    splits = np.linspace(0, dim_size, 4)
    overlap_pixels = int(dim_size * overlap_ratio)

    # Define slab boundaries with overlap
    # Slab 0: 0 -> 1/3 + overlap
    s0_start = 0
    s0_end = int(min(splits[1] + overlap_pixels, dim_size))

    # Slab 1: 1/3 - overlap -> 2/3 + overlap
    s1_start = int(max(splits[1] - overlap_pixels, 0))
    s1_end = int(min(splits[2] + overlap_pixels, dim_size))

    # Slab 2: 2/3 - overlap -> 1
    s2_start = int(max(splits[2] - overlap_pixels, 0))
    s2_end = int(dim_size)

    slabs_indices = [(s0_start, s0_end), (s1_start, s1_end), (s2_start, s2_end)]

    channels = []
    for start, end in slabs_indices:
        # Handle degenerate cases where start >= end (e.g., very small volumes)
        if start >= end:
            # Create an empty slice of correct shape
            if axis == 0:
                # Result shape (Y, X)
                slab_mip = np.zeros(
                    (volume.shape[1], volume.shape[2]), dtype=np.float32
                )
            else:
                # Result shape (Z, X)
                slab_mip = np.zeros(
                    (volume.shape[0], volume.shape[2]), dtype=np.float32
                )
        else:
            # Slice the volume along the specified axis
            if axis == 0:
                slab = volume[start:end, :, :]
            else:
                slab = volume[:, start:end, :]

            # Compute Maximum Intensity Projection (MIP)
            if slab.shape[axis] > 0:
                slab_mip = np.max(slab, axis=axis)
            else:
                # Fallback if slice is somehow empty
                if axis == 0:
                    slab_mip = np.zeros(
                        (volume.shape[1], volume.shape[2]), dtype=np.float32
                    )
                else:
                    slab_mip = np.zeros(
                        (volume.shape[0], volume.shape[2]), dtype=np.float32
                    )

        # Windowing and Normalization
        # Standard Lung Window: [-1000, 400] HU
        # -1000 is air, 400 is bone/dense tissue.
        # This range covers lung parenchyma (-700 to -600) and abnormalities.
        min_hu = -1000.0
        max_hu = 400.0

        slab_mip = np.clip(slab_mip, min_hu, max_hu)

        # Normalize to 0-1
        slab_mip = (slab_mip - min_hu) / (max_hu - min_hu)

        # Resize to target resolution
        # cv2.resize expects (width, height)
        resized = cv2.resize(
            slab_mip, (img_size, img_size), interpolation=cv2.INTER_LINEAR
        )

        channels.append(resized)

    # Stack channels to form (H, W, 3) image
    merged = np.stack(channels, axis=-1)

    # Convert to uint8 [0, 255]
    merged = (merged * 255).astype(np.uint8)

    return merged


def process_patient(patient_id, dicom_dir, load_cached_data=True):
    """
    Main processing function for a single patient.
    Generates Axial and Coronal Tri-Slab images.
    Implements caching to disk to save processing time on subsequent runs.

    Args:
        patient_id: Unique patient identifier
        dicom_dir: Path to the directory containing DICOM files
        load_cached_data: If True, attempts to load from cache first

    Returns:
        Tuple (axial_img, coronal_img), both numpy arrays of shape (224, 224, 3)
    """
    # Ensure cache directory exists
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(axial_path) and os.path.exists(coronal_path):
            try:
                axial = np.load(axial_path)
                coronal = np.load(coronal_path)
                return axial, coronal
            except Exception:
                # If load fails (corrupt file), proceed to re-process
                pass

    # 2. Process from scratch
    volume = read_dicom_volume(dicom_dir)

    # Generate Axial Tri-Slab (Axis 0 = Z)
    axial = generate_tri_slabs(
        volume, axis=0, img_size=Config.IMG_SIZE, overlap_ratio=Config.SLAB_OVERLAP
    )

    # Generate Coronal Tri-Slab (Axis 1 = Y)
    coronal = generate_tri_slabs(
        volume, axis=1, img_size=Config.IMG_SIZE, overlap_ratio=Config.SLAB_OVERLAP
    )

    # 3. Save to cache
    try:
        np.save(axial_path, axial)
        np.save(coronal_path, coronal)
    except Exception as e:
        print(f"Warning: Failed to save cache for {patient_id}: {e}")

    return axial, coronal
