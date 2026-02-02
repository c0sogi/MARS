import os
import numpy as np
import cv2
import pydicom
from library.config import Config


def load_dicom_volume(dicom_dir):
    """
    Reads a directory of DICOM files, sorts them by slice location,
    and returns the volume as a numpy array in Hounsfield Units.
    """
    if not os.path.exists(dicom_dir):
        return None

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return None

    # Read all files
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, f))
            # Ensure we have the necessary attributes for sorting and conversion
            if not hasattr(ds, "pixel_array"):
                continue
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None

    # Sort by ImagePositionPatient Z coordinate
    # Fallback to InstanceNumber if spatial coords are missing
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(
            key=lambda x: (
                int(x.InstanceNumber) if hasattr(x, "InstanceNumber") else x.filename
            )
        )

    # Stack images
    try:
        # Get slope and intercept for HU conversion
        # We assume they are constant across the scan, taking from the first slice
        slope = getattr(slices[0], "RescaleSlope", 1)
        intercept = getattr(slices[0], "RescaleIntercept", 0)

        # Stack pixel arrays
        image_stack = np.stack([s.pixel_array.astype(np.float32) for s in slices])

        # Convert to Hounsfield Units (HU)
        image_stack = image_stack * slope + intercept

        return image_stack
    except Exception as e:
        print(f"Error processing DICOM volume at {dicom_dir}: {e}")
        return None


def normalize_volume(volume):
    """
    Clips HU values to lung window [-1000, 400] and normalizes to [0, 1].
    """
    min_hu = -1000.0
    max_hu = 400.0
    volume = np.clip(volume, min_hu, max_hu)
    volume = (volume - min_hu) / (max_hu - min_hu)
    return volume


def generate_tri_slab(volume, axis="axial", overlap=0.15):
    """
    Generates a 3-channel image using MIP over 3 overlapping slabs along the specified axis.
    axis: 'axial' (split Z) or 'coronal' (split Y).
    """
    # volume shape: (Z, Y, X)
    if axis == "axial":
        # Split along Z (dim 0)
        split_axis = 0
    elif axis == "coronal":
        # Split along Y (dim 1)
        split_axis = 1
    else:
        raise ValueError("Axis must be 'axial' or 'coronal'")

    # Move split axis to 0 for uniform processing
    vol = np.moveaxis(volume, split_axis, 0)
    depth = vol.shape[0]

    # Calculate slab boundaries
    # We create 3 slabs.
    # Chunk size is Depth / 3.
    # Overlap is calculated as a fraction of the chunk size.
    num_slabs = 3
    chunk_size = depth / num_slabs
    overlap_px = chunk_size * overlap

    channels = []
    for i in range(num_slabs):
        # Calculate base start/end
        start_exact = i * chunk_size
        end_exact = (i + 1) * chunk_size

        # Extend by overlap to create the slab
        start = int(max(0, start_exact - overlap_px))
        end = int(min(depth, end_exact + overlap_px))

        # Handle edge case of very thin volume
        if start >= end:
            start = max(0, depth - 1)
            end = depth

        slab = vol[start:end, :, :]

        # Maximum Intensity Projection (MIP) along the depth of the slab
        if slab.shape[0] > 0:
            mip = np.max(slab, axis=0)
        else:
            # Fallback for empty slab
            mip = np.zeros((vol.shape[1], vol.shape[2]), dtype=np.float32)

        channels.append(mip)

    # Stack the 3 MIPs into channels (H, W, 3)
    img = np.stack(channels, axis=-1)
    return img


def resize_image(image, size=240):
    """
    Resizes the image to the target square size (size, size).
    """
    # cv2.resize expects (W, H). Image is (H, W, C).
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def process_patient(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Orchestrates the loading, processing, and caching of patient data.
    Returns (axial_img, coronal_img) as numpy arrays (size, size, 3).
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    axial_file = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_file = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached:
        if os.path.exists(axial_file) and os.path.exists(coronal_file):
            try:
                axial = np.load(axial_file)
                coronal = np.load(coronal_file)
                return axial, coronal
            except Exception:
                # If load fails (corrupt file), proceed to recompute
                pass

    # 2. Process from scratch
    volume = load_dicom_volume(dicom_dir)

    # Handle failure to load volume (e.g., missing files)
    if volume is None:
        # Return blank images
        blank = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return blank, blank

    # Normalize HU values
    volume = normalize_volume(volume)

    # Generate Axial Tri-Slab (Z-axis split)
    axial = generate_tri_slab(volume, axis="axial", overlap=Config.SLAB_OVERLAP)
    axial = resize_image(axial, size=Config.IMG_SIZE)

    # Generate Coronal Tri-Slab (Y-axis split)
    coronal = generate_tri_slab(volume, axis="coronal", overlap=Config.SLAB_OVERLAP)
    coronal = resize_image(coronal, size=Config.IMG_SIZE)

    # 3. Save to cache
    try:
        np.save(axial_file, axial)
        np.save(coronal_file, coronal)
    except Exception as e:
        print(f"Warning: Could not save cache for {patient_id}: {e}")

    return axial, coronal
