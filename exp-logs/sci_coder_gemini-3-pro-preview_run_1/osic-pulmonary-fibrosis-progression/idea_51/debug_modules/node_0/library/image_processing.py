import os
import numpy as np
import cv2
import pydicom
from library.config import Config


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them, and converts to Hounsfield Units.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        np.ndarray: 3D volume (Depth, Height, Width) in Hounsfield Units, or None if failed.
    """
    if not os.path.exists(path):
        return None

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return None

    # Read DICOMs
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(path, f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None

    # Sort logic: InstanceNumber -> ImagePositionPatient Z -> Filename
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(os.path.basename(x.filename).split(".")[0]))

    # Extract images and convert to HU
    images = []
    for s in slices:
        # Convert to float32
        img_2d = s.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units (HU)
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)

        if slope != 1:
            img_2d = slope * img_2d

        img_2d += intercept
        images.append(img_2d)

    # Stack into 3D volume (Depth, Height, Width)
    volume = np.stack(images)
    return volume


def normalize_hu(volume):
    """
    Clips to lung window and normalizes to [0, 1].
    Range: [-1000, 400] covers lung parenchyma and soft tissue.
    """
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0

    volume = (volume - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    volume = np.clip(volume, 0, 1)
    return volume


def generate_tri_slab(volume, axis="axial"):
    """
    Generates a 3-channel image using Fixed Overlapping Orthogonal Tri-Slabs.

    Args:
        volume: 3D numpy array (Depth, Height, Width)
        axis: 'axial' or 'coronal'

    Returns:
        2D numpy array (H, W, 3) normalized to [0, 1]
    """
    # Handle Axis Permutation
    # Axial: (Depth, Height, Width) -> Slice along Depth (axis 0)
    # Coronal: (Depth, Height, Width) -> Slice along Height (axis 1)
    # To unify logic, we transpose Coronal to (Height, Depth, Width) so we always slice axis 0

    if axis == "coronal":
        volume = np.transpose(volume, (1, 0, 2))

    n_slices = volume.shape[0]

    # Define slab boundaries
    # We divide the volume into 3 base segments.
    # Overlap is calculated as a fraction of the segment size (slab size).
    slab_size = n_slices / 3.0
    overlap_px = int(slab_size * Config.SLAB_OVERLAP)

    # Slab 1: 0 to 1/3 + overlap
    s1_start = 0
    s1_end = int(slab_size + overlap_px)

    # Slab 2: 1/3 - overlap to 2/3 + overlap
    s2_start = int(slab_size - overlap_px)
    s2_end = int(2 * slab_size + overlap_px)

    # Slab 3: 2/3 - overlap to end
    s3_start = int(2 * slab_size - overlap_px)
    s3_end = n_slices

    # Clamp indices to valid range
    s1_end = min(s1_end, n_slices)
    s2_start = max(0, s2_start)
    s2_end = min(s2_end, n_slices)
    s3_start = max(0, s3_start)

    # Helper for MIP
    def get_mip(start, end):
        if start >= end:
            # Handle empty slab case (e.g. single slice volume)
            return np.zeros((volume.shape[1], volume.shape[2]), dtype=np.float32)
        slab = volume[start:end, :, :]
        return np.max(slab, axis=0)

    mip1 = get_mip(s1_start, s1_end)
    mip2 = get_mip(s2_start, s2_end)
    mip3 = get_mip(s3_start, s3_end)

    # Stack to RGB (H_view, W_view, 3)
    img = np.stack([mip1, mip2, mip3], axis=-1)

    # Resize to target size (224, 224)
    # cv2.resize expects (W, H)
    img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

    # Ensure range [0, 1] after interpolation
    img_resized = np.clip(img_resized, 0, 1)

    return img_resized


def process_patient(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Full pipeline: Load -> Process -> Cache -> Return.

    Args:
        patient_id (str): Unique patient identifier.
        dicom_dir (str): Full path to the directory containing the patient's DICOM files.
        cache_dir (str): Directory to save/load .npy files.
        load_cached (bool): If True, attempt to load from cache first.

    Returns:
        dict: {'axial': np.ndarray, 'coronal': np.ndarray}
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    path_axial = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    path_coronal = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Load from Cache
    if load_cached and os.path.exists(path_axial) and os.path.exists(path_coronal):
        try:
            return {"axial": np.load(path_axial), "coronal": np.load(path_coronal)}
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from Scratch
    volume = load_scan(dicom_dir)

    if volume is None:
        # Fallback for missing/corrupt data: return black images
        black_img = np.zeros(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.float32
        )
        return {"axial": black_img, "coronal": black_img}

    # Normalize HU once
    volume = normalize_hu(volume)

    # Generate Views
    img_axial = generate_tri_slab(volume, axis="axial")
    img_coronal = generate_tri_slab(volume, axis="coronal")

    # 3. Save to Cache
    try:
        np.save(path_axial, img_axial)
        np.save(path_coronal, img_coronal)
    except Exception as e:
        print(f"Warning: Failed to save cache for {patient_id}: {e}")

    return {"axial": img_axial, "coronal": img_coronal}
