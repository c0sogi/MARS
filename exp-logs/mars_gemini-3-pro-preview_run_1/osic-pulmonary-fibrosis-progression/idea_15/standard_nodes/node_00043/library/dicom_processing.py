import os
import numpy as np
import cv2
import pydicom
from library.config import Config


def get_slab_indices(total_size, n_slabs=3, overlap_ratio=0.15):
    """
    Calculates start and end indices for overlapping slabs along an axis.

    Args:
        total_size (int): Total number of slices/pixels along the axis.
        n_slabs (int): Number of slabs to generate.
        overlap_ratio (float): Fraction of the slab size to overlap.

    Returns:
        list of tuples: [(start_idx, end_idx), ...]
    """
    if total_size == 0:
        return []

    slab_depth = total_size / n_slabs
    margin = slab_depth * overlap_ratio

    indices = []
    for i in range(n_slabs):
        # Base boundaries
        start_base = i * slab_depth
        end_base = (i + 1) * slab_depth

        # Add margin (overlap)
        # We clamp indices to [0, total_size]
        start = max(0, int(start_base - margin))
        end = min(total_size, int(end_base + margin))

        # Ensure at least 1 slice is selected
        if end <= start:
            end = start + 1

        indices.append((start, end))

    return indices


def load_scan(path):
    """
    Loads DICOM files from a directory, sorts them by InstanceNumber,
    and converts to Hounsfield Units (HU).

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        np.array: 3D volume (Depth, Height, Width) in HU, or None if failed.
    """
    if not os.path.exists(path):
        return None

    # List all dcm files
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return None

    # Read DICOMs
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None

    # Sort by InstanceNumber (Z-position)
    # Fallback to ImagePositionPatient Z coordinate if InstanceNumber is missing
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            pass  # Keep file system order if sorting fails

    # Extract images and Convert to HU
    images = []
    for s in slices:
        try:
            img = s.pixel_array.astype(np.float32)

            # Convert to HU: pixel * slope + intercept
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)

            if slope != 1:
                img = slope * img.astype(np.float64)
                img = img.astype(np.float32)

            img += np.float32(intercept)
            images.append(img)
        except Exception:
            continue

    if not images:
        return None

    # Stack to 3D volume (D, H, W)
    volume = np.stack(images, axis=0)

    # Handle padding/artifacts (e.g. circular CT field of view padding often -2000)
    # Clip values < -1000 (Air) to -1000 to remove background noise
    volume[volume < -1000] = -1000

    return volume


def normalize_and_resize(img, target_size=(224, 224)):
    """
    Clips HU to a fixed physical range, normalizes to [0, 255], and resizes.

    Args:
        img (np.array): 2D image array in HU.
        target_size (tuple): (width, height).

    Returns:
        np.array: uint8 image normalized and resized.
    """
    # Clip to physical range (Air to Bone/Tissue)
    # -1000 HU is Air, 400 HU covers bone and soft tissue.
    # We use a fixed window to preserve physical density signal.
    min_hu = -1000
    max_hu = 400

    img = np.clip(img, min_hu, max_hu)

    # Normalize to 0-1
    img = (img - min_hu) / (max_hu - min_hu)

    # Resize
    # cv2.resize expects (width, height)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert to 0-255 uint8
    img_uint8 = (img_resized * 255).astype(np.uint8)

    return img_uint8


def generate_dual_view_tri_slabs(patient_id, dicom_dir, load_cached_data=True):
    """
    Generates Axial and Coronal Tri-Slab inputs for a patient.

    Logic:
    1. Checks cache for processed .npy files.
    2. If not found, loads DICOM volume.
    3. Generates Axial Tri-Slab (Split Z-axis, MIP).
    4. Generates Coronal Tri-Slab (Split Y-axis, MIP).
    5. Resizes to 224x224 and saves to cache.

    Args:
        patient_id (str): The patient ID.
        dicom_dir (str): Full path to the directory containing DICOM files.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (axial_img, coronal_img)
            - axial_img: np.array of shape (224, 224, 3)
            - coronal_img: np.array of shape (224, 224, 3)
    """
    cache_dir = Config.CACHE_DIR
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Load Cache
    if load_cached_data:
        if os.path.exists(axial_path) and os.path.exists(coronal_path):
            try:
                axial_img = np.load(axial_path)
                coronal_img = np.load(coronal_path)
                return axial_img, coronal_img
            except Exception:
                # If load fails, proceed to recompute
                pass

    # 2. Compute from Scratch
    # Initialize empty (black) images in case of failure/empty volume
    axial_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
    coronal_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    volume = load_scan(dicom_dir)

    if volume is not None and volume.shape[0] > 0:
        D, H, W = volume.shape

        # --- Axial Tri-Slab (Split along D/Z axis) ---
        axial_channels = []
        indices_z = get_slab_indices(D, n_slabs=3, overlap_ratio=Config.SLAB_OVERLAP)

        for start, end in indices_z:
            slab = volume[start:end, :, :]
            if slab.shape[0] > 0:
                mip = np.max(slab, axis=0)  # MIP along Z -> (H, W)
            else:
                mip = np.zeros((H, W), dtype=np.float32) - 1000

            axial_channels.append(
                normalize_and_resize(mip, (Config.IMG_SIZE, Config.IMG_SIZE))
            )

        axial_img = np.stack(axial_channels, axis=-1)  # (224, 224, 3)

        # --- Coronal Tri-Slab (Split along H/Y axis) ---
        # Coronal view corresponds to the (D, W) plane. We split along H (Y-axis).
        coronal_channels = []
        indices_y = get_slab_indices(H, n_slabs=3, overlap_ratio=Config.SLAB_OVERLAP)

        for start, end in indices_y:
            slab = volume[:, start:end, :]
            if slab.shape[1] > 0:
                mip = np.max(slab, axis=1)  # MIP along Y -> (D, W)
            else:
                mip = np.zeros((D, W), dtype=np.float32) - 1000

            coronal_channels.append(
                normalize_and_resize(mip, (Config.IMG_SIZE, Config.IMG_SIZE))
            )

        coronal_img = np.stack(coronal_channels, axis=-1)  # (224, 224, 3)

    # 3. Save Cache
    try:
        np.save(axial_path, axial_img)
        np.save(coronal_path, coronal_img)
    except Exception as e:
        # Non-critical failure, just print warning
        print(f"Warning: Failed to save cache for {patient_id}: {e}")

    return axial_img, coronal_img
