import os
import numpy as np
import cv2
import pydicom
from library.config import Config


def load_scan(path):
    """
    Loads DICOM scans from a directory and sorts them by InstanceNumber.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        list: A list of pydicom datasets sorted by InstanceNumber.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    slices = []
    for s in os.listdir(path):
        if s.lower().endswith(".dcm"):
            try:
                full_path = os.path.join(path, s)
                ds = pydicom.dcmread(full_path)
                slices.append(ds)
            except Exception as e:
                # Skip unreadable files
                continue

    if not slices:
        # Return empty list or raise error.
        # Raising error is safer to detect data issues.
        raise ValueError(f"No valid DICOM files found in {path}")

    # Sort by InstanceNumber.
    # Some files might not have InstanceNumber, handle gracefully or assume they do.
    slices.sort(
        key=lambda x: int(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
    )
    return slices


def get_pixels_hu(scans):
    """
    Converts raw pixel data to Hounsfield Units (HU) and applies thresholding
    to mask lung tissue.

    Args:
        scans (list): List of pydicom datasets.

    Returns:
        np.ndarray: 3D numpy array of HU values clipped to the lung window.
    """
    # Stack the pixel arrays
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.float32)

    # Convert to Hounsfield Units (HU)
    # HU = pixel_value * slope + intercept
    # We assume the slope and intercept are consistent across the scan
    intercept = (
        scans[0].RescaleIntercept if hasattr(scans[0], "RescaleIntercept") else -1024
    )
    slope = scans[0].RescaleSlope if hasattr(scans[0], "RescaleSlope") else 1

    if slope != 1:
        image = slope * image

    image += intercept

    # Handle Pixel Padding Value if present (set to min HU)
    # (Optional refinement, but often padding is -2000 or similar)
    if hasattr(scans[0], "PixelPaddingValue"):
        padding_val = scans[0].PixelPaddingValue
        image[image == padding_val] = Config.HU_MIN

    # Apply Thresholding / Masking
    # Clip values to the defined lung window [HU_MIN, HU_MAX]
    # This effectively masks air (< -1000) and bone/dense tissue (> -400)
    # by clamping them to the boundaries.
    image = np.clip(image, Config.HU_MIN, Config.HU_MAX)

    return image


def generate_orthogonal_mips(volume):
    """
    Generates Maximum Intensity Projections along Axial, Coronal, and Sagittal axes.

    Args:
        volume (np.ndarray): 3D array of HU values (Z, Y, X).

    Returns:
        list: A list of 3 2D numpy arrays [Axial_MIP, Coronal_MIP, Sagittal_MIP].
    """
    # Axial (Top-down): Max projection over Z-axis (axis 0) -> Shape (Y, X)
    mip_axial = np.max(volume, axis=0)

    # Coronal (Front-back): Max projection over Y-axis (axis 1) -> Shape (Z, X)
    mip_coronal = np.max(volume, axis=1)

    # Sagittal (Side-side): Max projection over X-axis (axis 2) -> Shape (Z, Y)
    mip_sagittal = np.max(volume, axis=2)

    return [mip_axial, mip_coronal, mip_sagittal]


def select_variance_slices(volume):
    """
    Selects the top N axial slices with the highest pixel variance.
    This helps capture texture heterogeneity (e.g., fibrosis).

    Args:
        volume (np.ndarray): 3D array of HU values (Z, Y, X).

    Returns:
        list: A list of 2D numpy arrays corresponding to the selected slices.
    """
    # Calculate variance for each slice along spatial dimensions (Y, X)
    # axis=(1, 2) computes variance of the 2D image
    slice_variances = np.var(volume, axis=(1, 2))

    # Get indices of the slices with highest variance
    # argsort sorts in ascending order, so we take the last N and reverse
    top_indices = np.argsort(slice_variances)[-Config.NUM_TOP_VARIANCE_SLICES :][::-1]

    # Handle edge case where volume has fewer slices than requested
    if len(top_indices) < Config.NUM_TOP_VARIANCE_SLICES:
        # Pad with the best slice (index 0 of top_indices)
        best_idx = top_indices[0] if len(top_indices) > 0 else 0
        current_count = len(top_indices)
        needed = Config.NUM_TOP_VARIANCE_SLICES - current_count
        padding = [best_idx] * needed
        top_indices = np.concatenate([top_indices, padding]).astype(int)

    selected_slices = [volume[i] for i in top_indices]
    return selected_slices


def resize_and_normalize(image):
    """
    Resizes an image to the target size and normalizes values to [0, 1].

    Args:
        image (np.ndarray): 2D numpy array.

    Returns:
        np.ndarray: Resized and normalized image of type float32.
    """
    # Resize to target dimensions
    # cv2.resize expects (Width, Height)
    img_resized = cv2.resize(
        image, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )

    # Normalize to [0, 1] based on the known HU range
    min_val = Config.HU_MIN
    max_val = Config.HU_MAX

    # Linear scaling
    img_norm = (img_resized - min_val) / (max_val - min_val)

    # Clip to ensure bounds
    img_norm = np.clip(img_norm, 0.0, 1.0)

    return img_norm.astype(np.float32)


def process_patient(patient_id, dcm_rel_path, load_cached_data=True):
    """
    Orchestrates the processing pipeline for a single patient.
    Loads DICOMs, processes volume, extracts features (MIPs + Slices),
    resizes/normalizes, and caches the result.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_rel_path (str): Relative path to the DICOM directory (e.g., 'train/ID...').
        load_cached_data (bool): If True, attempts to load from disk cache first.

    Returns:
        np.ndarray: Array of shape (6, IMG_SIZE, IMG_SIZE) containing processed images.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            if data.shape == (
                Config.TOTAL_IMAGES_PER_PATIENT,
                Config.IMG_SIZE,
                Config.IMG_SIZE,
            ):
                return data
        except Exception:
            # If load fails, proceed to re-process
            pass

    # 2. Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, dcm_rel_path)

    try:
        # Load and convert to HU
        scans = load_scan(full_path)
        volume = get_pixels_hu(scans)

        # Generate Features
        mips = generate_orthogonal_mips(volume)  # List of 3
        slices = select_variance_slices(volume)  # List of 3

        # Combine into a single list of 6 images
        raw_images = mips + slices

        # Resize and Normalize
        processed_images = np.array([resize_and_normalize(img) for img in raw_images])

        # 3. Save to cache
        np.save(cache_path, processed_images)

        return processed_images

    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        # Return a zero array to prevent pipeline crash, but log error
        return np.zeros(
            (Config.TOTAL_IMAGES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )
