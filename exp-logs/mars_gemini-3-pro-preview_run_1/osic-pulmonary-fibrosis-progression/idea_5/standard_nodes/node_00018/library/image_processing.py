import os
import numpy as np
import pydicom
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_scan(path):
    """
    Loads DICOM files from a directory and sorts them by InstanceNumber.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                # Check for necessary attributes
                if hasattr(ds, "InstanceNumber"):
                    slices.append(ds)
            except Exception:
                continue

    if not slices:
        # Fallback: Create a dummy volume if no valid dicoms found (should not happen based on EDA)
        # Return empty list to be handled by caller
        return []

    # Sort by InstanceNumber to ensure correct Z-ordering
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel_array to Hounsfield Units (HU).
    Handles slope and intercept.
    """
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (standard for HU)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately -1000
    image[image == -2000] = 0

    # Convert to HU
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def window_image(img, window_center=-600, window_width=1500):
    """
    Applies a lung window to the HU image and normalizes to [0, 1].
    Standard Lung Window: Center=-600, Width=1500 (Range: -1350 to 150)
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    # Normalize to [0, 1]
    img = (img - img_min) / (img_max - img_min)
    return img


def resize_image(img, size=(224, 224)):
    """
    Resizes image to target dimensions using Linear interpolation.
    Input shape: (H, W, C) or (H, W)
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def get_tri_slab_mip(volume, overlap_percent=0.15):
    """
    Splits the volume depth-wise into 3 overlapping slabs and computes MIP.
    Input volume shape: (Depth, H, W)
    Output shape: (H, W, 3) (Channels: Top, Middle, Bottom)
    """
    depth = volume.shape[0]

    # Handle small depth
    if depth < 3:
        # If depth is too small, just repeat the max projection
        mip = np.max(volume, axis=0)
        return np.stack([mip, mip, mip], axis=-1)

    # Calculate slab boundaries
    # We want 3 slabs covering the range [0, depth]
    # Base chunk size
    chunk_size = depth / 3.0

    # Overlap amount (in pixels)
    overlap = int(chunk_size * overlap_percent)

    # Define start and end indices for 3 slabs
    # Slab 0: Top
    s0_start = 0
    s0_end = int(chunk_size + overlap)
    s0_end = min(s0_end, depth)

    # Slab 1: Middle
    s1_start = int(chunk_size - overlap)
    s1_start = max(0, s1_start)
    s1_end = int(2 * chunk_size + overlap)
    s1_end = min(s1_end, depth)

    # Slab 2: Bottom
    s2_start = int(2 * chunk_size - overlap)
    s2_start = max(0, s2_start)
    s2_end = depth

    # Extract slabs
    slab0 = volume[s0_start:s0_end, :, :]
    slab1 = volume[s1_start:s1_end, :, :]
    slab2 = volume[s2_start:s2_end, :, :]

    # Compute MIP (Maximum Intensity Projection) along depth axis
    # Handle case where slab might be empty due to rounding (unlikely with logic above)
    mip0 = np.max(slab0, axis=0) if slab0.shape[0] > 0 else np.zeros_like(volume[0])
    mip1 = np.max(slab1, axis=0) if slab1.shape[0] > 0 else np.zeros_like(volume[0])
    mip2 = np.max(slab2, axis=0) if slab2.shape[0] > 0 else np.zeros_like(volume[0])

    # Stack into RGB channels
    # Shape: (H, W, 3)
    img = np.stack([mip0, mip1, mip2], axis=-1)

    return img


def generate_dual_views(dicom_path, patient_id, load_cached_data=True):
    """
    Main processing function.
    1. Checks cache for {patient_id}.npy
    2. Loads DICOM scan
    3. Generates Axial Tri-Slab MIP
    4. Generates Coronal Tri-Slab MIP
    5. Resizes and stacks

    Returns:
        numpy array of shape (2, 224, 224, 3)
        Index 0: Axial View
        Index 1: Coronal View
        Values: float32 in [0, 1]
    """
    # Define cache paths
    cache_dir = "./working/idea_5"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{patient_id}.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            if data.shape == (2, 224, 224, 3):
                return data
        except Exception:
            # If load fails, recompute
            pass

    # 2. Load and Process Scan
    try:
        scans = load_scan(dicom_path)
        if not scans:
            # Return zero array if scan load fails
            return np.zeros((2, 224, 224, 3), dtype=np.float32)

        vol_hu = get_pixels_hu(scans)  # Shape: (Z, Y, X)

        # 3. Axial View (Z is depth)
        # Input: (Z, Y, X) -> Output: (Y, X, 3)
        axial_img = get_tri_slab_mip(vol_hu, overlap_percent=0.15)
        axial_img = window_image(axial_img)
        axial_img = resize_image(axial_img, size=(224, 224))

        # 4. Coronal View (Y is depth)
        # Transpose to (Y, Z, X) so Y becomes the first dimension (depth)
        vol_coronal = np.transpose(vol_hu, (1, 0, 2))
        coronal_img = get_tri_slab_mip(vol_coronal, overlap_percent=0.15)
        coronal_img = window_image(coronal_img)
        coronal_img = resize_image(coronal_img, size=(224, 224))

        # 5. Stack
        # Shape: (2, 224, 224, 3)
        final_data = np.stack([axial_img, coronal_img], axis=0).astype(np.float32)

        # 6. Save to Cache
        np.save(cache_file, final_data)

        return final_data

    except Exception as e:
        # Fallback for any processing error
        # print(f"Error processing {patient_id}: {e}")
        return np.zeros((2, 224, 224, 3), dtype=np.float32)
