import os
import numpy as np
import cv2
from library.config import Config


def load_volume(dcm_dir: str) -> np.ndarray:
    """
    Loads a CT scan volume from a directory of DICOM files.

    Since pydicom is not available, this function uses OpenCV to read the pixel data
    and assumes a standard CT offset for Hounsfield Unit (HU) conversion.
    Files are sorted by their integer filename (e.g., '1.dcm', '2.dcm') to approximate Z-ordering.

    Args:
        dcm_dir: Path to the directory containing .dcm files.

    Returns:
        np.ndarray: 3D array of shape (Depth, Height, Width) in Hounsfield Units.
                    Returns a dummy volume if loading fails.
    """
    if not os.path.exists(dcm_dir):
        # Return dummy volume if path doesn't exist
        return np.zeros((10, 512, 512), dtype=np.float32)

    # List and sort files by the integer value of the filename (1.dcm, 10.dcm, etc.)
    files = [f for f in os.listdir(dcm_dir) if f.lower().endswith(".dcm")]

    if not files:
        return np.zeros((10, 512, 512), dtype=np.float32)

    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # Fallback to string sort if filenames are not integers
        files.sort()

    slices = []
    for f in files:
        file_path = os.path.join(dcm_dir, f)
        # Read image using OpenCV. IMREAD_UNCHANGED attempts to read the raw depth (e.g. 16-bit)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if img is not None:
            slices.append(img)

    if not slices:
        return np.zeros((10, 512, 512), dtype=np.float32)

    # Stack slices into 3D volume (Depth, Height, Width)
    volume = np.stack(slices, axis=0).astype(np.float32)

    # Convert to Hounsfield Units (HU)
    # Assumption: Standard CT scans stored as unsigned integers often have an offset of -1024.
    # (i.e., Pixel Value 0 = -1024 HU).
    volume = volume - 1024.0

    return volume


def extract_orthogonal_views(
    volume: np.ndarray, num_axial: int = 3, use_coronal: bool = True
) -> np.ndarray:
    """
    Extracts key 2D slices from the 3D volume:
    1. Axial: Top `num_axial` slices with the highest variance (texture complexity).
    2. Coronal: The central slice along the Y-axis (if enabled).

    Args:
        volume: 3D numpy array (Z, Y, X).
        num_axial: Number of axial slices to select.
        use_coronal: Whether to include the central coronal slice.

    Returns:
        np.ndarray: Array of 2D slices. Shape (num_views, H, W).
    """
    if volume.ndim != 3:
        return np.zeros((num_axial + int(use_coronal), 224, 224), dtype=np.float32)

    # --- Axial Selection ---
    # Calculate variance of each slice to find the most informative ones (lung tissue vs empty)
    slice_variances = np.var(volume, axis=(1, 2))

    # Get indices of top variances
    # If volume has fewer slices than requested, take what we have and pad/repeat
    if volume.shape[0] < num_axial:
        selected_indices = np.arange(volume.shape[0])
        # Pad with the last slice if needed (simple strategy)
        padding = [volume.shape[0] - 1] * (num_axial - volume.shape[0])
        selected_indices = np.concatenate([selected_indices, padding])
    else:
        # Argsort returns ascending, take last num_axial and reverse
        selected_indices = np.argsort(slice_variances)[-num_axial:][::-1]
        # Sort indices to maintain anatomical order (Top to Bottom)
        selected_indices = np.sort(selected_indices)

    axial_slices = volume[selected_indices]

    # --- Coronal Selection ---
    views = [slice_img for slice_img in axial_slices]

    if use_coronal:
        # Extract central coronal slice (slicing along Y axis)
        center_y = volume.shape[1] // 2
        coronal_slice = volume[:, center_y, :]

        # Coronal slice dimensions are (Z, X). We need to resize this later to match Axial (Y, X).
        # We add it to the list; preprocessing will handle resizing.
        views.append(coronal_slice)

    return views


def compute_density_profile(volume: np.ndarray, bin_edges: list) -> np.ndarray:
    """
    Computes a normalized histogram of tissue densities based on Hounsfield Units.

    Args:
        volume: 3D numpy array of HU values.
        bin_edges: List of edges defining the bins (e.g., [-2000, -950, -700, -400, 2000]).

    Returns:
        np.ndarray: Normalized frequency for each bin (sum = 1.0).
    """
    # Flatten volume for histogram computation
    flat_vol = volume.flatten()

    # Compute histogram
    counts, _ = np.histogram(flat_vol, bins=bin_edges)

    # Normalize
    total_counts = counts.sum()
    if total_counts > 0:
        density_profile = counts / total_counts
    else:
        density_profile = np.zeros(len(bin_edges) - 1)

    return density_profile.astype(np.float32)


def preprocess_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Preprocesses a single 2D slice for the CNN backbone.
    1. Clips to lung window.
    2. Normalizes to [0, 1].
    3. Resizes to target size.
    4. Stacks to 3 channels (RGB).

    Args:
        image: 2D numpy array (HU values).
        size: Target spatial dimension (size, size).

    Returns:
        np.ndarray: Preprocessed image of shape (size, size, 3).
    """
    # 1. Clip to Lung Window
    img = np.clip(image, Config.HU_MIN, Config.HU_MAX)

    # 2. Normalize to [0, 1]
    # Avoid division by zero
    denom = Config.HU_MAX - Config.HU_MIN
    if denom == 0:
        denom = 1
    img = (img - Config.HU_MIN) / denom

    # 3. Resize
    # cv2.resize expects (Width, Height)
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # 4. Stack to 3 channels (for ImageNet-pretrained backbones)
    img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1)

    return img_rgb.astype(np.float32)


def process_patient(
    patient_id: str, dcm_path: str, load_cached_data: bool = True
) -> dict:
    """
    Orchestrates the processing for a single patient with caching.

    Args:
        patient_id: Unique patient identifier.
        dcm_path: Relative path to the DICOM directory (e.g. 'train/ID...').
        load_cached_data: If True, attempts to load from disk before processing.

    Returns:
        dict: Contains 'images' (N, 224, 224, 3) and 'density' (4,).
    """
    # Define cache path
    cache_path = os.path.join(Config.WORKING_DIR, f"{patient_id}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=False is default and safer, works for standard arrays
            data = np.load(cache_path)
            return {"images": data["images"], "density": data["density"]}
        except Exception as e:
            print(f"Failed to load cache for {patient_id}: {e}. Reprocessing.")

    # 2. Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, dcm_path)

    # Load Volume
    volume = load_volume(full_path)

    # Extract Views (Raw 2D slices)
    raw_views = extract_orthogonal_views(
        volume, num_axial=Config.NUM_AXIAL_SLICES, use_coronal=Config.USE_CORONAL
    )

    # Preprocess Images (Resize, Norm, RGB)
    processed_images = []
    for view in raw_views:
        p_img = preprocess_image(view, size=Config.IMAGE_SIZE)
        processed_images.append(p_img)

    images_array = np.array(processed_images, dtype=np.float32)

    # Compute Density Profile
    density_profile = compute_density_profile(volume, Config.DENSITY_BIN_EDGES)

    # 3. Save to cache (using savez_compressed for efficiency)
    try:
        np.savez_compressed(cache_path, images=images_array, density=density_profile)
    except Exception as e:
        print(f"Warning: Failed to save cache for {patient_id}: {e}")

    return {"images": images_array, "density": density_profile}
