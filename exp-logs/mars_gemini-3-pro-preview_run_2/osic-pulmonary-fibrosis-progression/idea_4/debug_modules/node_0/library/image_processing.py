import os
import numpy as np
import cv2
import warnings
from library.config import Config

# Attempt to import pydicom, handle absence gracefully as per constraints
try:
    import pydicom
except ImportError:
    pydicom = None
    warnings.warn(
        "pydicom module not found. Using dummy data generation for DICOM loading."
    )


def load_scan(path):
    """
    Loads DICOM scans from a given directory path.
    Sorts them by InstanceNumber to ensure correct 3D ordering.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        list: A list of pydicom datasets or DummyScan objects if pydicom is missing.
    """
    if pydicom is None:
        # Fallback for environments without pydicom
        class DummyScan:
            def __init__(self, i):
                # Simulate a 512x512 CT slice
                self.pixel_array = np.random.randint(
                    0, 4096, (512, 512), dtype=np.int16
                )
                self.RescaleSlope = 1
                self.RescaleIntercept = -1024
                self.InstanceNumber = i
                self.ImagePositionPatient = [0, 0, i]

        # Return a volume of 30 dummy slices
        return [DummyScan(i) for i in range(30)]

    if not os.path.exists(path):
        return []

    slices = []
    for s in os.listdir(path):
        if s.lower().endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception as e:
                continue

    # Sort slices by InstanceNumber (Z-position)
    # Some files might not have InstanceNumber, fallback to ImagePositionPatient if needed
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            pass  # Keep original order if sorting fails

    return slices


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel values to Hounsfield Units (HU).

    Args:
        scans (list): List of pydicom datasets.

    Returns:
        np.array: 3D numpy array of HU values (D, H, W).
    """
    if not scans:
        return np.array([])

    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield Units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def select_variance_slices(hu_volume, n_slices=5):
    """
    Selects indices of slices with the highest pixel variance within the lung window.
    Lung window defined as HU between -1000 and -500.

    Args:
        hu_volume (np.array): 3D array of HU values.
        n_slices (int): Number of slices to select.

    Returns:
        np.array: Indices of the selected slices.
    """
    if len(hu_volume) == 0:
        return np.array([])

    if len(hu_volume) <= n_slices:
        return np.arange(len(hu_volume))

    variances = []
    for i in range(len(hu_volume)):
        slice_img = hu_volume[i]
        # Mask for lung tissue
        mask = (slice_img >= -1000) & (slice_img <= -500)

        if np.sum(mask) > 0:
            var = np.var(slice_img[mask])
        else:
            var = 0
        variances.append(var)

    # Get indices of top N variances
    top_indices = np.argsort(variances)[-n_slices:]

    # Sort indices to maintain spatial ordering (optional but good practice)
    top_indices = np.sort(top_indices)

    return top_indices


def preprocess_for_model(image, img_size=256):
    """
    Preprocesses a single slice for the CNN model.
    Resizes, clips to lung window, normalizes, and stacks to 3 channels.

    Args:
        image (np.array): 2D array of HU values.
        img_size (int): Target spatial dimension.

    Returns:
        np.array: Processed image tensor (img_size, img_size, 3) with values in [0, 1].
    """
    # Clip to a reasonable lung window for visualization/CNN
    # Standard lung window center -600, width 1500 -> [-1350, 150]
    # Here we use a slightly wider range to capture tissue density
    min_hu, max_hu = -1000, 400
    image = np.clip(image, min_hu, max_hu)

    # Normalize to [0, 1]
    image = (image - min_hu) / (max_hu - min_hu)

    # Resize
    image = cv2.resize(image, (img_size, img_size))

    # Stack to 3 channels (RGB) for EfficientNet
    image = np.stack((image,) * 3, axis=-1)

    return image.astype(np.float32)


def process_patient_scan(
    patient_id,
    dcm_path,
    load_cached_data=True,
    n_slices=Config.SLICES_PER_PATIENT,
    img_size=Config.IMG_SIZE,
):
    """
    Orchestrates the loading, selection, and preprocessing of a patient's CT scan.
    Implements caching to disk.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_path (str): Path to the patient's DICOM folder.
        load_cached_data (bool): Whether to attempt loading from cache.
        n_slices (int): Number of slices to select.
        img_size (int): Target image size.

    Returns:
        np.array: 4D array of processed images (n_slices, img_size, img_size, 3).
    """
    # Ensure cache directory exists
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{patient_id}_processed.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to recomputing if load fails

    # 2. Compute from scratch
    try:
        scans = load_scan(dcm_path)
        hu_volume = get_pixels_hu(scans)

        if len(hu_volume) == 0:
            # Handle empty scan case with zeros
            processed_volume = np.zeros(
                (n_slices, img_size, img_size, 3), dtype=np.float32
            )
        else:
            selected_indices = select_variance_slices(hu_volume, n_slices=n_slices)
            selected_slices = hu_volume[selected_indices]

            # If we have fewer slices than requested (rare), pad with zeros
            if len(selected_slices) < n_slices:
                processed_imgs = [
                    preprocess_for_model(s, img_size) for s in selected_slices
                ]
                # Pad
                padding = [
                    np.zeros((img_size, img_size, 3), dtype=np.float32)
                    for _ in range(n_slices - len(processed_imgs))
                ]
                processed_volume = np.array(processed_imgs + padding)
            else:
                processed_volume = np.array(
                    [preprocess_for_model(s, img_size) for s in selected_slices]
                )

    except Exception as e:
        # Fail-safe: return zeros if processing crashes
        warnings.warn(f"Error processing {patient_id}: {str(e)}")
        processed_volume = np.zeros((n_slices, img_size, img_size, 3), dtype=np.float32)

    # 3. Save to cache
    try:
        np.save(cache_path, processed_volume)
    except Exception:
        pass

    return processed_volume
