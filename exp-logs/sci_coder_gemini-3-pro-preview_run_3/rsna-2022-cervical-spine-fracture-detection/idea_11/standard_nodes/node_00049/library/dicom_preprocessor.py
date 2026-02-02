import os
import glob
import numpy as np
import cv2
import logging
import pandas as pd
from library.config import Config
from library.utils import get_logger

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logger = get_logger("dicom_preprocessor")


def read_dicom_image(path):
    """
    Reads a DICOM file and returns the pixel array and metadata.
    Attempts to use pydicom first, then falls back to OpenCV/Raw reading.

    Returns:
        tuple: (image_array, z_pos, slope, intercept)
        image_array will be None if reading fails.
    """
    # Default values
    try:
        # Fallback Z-position from filename (e.g., '10.dcm' -> 10.0)
        filename_idx = int(os.path.basename(path).split(".")[0])
        z_pos = float(filename_idx)
    except ValueError:
        z_pos = 0.0

    slope = 1.0
    intercept = -1024.0  # Standard CT intercept
    image = None

    # Strategy 1: Use pydicom (Preferred for accurate metadata)
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)

            # Extract Z position from ImagePositionPatient
            if hasattr(dcm, "ImagePositionPatient"):
                z_pos = float(dcm.ImagePositionPatient[2])

            # Extract Rescale Slope/Intercept
            if hasattr(dcm, "RescaleSlope"):
                slope = float(dcm.RescaleSlope)
            if hasattr(dcm, "RescaleIntercept"):
                intercept = float(dcm.RescaleIntercept)

            image = dcm.pixel_array.astype(np.float32)
            return image, z_pos, slope, intercept
        except Exception:
            pass

    # Strategy 2: OpenCV (Fallback for simple formats or if pydicom missing)
    try:
        img_cv = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_cv is not None:
            return img_cv.astype(np.float32), z_pos, slope, intercept
    except Exception:
        pass

    # Strategy 3: Raw JPEG Extraction (Fallback for compressed DICOM without pydicom)
    # DICOM often encapsulates JPEG streams. We look for the JPEG SOI marker (0xFFD8).
    try:
        with open(path, "rb") as f:
            data = f.read()

        start = data.find(b"\xff\xd8")
        if start != -1:
            jpeg_data = data[start:]
            img_array = np.frombuffer(jpeg_data, np.uint8)
            img_decoded = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img_decoded is not None:
                if len(img_decoded.shape) > 2:
                    img_decoded = img_decoded[:, :, 0]  # Take first channel if RGB
                return img_decoded.astype(np.float32), z_pos, slope, intercept
    except Exception:
        pass

    return None, z_pos, slope, intercept


def load_dicom_stack(study_dir):
    """
    Loads all DICOM files in a directory, sorts them by Z-position,
    converts to Hounsfield Units (HU), and stacks them into a 3D volume.
    """
    files = glob.glob(os.path.join(study_dir, "*.dcm"))
    if not files:
        # Try matching any file if .dcm extension is missing
        files = glob.glob(os.path.join(study_dir, "*"))
        files = [f for f in files if os.path.isfile(f)]

    if not files:
        return None

    slices = []
    for f in files:
        img, z, slope, intercept = read_dicom_image(f)
        if img is not None:
            # Convert to Hounsfield Units immediately
            img = img * slope + intercept
            slices.append({"img": img, "z": z})

    if not slices:
        return None

    # Sort strictly by Z position
    slices.sort(key=lambda x: x["z"])

    # Stack into volume (D, H, W)
    volume = np.stack([s["img"] for s in slices])
    return volume


def apply_windowing(volume, level, width):
    """
    Applies the standard bone window and converts to uint8 [0, 255].
    Formula: (pixel - (level - width/2)) / width
    """
    lower = level - width / 2
    upper = level + width / 2

    # Clip to window
    volume = np.clip(volume, lower, upper)

    # Normalize to 0-1
    volume = (volume - lower) / (upper - lower)

    # Scale to 0-255 and cast
    volume = volume * 255.0
    return volume.astype(np.uint8)


def resize_and_sample(volume):
    """
    Resizes spatial dimensions to Config.IMAGE_SIZE and
    samples/pads depth to Config.NUM_SLICES.
    """
    # 1. Resize spatial dimensions (H, W) -> (IMAGE_SIZE, IMAGE_SIZE)
    resized_slices = []
    for i in range(volume.shape[0]):
        # cv2.resize expects (width, height)
        res = cv2.resize(
            volume[i],
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        resized_slices.append(res)

    volume = np.stack(resized_slices)

    # 2. Sample or Pad depth to NUM_SLICES
    current_depth = volume.shape[0]
    target_depth = Config.NUM_SLICES

    if current_depth == target_depth:
        return volume

    if current_depth > target_depth:
        # Uniform sampling
        indices = np.linspace(0, current_depth - 1, target_depth).astype(int)
        volume = volume[indices]
    else:
        # Pad with zeros (inferior/superior padding)
        pad_size = target_depth - current_depth
        padding = np.zeros(
            (pad_size, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
        )
        volume = np.concatenate([volume, padding], axis=0)

    return volume


def preprocess_and_cache(metadata_df, load_cached_data=True):
    """
    Main function to preprocess studies listed in metadata and cache them as .npy files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'StudyInstanceUID' and 'image_path'.
        load_cached_data (bool): If True, skips processing if file exists.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if "StudyInstanceUID" not in metadata_df.columns:
        logger.error("Metadata missing StudyInstanceUID column.")
        return

    # Process unique studies
    unique_studies = metadata_df[["StudyInstanceUID", "image_path"]].drop_duplicates()

    logger.info(
        f"Preprocessing {len(unique_studies)} studies. Cache dir: {Config.CACHE_DIR}"
    )

    for idx, row in unique_studies.iterrows():
        study_id = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        save_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

        # Check cache
        if load_cached_data and os.path.exists(save_path):
            continue

        try:
            # Load Volume
            volume = load_dicom_stack(full_path)

            if volume is None:
                logger.warning(
                    f"Could not load volume for {study_id}. Generating dummy."
                )
                # Create zero volume to prevent pipeline failure
                volume = np.zeros(
                    (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    dtype=np.uint8,
                )
            else:
                # Apply Windowing
                volume = apply_windowing(
                    volume, Config.WINDOW_LEVEL, Config.WINDOW_WIDTH
                )
                # Resize and Sample
                volume = resize_and_sample(volume)

            # Save to cache
            np.save(save_path, volume)

        except Exception as e:
            logger.error(f"Error processing {study_id}: {e}")
            # Save dummy to ensure training doesn't fail on file not found
            dummy = np.zeros(
                (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                dtype=np.uint8,
            )
            np.save(save_path, dummy)

    logger.info("Preprocessing completed.")
