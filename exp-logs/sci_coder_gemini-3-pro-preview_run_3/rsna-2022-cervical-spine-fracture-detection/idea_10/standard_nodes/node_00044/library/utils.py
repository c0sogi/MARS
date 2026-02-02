import os
import glob
import numpy as np
import cv2
import torch
from library.config import Config, seed_everything

# Attempt to import pydicom for metadata-based sorting and precise HU conversion.
# If not installed, fallback to filename sorting and raw reading.
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def read_dicom_series(directory):
    """
    Reads DICOM files from a directory and sorts them.
    - If pydicom is available: Sorts strictly by ImagePositionPatient Z-coordinate.
    - Fallback: Sorts by the integer index in the filename.
    """
    files = glob.glob(os.path.join(directory, "*.dcm"))
    if not files:
        return []

    if HAS_PYDICOM:
        # Sort by Z position using pydicom headers
        sorted_files = []
        for f in files:
            try:
                # Read only specific tags for speed (stop_before_pixels=True)
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                # ImagePositionPatient is tag (0x0020, 0x0032), we need Z (index 2)
                pos_z = float(ds.ImagePositionPatient[2])
                sorted_files.append((pos_z, f))
            except Exception:
                # Fallback to filename if header read fails
                try:
                    name_idx = int(os.path.splitext(os.path.basename(f))[0])
                    sorted_files.append((name_idx, f))
                except:
                    sorted_files.append((0, f))

        # Sort based on the extracted key
        sorted_files.sort(key=lambda x: x[0])
        return [x[1] for x in sorted_files]
    else:
        # Fallback sorting by filename integer
        def get_index(f):
            try:
                return int(os.path.splitext(os.path.basename(f))[0])
            except:
                return 0

        files.sort(key=get_index)
        return files


def process_dicom_slice(file_path, img_size=None):
    """
    Reads a DICOM file, converts to Hounsfield Units (HU), applies
    Standard Bone Window (L:400, W:1800), and normalizes to 0-255 uint8.
    """
    # Bone Window Settings
    center = Config.WINDOW_LEVEL
    width = Config.WINDOW_WIDTH
    lower = center - width / 2
    upper = center + width / 2

    img = None

    # 1. Try reading with pydicom for accurate HU conversion
    if HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(file_path)
            pixel_array = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)
            img = pixel_array * slope + intercept
        except Exception:
            pass

    # 2. Fallback to OpenCV if pydicom missing or failed
    if img is None:
        # cv2.imread with -1 loads raw data (usually uint16)
        raw_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            # Return black image if read fails completely
            if img_size:
                return np.zeros(img_size, dtype=np.uint8)
            else:
                return np.zeros((256, 256), dtype=np.uint8)

        img = raw_img.astype(np.float32)
        # Approximate HU conversion for fallback: assume standard CT intercept of -1024
        img = img - 1024

    # 3. Apply Windowing
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    # 4. Resize
    if img_size:
        # cv2.resize expects (width, height)
        img = cv2.resize(img, (img_size[1], img_size[0]))

    return (img * 255).astype(np.uint8)


def load_and_preprocess_scan(study_uid, image_root_dir, load_cached_data=True):
    """
    Loads a scan, processes it into a 2.5D volume, and returns it.

    Logic:
    1. Check cache for {study_uid}.npy. If exists and valid, return it.
    2. If not, read DICOMs, sort, sample 64 slices uniformly.
    3. Construct 2.5D inputs (z-1, z, z+1) for each sampled slice.
    4. Save to cache and return.

    Returns:
        np.ndarray of shape (NUM_SLICES, H, W, 3) with dtype uint8.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_uid}.npy")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Verify shape integrity
            if data.shape[0] == Config.NUM_SLICES and data.shape[-1] == 3:
                return data
        except Exception:
            pass  # Corrupt cache, proceed to recompute

    # 2. Compute from scratch
    study_dir = os.path.join(image_root_dir, study_uid)

    # Read and sort files
    sorted_paths = read_dicom_series(study_dir)
    num_files = len(sorted_paths)

    if num_files == 0:
        # Return empty volume if no images found
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3),
            dtype=np.uint8,
        )

    # Select indices (Uniform sampling)
    indices = np.linspace(0, num_files - 1, Config.NUM_SLICES).astype(int)

    # Identify all unique file indices needed (including neighbors for 2.5D)
    needed_indices = set()
    for idx in indices:
        needed_indices.add(idx)
        needed_indices.add(max(0, idx - 1))
        needed_indices.add(min(num_files - 1, idx + 1))

    # Load needed slices into memory
    # Optimization: Only load what we need to minimize I/O and processing
    slice_cache = {}
    for idx in needed_indices:
        fpath = sorted_paths[idx]
        slice_cache[idx] = process_dicom_slice(fpath, Config.IMG_SIZE)

    # Construct 2.5D Volume
    volume = []
    for idx in indices:
        prev_idx = max(0, idx - 1)
        next_idx = min(num_files - 1, idx + 1)

        # Channels: z-1, z, z+1
        ch0 = slice_cache[prev_idx]
        ch1 = slice_cache[idx]
        ch2 = slice_cache[next_idx]

        # Stack channels: (H, W, 3)
        img_25d = np.stack([ch0, ch1, ch2], axis=-1)
        volume.append(img_25d)

    volume = np.array(volume, dtype=np.uint8)

    # 3. Save to cache
    try:
        np.save(cache_path, volume)
    except Exception:
        pass  # Don't crash if save fails (e.g. disk full)

    return volume
