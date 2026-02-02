import os
import cv2
import numpy as np
import torch
import random
import struct
from library.config import SEED


def seed_everything(seed=SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_coupled(file_path):
    """
    Reads a DICOM file, handling MONOCHROME1 inversion and returning
    the image tensor coupled with its original dimensions.

    This function implements a custom binary parser to extract critical metadata
    (Dimensions and Photometric Interpretation) and uses OpenCV for efficient
    pixel decoding, with a fallback to raw buffer reading.

    Args:
        file_path (str): Path to the DICOM file.

    Returns:
        tuple: (image_array, (original_height, original_width))
            - image_array: Numpy array of the image (semantically normalized, Black=0).
            - (original_height, original_width): Tuple of integers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    # Read file content as binary for metadata parsing
    with open(file_path, "rb") as f:
        data = f.read()

    # --- Metadata Extraction (Binary Pattern Matching) ---
    # We look for specific byte sequences corresponding to Explicit VR Little Endian tags.
    # This is a robust heuristic for standard DICOM files.

    # Sequence includes Tag + VR + Length (for US)
    # Rows (0028,0010) US(2 bytes) Len(2 bytes) -> b'\x10\x00\x28\x00\x55\x53\x02\x00'
    rows_seq = b"\x10\x00\x28\x00\x55\x53\x02\x00"
    cols_seq = b"\x11\x00\x28\x00\x55\x53\x02\x00"

    h, w = None, None

    # Find Rows
    idx_rows = data.find(rows_seq)
    if idx_rows != -1:
        # Value follows immediately (2 bytes, unsigned short)
        h = struct.unpack_from("<H", data, idx_rows + len(rows_seq))[0]

    # Find Cols
    idx_cols = data.find(cols_seq)
    if idx_cols != -1:
        w = struct.unpack_from("<H", data, idx_cols + len(cols_seq))[0]

    # Check Photometric Interpretation for MONOCHROME1
    # Tag (0028,0004) VR CS (Code String) -> b'\x04\x00\x28\x00\x43\x53'
    # Followed by 2 bytes length, then string value
    photo_seq = b"\x04\x00\x28\x00\x43\x53"
    idx_photo = data.find(photo_seq)
    is_monochrome1 = False

    if idx_photo != -1:
        # Read length (2 bytes)
        length = struct.unpack_from("<H", data, idx_photo + len(photo_seq))[0]
        # Read value
        val_start = idx_photo + len(photo_seq) + 2
        val_bytes = data[val_start : val_start + length]
        val_str = val_bytes.decode("ascii", errors="ignore").strip()
        if "MONOCHROME1" in val_str:
            is_monochrome1 = True

    # --- Image Loading ---
    # Attempt 1: OpenCV (Fast, handles compression)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # Attempt 2: Raw Pixel Parsing (Fallback)
    if img is None:
        # Find Pixel Data Tag (7FE0,0010)
        # Search for Tag only: b'\x10\x00\xe0\x7f'
        pixel_tag = b"\x10\x00\xe0\x7f"
        idx_pixel = data.find(pixel_tag)

        if idx_pixel != -1 and h is not None and w is not None:
            # Determine offset based on VR (Explicit vs Implicit)
            # Check bytes after tag (4 bytes)
            vr = data[idx_pixel + 4 : idx_pixel + 6]
            if vr in [b"OB", b"OW"]:
                # Explicit VR: Tag(4) + VR(2) + Reserved(2) + Length(4) -> Data
                offset = 12
            else:
                # Implicit VR or other: Tag(4) + Length(4) -> Data
                # Note: This is a simplification, but covers common cases
                offset = 8

            data_start = idx_pixel + offset

            # Assume 8-bit or 16-bit based on size
            expected_pixels = h * w
            remaining_bytes = len(data) - data_start

            if remaining_bytes >= expected_pixels * 2:
                # 16-bit
                arr = np.frombuffer(
                    data, dtype=np.uint16, count=expected_pixels, offset=data_start
                )
                img = arr.reshape((h, w))
            elif remaining_bytes >= expected_pixels:
                # 8-bit
                arr = np.frombuffer(
                    data, dtype=np.uint8, count=expected_pixels, offset=data_start
                )
                img = arr.reshape((h, w))

    if img is None:
        raise ValueError(f"Could not decode image: {file_path}")

    # Ensure dimensions match metadata if we found it
    if h is None or w is None:
        h, w = img.shape[:2]

    # --- Semantic Normalization ---
    # If MONOCHROME1 (0=White), invert so 0=Black (standard)
    if is_monochrome1:
        if img.dtype == np.uint8:
            max_val = 255
        elif img.dtype == np.uint16:
            max_val = 65535
        else:
            max_val = np.max(img)

        img = max_val - img

    # Ensure consistent shape (H, W) - convert 3 channel to grayscale if necessary
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, (h, w)


def get_image_data(image_id, file_path, cache_dir, load_cached_data=True):
    """
    Retrieves image data, using a cache to store processed numpy arrays.
    Strictly couples metadata (dimensions) with the loaded image artifact.

    Args:
        image_id (str): Unique identifier for the image (used for cache filename).
        file_path (str): Full path to the source DICOM file.
        cache_dir (str): Directory to store/retrieve cached .npy files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (image_array, (height, width))
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{image_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            img = np.load(cache_path)
            # Extract dimensions directly from the loaded artifact
            h, w = img.shape[:2]
            return img, (h, w)
        except Exception:
            # If cache is corrupt or load fails, proceed to re-compute
            pass

    # 2. Compute from scratch
    img, (h, w) = read_dicom_coupled(file_path)

    # 3. Save to cache
    np.save(cache_path, img)

    return img, (h, w)
