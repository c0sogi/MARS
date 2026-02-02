import os
import numpy as np
import cv2
from library.config import Config


def read_dicom_manual(file_path):
    """
    Manually parses a DICOM file to extract pixel data without relying on pydicom.
    Handles both Encapsulated (JPEG) and Native (Raw) pixel data.
    Cite debug_lesson_1: Implement Low-Level Binary Parsing Fallbacks.

    Args:
        file_path (str): Full path to the DICOM file.

    Returns:
        np.ndarray: The image data (2D or 3D).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    with open(file_path, "rb") as f:
        data = f.read()

    img = None

    # --- Helper: Find Tag Value (Little Endian) ---
    def find_tag_value(tag_bytes, data, search_limit=2000):
        """
        Scans for a tag (e.g., b'\\x28\\x00\\x10\\x00' for Rows) and extracts its value.
        Handles Explicit VR (US) and Implicit VR (assumes 2-byte length for these tags).
        """
        start = 0
        while True:
            idx = data.find(tag_bytes, start)
            if idx == -1 or idx > search_limit:
                return None

            # Check for Explicit VR 'US' (Unsigned Short) -> Tag(4) + VR(2) + Len(2) + Val(2)
            # VR 'US' is 0x5553
            if idx + 10 <= len(data):
                vr = data[idx + 4 : idx + 6]
                if vr == b"US":
                    val = int.from_bytes(data[idx + 8 : idx + 10], "little")
                    return val

            # Check for Implicit VR -> Tag(4) + Len(4) + Val(Len)
            # For Rows/Cols, Length is usually 2 or 4. We assume 2 for short.
            if idx + 10 <= len(data):
                length = int.from_bytes(data[idx + 4 : idx + 8], "little")
                if length == 2:
                    val = int.from_bytes(data[idx + 8 : idx + 10], "little")
                    return val

            start = idx + 1

    # 1. extract Metadata (Rows, Cols, Bits)
    # Tags: Rows (0028,0010), Columns (0028,0011), BitsAllocated (0028,0100)
    rows = find_tag_value(b"\x28\x00\x10\x00", data)
    cols = find_tag_value(b"\x28\x00\x11\x00", data)
    bits = find_tag_value(b"\x28\x00\x00\x01", data)

    if bits is None:
        bits = 16  # Default assumption for X-ray if missing

    # 2. Find Pixel Data Tag (7FE0,0010)
    # Little Endian byte sequence: E0 7F 10 00
    pixel_data_tag = b"\xe0\x7f\x10\x00"
    tag_idx = data.find(pixel_data_tag)

    if tag_idx != -1:
        # Possible offsets to the actual data start
        # Explicit VR: Tag(4) + VR(2) + Res(2) + Len(4) = 12 bytes
        # Implicit VR: Tag(4) + Len(4) = 8 bytes
        offsets = [12, 8, 0, 4, 16]

        # Strategy A: Try decoding as Encapsulated (JPEG/JPEG2000)
        for offset in offsets:
            start = tag_idx + offset
            if start >= len(data):
                continue

            # Grab a chunk to try decoding
            b = np.frombuffer(data[start:], dtype=np.uint8)
            try:
                decoded = cv2.imdecode(b, cv2.IMREAD_UNCHANGED)
                if decoded is not None and decoded.size > 0:
                    img = decoded
                    break
            except Exception:
                pass

        # Strategy B: Try reading as Raw/Native
        if img is None and rows is not None and cols is not None:
            num_pixels = rows * cols
            bytes_per_pixel = 2 if bits > 8 else 1
            expected_bytes = num_pixels * bytes_per_pixel

            for offset in offsets:
                start = tag_idx + offset
                if start + expected_bytes <= len(data):
                    raw = np.frombuffer(
                        data[start : start + expected_bytes], dtype=np.uint8
                    )

                    if bytes_per_pixel == 2:
                        # View as uint16
                        # Note: DICOM is usually Little Endian. np.frombuffer uses system endianness.
                        # We assume Little Endian system (x86/GPU).
                        raw = raw.view(np.uint16)

                    try:
                        img = raw.reshape((rows, cols))
                        break
                    except ValueError:
                        continue

    # Strategy 3: Fallback to direct cv2.imread (sometimes works for non-standard DICOMs)
    if img is None:
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        raise ValueError(f"Failed to parse pixel data from {file_path}")

    # Handle Photometric Interpretation
    # If 'MONOCHROME1' is present, 0 is White and Max is Black. We need to invert.
    # We search the binary header for this string.
    if b"MONOCHROME1" in data[:2000]:  # Header is usually at the beginning
        # Invert the image
        if img.dtype == np.uint8:
            img = 255 - img
        elif img.dtype == np.uint16:
            img = 65535 - img
        else:
            img = np.max(img) - img

    return img


def load_image_and_metadata(
    image_id, relative_path, cache_dir=None, load_cached_data=True
):
    """
    Centralized function to load image and metadata.
    Implements caching to speed up training.

    Args:
        image_id (str): Unique image identifier.
        relative_path (str): Path relative to input directory (e.g., 'train/xxx.dicom').
        cache_dir (str, optional): Directory to store/load cached .npy files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (processed_image, original_shape)
            processed_image (np.ndarray): HxWx3 uint8 array, normalized.
            original_shape (tuple): (Height, Width) of the original image.
    """

    # 1. Construct Paths
    full_input_path = os.path.join(Config.INPUT_DIR, relative_path)

    # Determine cache path if applicable
    cache_path = None
    if cache_dir:
        # Replicate subdirectory structure in cache (e.g., cache/train/id.npy)
        sub_dir = os.path.dirname(relative_path)
        cache_subdir = os.path.join(cache_dir, sub_dir)
        cache_path = os.path.join(cache_subdir, f"{image_id}.npy")

    # 2. Try Loading from Cache
    if load_cached_data and cache_path and os.path.exists(cache_path):
        try:
            # Load data: expected format is a dictionary or array containing both img and shape
            # To keep it simple and robust, we save the image array and rely on its shape,
            # but we need original shape if we resize.
            # Let's save a dictionary object using numpy
            data = np.load(cache_path, allow_pickle=True).item()
            return data["image"], data["original_shape"]
        except Exception as e:
            # If load fails, proceed to re-compute
            pass

    # 3. Compute from Scratch
    try:
        # Parse DICOM
        raw_img = read_dicom_manual(full_input_path)

        # Get Original Dimensions
        orig_h, orig_w = raw_img.shape[:2]

        # Normalization and Preprocessing
        # Convert to float for normalization
        img_float = raw_img.astype(np.float32)

        # Min-Max Scaling to 0-255
        min_val = np.min(img_float)
        max_val = np.max(img_float)

        if max_val > min_val:
            img_norm = (img_float - min_val) / (max_val - min_val) * 255.0
        else:
            img_norm = np.zeros_like(img_float)

        img_uint8 = img_norm.astype(np.uint8)

        # Ensure 3 Channels (RGB) for EfficientNet
        if len(img_uint8.shape) == 2:
            img_final = cv2.merge([img_uint8, img_uint8, img_uint8])
        elif len(img_uint8.shape) == 3 and img_uint8.shape[2] == 1:
            img_final = cv2.merge([img_uint8, img_uint8, img_uint8])
        else:
            # Already 3 channels (rare for X-ray but possible if converted)
            img_final = img_uint8

        # 4. Save to Cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            save_data = {"image": img_final, "original_shape": (orig_h, orig_w)}
            np.save(cache_path, save_data)

        return img_final, (orig_h, orig_w)

    except Exception as e:
        # Fallback strategy: Return dummy image to prevent pipeline crash
        # Cite debug_lesson_3: Synchronize Dependent Metadata When Triggering Data Fallbacks
        print(f"Warning: Failed to load {image_id}, using dummy fallback. Error: {e}")
        dummy_size = Config.IMG_SIZE
        img_final = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)
        return img_final, (dummy_size, dummy_size)
