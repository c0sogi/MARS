import os
import numpy as np
import cv2
import io
import rasterio
from PIL import Image
from library.config import Config


def read_dicom_manual(file_path):
    """
    Manually parses a DICOM file to extract pixel data without relying on pydicom.
    Leverages rasterio and PIL for robust format support.
    Cite debug_lesson_1: Implement Low-Level Binary Parsing Fallbacks.

    Args:
        file_path (str): Full path to the DICOM file.

    Returns:
        np.ndarray: The image data (2D or 3D).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    # Strategy 1: Rasterio (GDAL backend)
    # Rasterio is robust for many geospatial and medical formats if drivers are present.
    try:
        with rasterio.open(file_path) as src:
            # Read the first band
            img = src.read(1)
            if img is not None and img.size > 0:
                return img
    except Exception:
        pass

    # Read binary data for manual extraction
    with open(file_path, "rb") as f:
        data = f.read()

    img = None

    # Strategy 2: Encapsulated Stream Extraction (JPEG / JPEG 2000)
    # We look for Start of Image (SOI) markers.
    # JPEG SOI: FF D8
    # JPEG 2000 SOC: FF 4F FF 51

    signatures = [(b"\xff\xd8", "JPEG"), (b"\xff\x4f\xff\x51", "JP2")]

    for sig, fmt in signatures:
        start = 0
        attempts = 0
        # Try finding the signature within the first few KB or throughout
        # DICOM headers can be variable length, so we search.
        while attempts < 5:
            idx = data.find(sig, start)
            if idx == -1:
                break

            # Extract potential stream from here to end
            # PIL is usually smart enough to stop reading at EOI
            stream = data[idx:]

            try:
                # Use PIL to open the bytes
                with Image.open(io.BytesIO(stream)) as pil_img:
                    # Convert to numpy
                    decoded = np.array(pil_img)

                    if decoded is not None and decoded.size > 0:
                        # Sanity check dimensions (X-rays are usually large)
                        if decoded.shape[0] > 64 and decoded.shape[1] > 64:
                            img = decoded
                            break
            except Exception:
                pass

            start = idx + 2
            attempts += 1

        if img is not None:
            break

    # Strategy 3: Structural Tag Search (Fallback for uncompressed/raw)
    if img is None:
        # Pixel Data Tag: 7FE0,0010 -> Little Endian: E0 7F 10 00
        pixel_data_tag = b"\xe0\x7f\x10\x00"
        tag_idx = data.find(pixel_data_tag)

        if tag_idx != -1:
            # Metadata extraction (Rows/Cols) needed for raw
            def find_val(tag, data):
                idx = data.find(tag)
                if idx != -1 and idx + 10 <= len(data):
                    # We try a few offsets for the value (Implicit vs Explicit VR)
                    for off in [8, 6, 10]:
                        try:
                            return int.from_bytes(
                                data[idx + off : idx + off + 2], "little"
                            )
                        except:
                            pass
                return None

            rows = find_val(b"\x28\x00\x10\x00", data)  # Rows
            cols = find_val(b"\x28\x00\x11\x00", data)  # Cols

            if rows and cols:
                # Assume 16-bit or 8-bit raw
                # Try offsets after tag
                for offset in [8, 12, 20]:  # Common header lengths
                    start_px = tag_idx + offset
                    expected_size = rows * cols

                    # Try 16-bit
                    if start_px + expected_size * 2 <= len(data):
                        raw = np.frombuffer(
                            data[start_px : start_px + expected_size * 2],
                            dtype=np.uint16,
                        )
                        if raw.size == expected_size:
                            img = raw.reshape((rows, cols))
                            break

                    # Try 8-bit
                    if start_px + expected_size <= len(data):
                        raw = np.frombuffer(
                            data[start_px : start_px + expected_size], dtype=np.uint8
                        )
                        if raw.size == expected_size:
                            img = raw.reshape((rows, cols))
                            break

    # Strategy 4: Direct OpenCV Read (Last Resort)
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
    if b"MONOCHROME1" in data[:2048]:  # Header is usually at the beginning
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
