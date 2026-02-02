import os
import numpy as np
import cv2

# Attempt to import pydicom (Tier 1)
# Note: pydicom is standard for DICOM but might be missing in some restricted environments.
# We wrap it in a try-except block to ensure the code runs even if it's not installed.
try:
    import pydicom

    _HAS_PYDICOM = True
except ImportError:
    _HAS_PYDICOM = False


def read_dicom_file(path: str) -> np.ndarray:
    """
    Reads a DICOM file using a Multi-Tiered Loading Strategy.

    Strategy:
    1. Standard Library (pydicom): Best for parsing headers and handling transfer syntaxes.
    2. Standard Library (OpenCV): Robust image reading, handles some DICOM formats.
    3. Raw Binary Tail-Read: Fallback for corrupt headers. Reads raw pixel bytes from file tail.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: The image data as a float32 numpy array.
                    Returns a zero-filled array (512x512) if all methods fail.
    """
    # 0. Check existence
    if not os.path.exists(path):
        return np.zeros((512, 512), dtype=np.float32)

    # 1. Tier 1: pydicom
    if _HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            # Convert to float32 immediately to preserve precision
            img = dcm.pixel_array.astype(np.float32)
            return img
        except Exception:
            # Continue to next tier on any error
            pass

    # 2. Tier 2: OpenCV
    try:
        # IMREAD_UNCHANGED is essential to preserve 16-bit depth
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img.astype(np.float32)
    except Exception:
        pass

    # 3. Tier 3: Raw Binary Tail-Read
    try:
        return _read_binary_fallback(path)
    except Exception:
        pass

    # 4. Final Fallback
    # Return a blank image to prevent pipeline crash.
    # Using 512x512 as a safe default for MRI.
    return np.zeros((512, 512), dtype=np.float32)


def _read_binary_fallback(path: str) -> np.ndarray:
    """
    Attempts to read pixel data by directly accessing the file tail.
    Useful when DICOM headers are corrupt but pixel data is intact.

    This function uses heuristics to match the file size against common
    MRI dimensions (512x512, 256x256) and bit depths (uint16, uint8).
    """
    file_size = os.path.getsize(path)

    # Heuristic candidates: (Height, Width, Dtype, BytesPerPixel)
    # Ordered by size descending to match largest possible valid data first.
    # MRI data is typically uint16 (2 bytes per pixel).
    candidates = [
        (512, 512, np.uint16, 2),  # ~512KB (Most common)
        (256, 256, np.uint16, 2),  # ~128KB
        (512, 512, np.uint8, 1),  # ~256KB
        (256, 256, np.uint8, 1),  # ~64KB
    ]

    with open(path, "rb") as f:
        for h, w, dtype, bpp in candidates:
            num_bytes = h * w * bpp

            # Check if file is large enough to contain this data
            # (File size = Header + Pixel Data, so File Size >= Pixel Data)
            if file_size >= num_bytes:
                try:
                    # Seek to the start of the pixel data (assuming it's at the end)
                    f.seek(-num_bytes, os.SEEK_END)
                    data = f.read(num_bytes)

                    # Convert bytes to numpy array
                    arr = np.frombuffer(data, dtype=dtype)

                    # Verify size and reshape
                    if arr.size == h * w:
                        return arr.reshape((h, w)).astype(np.float32)
                except Exception:
                    continue

    raise ValueError(
        "Binary fallback failed: Could not match file size to common dimensions."
    )
