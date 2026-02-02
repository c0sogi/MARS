import os
import numpy as np
import cv2

# Attempt to import pydicom, but handle the case where it might be missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def normalize_min_max(image, eps=1e-8):
    """
    Normalizes a numpy array to the range [0, 1] using min-max scaling.

    Args:
        image (np.ndarray): Input image array.
        eps (float): Small epsilon to prevent division by zero.

    Returns:
        np.ndarray: Normalized image float32 array in range [0, 1].
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    # Avoid division by zero if the image is constant (e.g., all black)
    if (max_val - min_val) < eps:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


def read_dicom_file(path, fix_monochrome=True):
    """
    Reads a DICOM file and returns the image as a numpy array.
    Implements a robust fallback mechanism:
    1. pydicom (Standard)
    2. OpenCV (Secondary)
    3. Raw Binary Fallback (Heuristic based on file size)

    Args:
        path (str): Path to the .dcm file.
        fix_monochrome (bool): If True, inverts MONOCHROME1 images so 0 is black.

    Returns:
        np.ndarray: Image array (uint16 or uint8). Returns a blank 256x256 array on total failure.
    """
    if not os.path.exists(path):
        # Return blank image to ensure pipeline continuity
        return np.zeros((256, 256), dtype=np.uint16)

    # --------------------------------------------------------------------------
    # Method 1: PyDicom (Preferred)
    # --------------------------------------------------------------------------
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array

            # Fix Photometric Interpretation: MONOCHROME1 means 0 is white, 1 is black.
            # We want 0 to be black (MONOCHROME2).
            if fix_monochrome and hasattr(dcm, "PhotometricInterpretation"):
                if dcm.PhotometricInterpretation == "MONOCHROME1":
                    img = np.amax(img) - img

            return img
        except Exception:
            # Continue to fallbacks if pydicom fails (e.g., corrupt header)
            pass

    # --------------------------------------------------------------------------
    # Method 2: OpenCV
    # --------------------------------------------------------------------------
    try:
        # cv2.IMREAD_UNCHANGED attempts to read the depth correctly (16-bit)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # --------------------------------------------------------------------------
    # Method 3: Raw Binary Fallback
    # --------------------------------------------------------------------------
    # If headers are corrupt, we assume the pixel data is a contiguous block
    # at the end of the file matching standard MRI dimensions.
    try:
        file_size = os.path.getsize(path)

        # Common MRI dimensions (Height, Width) for 16-bit images (2 bytes/pixel)
        # 512x512 -> 524,288 bytes
        # 256x256 -> 131,072 bytes
        candidates = [(512, 512), (256, 256), (192, 192)]

        with open(path, "rb") as f:
            content = f.read()

        for h, w in candidates:
            num_pixels = h * w
            num_bytes = num_pixels * 2  # 16-bit

            # Check if file is large enough to contain this resolution
            if file_size >= num_bytes:
                # Calculate header size. Valid DICOM headers are usually small (<10KB)
                # but can vary. We assume pixel data is at the very end.
                header_size = file_size - num_bytes

                # Heuristic: If the "header" is reasonable (positive and not huge)
                # We assume this is the correct resolution.
                if 0 <= header_size < 20000:
                    pixel_data = content[-num_bytes:]
                    # Load as uint16
                    img = np.frombuffer(pixel_data, dtype=np.uint16)
                    img = img.reshape((h, w))
                    return img
    except Exception:
        pass

    # --------------------------------------------------------------------------
    # Ultimate Fallback
    # --------------------------------------------------------------------------
    # Return a black image to prevent the dataloader from crashing
    return np.zeros((256, 256), dtype=np.uint16)
