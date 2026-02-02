import os
import cv2
import numpy as np
from library.config import Config


def read_dicom_bytes(path: str) -> np.ndarray:
    """
    Reads a DICOM file as a binary stream and attempts to decode it using OpenCV.
    Implements a robust fallback that scans for JPEG/JPEG2000 headers if direct
    decoding fails, bypassing the need for pydicom.

    Args:
        path: Path to the .dcm file.

    Returns:
        np.ndarray: Grayscale image array (H, W) or zeros if decoding fails.
    """
    if not os.path.exists(path):
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # Attempt 1: Direct decode
        # numpy frombuffer is faster than fromstring
        arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # Attempt 2: Scan for JPEG Header (0xFF 0xD8)
        if img is None:
            jpeg_start = raw_bytes.find(b"\xff\xd8")
            if jpeg_start != -1:
                arr = np.frombuffer(raw_bytes[jpeg_start:], np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # Attempt 3: Scan for JPEG2000 Header (0x00 0x00 0x00 0x0C 0x6A 0x50 ...)
        # Common signature part: 0x6A 0x50 0x20 0x20
        if img is None:
            jp2_start = raw_bytes.find(
                b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a"
            )
            if jp2_start != -1:
                arr = np.frombuffer(raw_bytes[jp2_start:], np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

        # Final check
        if img is None:
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

        # Ensure Grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Handle 16-bit images by scaling to 8-bit for consistent processing
        # (Though we will normalize to float later, some cv2 ops like CLAHE prefer uint8/16)
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            # Normalize to 0-255
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        return img

    except Exception as e:
        # In production/competition, silent failure with zero-image is often safer than crashing
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Min-Max normalizes the image to [0, 1] range.
    """
    img_float = image.astype(np.float32)
    img_min = np.min(img_float)
    img_max = np.max(img_float)

    if img_max > img_min:
        img_float = (img_float - img_min) / (img_max - img_min)
    else:
        img_float = np.zeros_like(img_float)

    return img_float


def apply_clahe(
    image: np.ndarray, clip_limit: float, tile_grid_size: tuple
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Expects uint8 input, returns [0, 1] float32.
    """
    # Ensure input is uint8 for OpenCV CLAHE
    if image.dtype != np.uint8:
        # Normalize to 0-255 first if not already
        image = (normalize_image(image) * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(image)

    return normalize_image(img_clahe)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Applies Gamma Correction.
    Expects [0, 1] float input, returns [0, 1] float32.
    """
    # Ensure input is [0, 1] float
    if image.dtype == np.uint8:
        image = normalize_image(image)

    # Apply gamma: Output = Input ^ Gamma
    # Add epsilon to avoid log(0) issues if implemented via logs,
    # but np.power handles 0 fine for positive gamma.
    img_gamma = np.power(image, gamma)

    return img_gamma


def generate_tri_spectral_tensor(
    path: str,
    size: int = Config.IMAGE_SIZE,
    gamma: float = Config.GAMMA_VALUE,
    clahe_clip: float = Config.CLAHE_CLIP_LIMIT,
    clahe_grid: tuple = Config.CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Generates the 3-Channel Tri-Spectral Tensor for the model.

    Channel 1: Linear (Min-Max Normalized) - Structure
    Channel 2: Texture (CLAHE) - Microcalcifications
    Channel 3: Density (Gamma Correction) - Dense Tissue

    Args:
        path: Path to the DICOM file.
        size: Target spatial dimension (size x size).
        gamma: Gamma value for density channel.
        clahe_clip: Clip limit for CLAHE.
        clahe_grid: Grid size for CLAHE.

    Returns:
        np.ndarray: Float32 array of shape (size, size, 3) in range [0, 1].
    """
    # 1. Ingestion
    img_raw = read_dicom_bytes(path)

    # 2. Resize
    # Resize before processing to save compute, or after?
    # Resizing raw uint8 is standard.
    if img_raw.shape[0] != size or img_raw.shape[1] != size:
        img_resized = cv2.resize(img_raw, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_raw

    # 3. Channel Generation

    # Channel 1: Linear (Structure)
    # Just normalize the resized raw image
    c1_linear = normalize_image(img_resized)

    # Channel 2: Texture (CLAHE)
    # CLAHE works best on integer types, so use the uint8 resized image
    c2_texture = apply_clahe(
        img_resized, clip_limit=clahe_clip, tile_grid_size=clahe_grid
    )

    # Channel 3: Density (Gamma)
    # Apply to the normalized linear channel
    c3_density = apply_gamma(c1_linear, gamma=gamma)

    # 4. Stack
    # Shape: (H, W, 3)
    tensor = np.stack([c1_linear, c2_texture, c3_density], axis=-1)

    return tensor.astype(np.float32)
