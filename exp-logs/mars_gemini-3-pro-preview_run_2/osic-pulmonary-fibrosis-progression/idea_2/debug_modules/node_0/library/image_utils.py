import os
import numpy as np
import cv2
import torch
from library.config import Config


def read_dicom_raw(path):
    """
    Reads a DICOM file as a raw binary file and attempts to extract the pixel data
    based on common CT image dimensions. This serves as a fallback mechanism since
    pydicom is not available in the environment.

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D numpy array of the image (int16), or a blank array on failure.
    """
    try:
        with open(path, "rb") as f:
            b = f.read()

        file_size = len(b)

        # List of common square dimensions for CT scans
        # 512 is standard, others are possible crops or high-res scans
        dims = [512, 768, 384, 256, 1024]

        for dim in dims:
            expected_pixels = dim * dim
            expected_bytes = expected_pixels * 2  # 16-bit depth (2 bytes per pixel)

            # DICOM files consist of a Header + Pixel Data.
            # We assume pixel data is at the end of the file.
            # We check if the file size is sufficient and if the header size is reasonable.
            diff = file_size - expected_bytes

            # Header is typically > 128 bytes (preamble) + metadata.
            # We set a loose upper bound for header size to avoid false positives on small random files.
            if 128 <= diff < 200000:
                # Extract the last N bytes corresponding to the pixel data
                pixel_data = b[-expected_bytes:]

                # Convert bytes to int16 numpy array
                img = np.frombuffer(pixel_data, dtype=np.int16).reshape(dim, dim)
                return img

        # If no dimension matches, return a blank 512x512 image to maintain pipeline stability
        return np.zeros((512, 512), dtype=np.int16)

    except Exception as e:
        # In case of I/O errors, return a blank image
        return np.zeros((512, 512), dtype=np.int16)


def select_slices(dcm_dir, ratios=None):
    """
    Selects representative slices from a patient's CT scan directory based on depth ratios.

    Args:
        dcm_dir (str): Path to the directory containing .dcm files.
        ratios (list of float, optional): Ratios of depth to select (e.g., [0.2, 0.5, 0.8]).
                                          Defaults to Config.SLICE_SELECTION_RATIOS.

    Returns:
        list: List of full file paths to the selected slices.
    """
    if ratios is None:
        ratios = Config.SLICE_SELECTION_RATIOS

    if not os.path.exists(dcm_dir):
        return []

    # List all .dcm files in the directory
    files = [f for f in os.listdir(dcm_dir) if f.lower().endswith(".dcm")]

    if not files:
        return []

    # Sort files numerically to ensure correct depth ordering.
    # Filenames are typically "1.dcm", "10.dcm", etc.
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        # Fallback to string sort if filenames do not follow the numeric convention
        files.sort()

    num_files = len(files)
    selected_paths = []

    for r in ratios:
        # Calculate the index corresponding to the ratio
        idx = int(r * num_files)
        # Ensure index is within bounds
        idx = max(0, min(idx, num_files - 1))
        selected_paths.append(os.path.join(dcm_dir, files[idx]))

    return selected_paths


def load_and_preprocess_scan(dcm_paths):
    """
    Loads DICOM images from the provided paths, preprocesses them (resize, normalize),
    and returns a torch Tensor suitable for input into a CNN backbone.

    Args:
        dcm_paths (list): List of file paths to .dcm files.

    Returns:
        torch.Tensor: Batch of images with shape (Batch_Size, 3, H, W).
                      Values are normalized floats using ImageNet statistics.
    """
    images = []

    # ImageNet normalization statistics
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for path in dcm_paths:
        # 1. Read Image (Raw fallback)
        img = read_dicom_raw(path)

        # 2. Intensity Normalization
        # CT scans use Hounsfield Units (HU).
        # We clip to a range covering lung tissue and some soft tissue/bone context.
        # Range [-1000, 1000] covers Air (-1000) to Bone (~400-1000).
        img = np.clip(img, -1000, 1000)

        # Min-Max Scale to [0, 1]
        img = (img - (-1000)) / (1000 - (-1000))

        # Convert to float32 for processing
        img = img.astype(np.float32)

        # 3. Resize
        # Config.IMG_SIZE is (H, W). cv2.resize expects (W, H).
        target_size = (Config.IMG_SIZE[1], Config.IMG_SIZE[0])
        img_resized = cv2.resize(img, target_size)

        # 4. Channel Expansion
        # Pre-trained models (ResNet) expect 3 channels (RGB).
        # We replicate the grayscale image across 3 channels.
        img_rgb = np.stack([img_resized] * 3, axis=-1)  # Shape: (H, W, 3)

        # 5. ImageNet Standardization
        img_normalized = (img_rgb - mean) / std

        # 6. Transpose to Channel-First format (C, H, W) for PyTorch
        img_tensor = np.transpose(img_normalized, (2, 0, 1))

        images.append(img_tensor)

    # Stack into a batch tensor
    if not images:
        return torch.empty(0, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])

    batch_tensor = torch.tensor(np.stack(images), dtype=torch.float32)

    return batch_tensor
