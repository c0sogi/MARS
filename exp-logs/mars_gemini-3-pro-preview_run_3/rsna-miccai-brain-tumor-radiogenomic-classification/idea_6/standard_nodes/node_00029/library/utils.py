import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_file(path: str) -> np.ndarray:
    """
    Reads a DICOM file from the given path and returns a 2D numpy array.
    Handles missing libraries by falling back to raw binary reading based on file size.
    Resizes the image to Config.IMG_SIZE.
    """
    # Construct full path if a relative path is provided
    if not os.path.isabs(path):
        full_path = os.path.join(Config.INPUT_DIR, path)
    else:
        full_path = path

    # Initialize placeholder
    img = None

    if os.path.exists(full_path):
        # Attempt 1: Try pydicom (most robust if available)
        try:
            import pydicom

            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(np.float32)
        except (ImportError, Exception):
            pass

        # Attempt 2: Try OpenCV (sometimes works, often fails on DICOM)
        if img is None:
            try:
                # -1 flag loads image as-is (unchanged)
                img_cv = cv2.imread(full_path, -1)
                if img_cv is not None:
                    img = img_cv.astype(np.float32)
            except Exception:
                pass

        # Attempt 3: Raw Binary Fallback (Heuristic based on file size)
        # BraTS data is typically uncompressed uint16.
        if img is None:
            try:
                file_size = os.path.getsize(full_path)

                # Heuristics for common dimensions
                # 512x512 uint16 = 524,288 bytes
                # 256x256 uint16 = 131,072 bytes

                shape = None
                pixel_bytes = 0

                if file_size > 524288:
                    shape = (512, 512)
                    pixel_bytes = 512 * 512 * 2
                elif file_size > 131072:
                    shape = (256, 256)
                    pixel_bytes = 256 * 256 * 2

                if shape is not None:
                    offset = file_size - pixel_bytes
                    if offset >= 0:
                        with open(full_path, "rb") as f:
                            f.seek(offset)
                            data = np.frombuffer(f.read(), dtype=np.uint16)
                            if data.size == shape[0] * shape[1]:
                                img = data.reshape(shape).astype(np.float32)
            except Exception:
                pass

    # Handle failure or missing file
    if img is None:
        # Return a black image of target size
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Resize to target size if necessary
    if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
        try:
            img = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
        except Exception:
            # Fallback for cv2 resize failure (e.g. weird shapes)
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    return img


def normalize_min_max(volume: np.ndarray) -> np.ndarray:
    """
    Normalizes a 3D volume (or 2D image) using its global minimum and maximum.
    Scales values to the range [0, 1].
    """
    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min > 1e-6:
        return (volume - v_min) / (v_max - v_min)

    # If volume is constant (e.g. all zeros), return as is
    return volume


def get_depth_indices(total_slices: int, num_target_slices: int) -> np.ndarray:
    """
    Calculates indices to extract 'num_target_slices' uniformly from the
    10% to 90% depth range of the volume.
    """
    if total_slices == 0:
        return np.zeros(num_target_slices, dtype=int)

    # Define the 10%-90% range
    start = int(total_slices * 0.10)
    end = int(total_slices * 0.90)

    # Ensure start < end
    if end <= start:
        start = 0
        end = total_slices

    # If the range is smaller than target, we might sample outside or duplicate
    # np.linspace handles duplication (nearest neighbor interpolation in index space)
    indices = np.linspace(start, end - 1, num_target_slices)
    indices = np.clip(indices, 0, total_slices - 1).astype(int)

    return indices
