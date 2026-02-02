import os
import re
import numpy as np
import cv2
import torch
from torchvision import transforms
from library.config import Config

# Define standard ImageNet normalization
# Input to this transform should be a Tensor of shape (C, H, W) in range [0, 1]
normalization_transform = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., 1.dcm, 2.dcm, 10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom_raw(path):
    """
    Reads a DICOM file as a raw binary buffer and extracts the pixel data.
    Assumes standard uncompressed 512x512 CT scan (int16).

    Args:
        path (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D array of shape (512, 512) with dtype int16.
    """
    try:
        with open(path, "rb") as f:
            b = f.read()

        # Standard CT size: 512x512 pixels * 2 bytes/pixel (int16) = 524288 bytes
        expected_bytes = 512 * 512 * 2

        if len(b) >= expected_bytes:
            # We assume pixel data is at the end of the file
            pixel_data = b[-expected_bytes:]
            img = np.frombuffer(pixel_data, dtype=np.int16)
            img = img.reshape((512, 512))
            return img
        else:
            # Fallback for unexpected file sizes (return empty black image)
            return np.zeros((512, 512), dtype=np.int16)

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return np.zeros((512, 512), dtype=np.int16)


def load_patient_scan(patient_id, dcm_path_rel):
    """
    Loads all DICOM slices for a patient from the directory.

    Args:
        patient_id (str): Unique patient identifier.
        dcm_path_rel (str): Relative path to the DICOM directory (e.g., 'train/ID...').

    Returns:
        list of np.ndarray: List of 2D raw image arrays.
    """
    full_path = os.path.join(Config.INPUT_DIR, dcm_path_rel)

    if not os.path.exists(full_path):
        print(f"Warning: Path not found {full_path}")
        return []

    files = [f for f in os.listdir(full_path) if f.lower().endswith(".dcm")]
    files.sort(key=natural_sort_key)

    scans = []
    for f in files:
        img = read_dicom_raw(os.path.join(full_path, f))
        scans.append(img)

    return scans


def get_variance_slices(scans, num_slices=Config.NUM_SLICES):
    """
    Selects the top N slices with the highest pixel variance.
    High variance typically indicates lung tissue/fibrosis vs empty space.

    Args:
        scans (list): List of 2D numpy arrays.
        num_slices (int): Number of slices to select.

    Returns:
        list: List of selected 2D numpy arrays.
    """
    if not scans:
        return [np.zeros((512, 512), dtype=np.int16) for _ in range(num_slices)]

    variances = [np.var(scan) for scan in scans]

    # Get indices of top variances
    # argsort returns ascending, so we take the tail and reverse
    sorted_indices = np.argsort(variances)
    top_indices = sorted_indices[-num_slices:][::-1]

    selected_scans = [scans[i] for i in top_indices]

    # Handle edge case where patient has fewer slices than requested
    if len(selected_scans) < num_slices:
        diff = num_slices - len(selected_scans)
        # Pad by repeating the last slice (or the only slice)
        padding = [selected_scans[-1]] * diff
        selected_scans.extend(padding)

    return selected_scans


def preprocess_image(img):
    """
    Preprocesses a single raw CT slice for the model.
    Steps: Windowing -> Normalize [0,1] -> Resize -> Stack Channels -> Normalize ImageNet.

    Args:
        img (np.ndarray): Raw 2D int16 array (512x512).

    Returns:
        torch.Tensor: Preprocessed tensor of shape (3, 224, 224).
    """
    # 1. Lung Windowing
    # WL: -600, WW: 1500 -> Range [-1350, 150]
    wl, ww = -600, 1500
    lower, upper = wl - ww // 2, wl + ww // 2

    img = np.clip(img, lower, upper)

    # 2. Normalize to [0, 1]
    img = (img - lower) / (upper - lower)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    # 3. Resize to target size
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    # 4. Stack to 3 channels (for EfficientNet)
    img = np.stack([img, img, img], axis=-1)  # (H, W, 3)

    # 5. To Tensor (HWC -> CHW)
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    # 6. Normalize (ImageNet)
    tensor = normalization_transform(tensor)

    return tensor


def get_patient_images(
    patient_id, dcm_path_rel, load_cached_data=Config.LOAD_CACHED_DATA
):
    """
    Main entry point to get processed images for a patient.
    Implements caching mechanism.

    Args:
        patient_id (str): Patient ID.
        dcm_path_rel (str): Relative path to DICOMs.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Tensor of shape (NUM_SLICES, 3, IMG_SIZE, IMG_SIZE).
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data_numpy = np.load(cache_path)
            return torch.from_numpy(data_numpy)
        except Exception as e:
            print(f"Failed to load cache for {patient_id}: {e}. Recomputing.")

    # 2. Process from scratch
    # Load raw scans
    scans = load_patient_scan(patient_id, dcm_path_rel)

    # Select slices
    selected_scans = get_variance_slices(scans, num_slices=Config.NUM_SLICES)

    # Preprocess each slice
    processed_tensors = [preprocess_image(s) for s in selected_scans]

    # Stack into a single tensor: (Num_Slices, 3, H, W)
    final_tensor = torch.stack(processed_tensors)

    # 3. Save to cache
    try:
        np.save(cache_path, final_tensor.numpy())
    except Exception as e:
        print(f"Failed to save cache for {patient_id}: {e}")

    return final_tensor
