import os
import cv2
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility using the Config class.
    """
    Config.set_seed(seed)


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask (0 or 1), 2D array.

    Returns:
        str: Space-delimited list of start positions and run lengths.
             Pixels are numbered from left to right, then top to bottom, starting at 1.
    """
    pixels = mask.flatten()
    # Pad with 0s to detect starts and ends of runs
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths: end - start
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def normalize_image(image):
    """
    Normalizes a uint16 image to float32 in range [0, 1] using min-max scaling.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Normalized float32 image.
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    else:
        image = np.zeros_like(image)

    return image


def load_volume_slice(volume_dir, z_index, x, y, width, height):
    """
    Loads a specific crop from a specific Z-slice in the volume.

    Args:
        volume_dir (str): Path to the directory containing .tif slices.
        z_index (int): The Z-index of the slice to load.
        x, y (int): Top-left coordinates of the crop.
        width, height (int): Dimensions of the crop.

    Returns:
        np.ndarray: The cropped image slice.
    """
    filename = f"{z_index:02d}.tif"
    filepath = os.path.join(volume_dir, filename)

    if not os.path.exists(filepath):
        # Fallback for missing files (e.g., if z is out of bounds), return zero array
        return np.zeros((height, width), dtype=np.uint16)

    # Load the full slice
    # Note: cv2.imread loads the whole image. For very large TIFs, memmap is better,
    # but given the constraints and libraries, we use cv2.
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img is None:
        return np.zeros((height, width), dtype=np.uint16)

    # Handle boundary conditions
    img_h, img_w = img.shape

    # Calculate valid crop region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + width)
    y2 = min(img_h, y + height)

    # Crop
    crop = img[y1:y2, x1:x2]

    # Pad if the crop is smaller than requested (e.g., at edges)
    pad_h = height - (y2 - y1)
    pad_w = width - (x2 - x1)

    if pad_h > 0 or pad_w > 0:
        crop = np.pad(
            crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    return crop


def project_slab(volume_dir, start_z, thickness, x, y, width, height):
    """
    Loads a stack of slices and performs Maximum Intensity Projection (MIP).

    Args:
        volume_dir (str): Path to volume directory.
        start_z (int): Starting Z-index.
        thickness (int): Number of slices to aggregate.
        x, y, width, height (int): Crop definition.

    Returns:
        np.ndarray: 2D projection of the slab.
    """
    slices = []
    for z in range(start_z, start_z + thickness):
        slice_img = load_volume_slice(volume_dir, z, x, y, width, height)
        slices.append(slice_img)

    stack = np.stack(slices, axis=0)
    # Maximum Intensity Projection (MIP) to capture ink
    projection = np.max(stack, axis=0)
    return projection


def generate_multiview_tensor(volume_dir, view_start_z, x, y, width, height):
    """
    Generates the 3-channel input tensor for the Translation-Invariant model.
    Based on the 'Overlapping Thick Slab' strategy defined in Config.

    Channels:
        0: Slab [z, z + thickness]
        1: Slab [z + overlap, z + overlap + thickness]
        2: Slab [z + 2*overlap, z + 2*overlap + thickness]

    Args:
        volume_dir (str): Path to volume directory.
        view_start_z (int): Base Z-index for the view (e.g., 16, 20, or 24).
        x, y, width, height (int): Crop definition.

    Returns:
        torch.Tensor: Normalized tensor of shape (3, Height, Width).
    """
    channels = []

    # Channel 0
    start_0 = view_start_z
    proj_0 = project_slab(
        volume_dir, start_0, Config.SLAB_THICKNESS, x, y, width, height
    )
    channels.append(normalize_image(proj_0))

    # Channel 1
    start_1 = view_start_z + Config.SLAB_OVERLAP
    proj_1 = project_slab(
        volume_dir, start_1, Config.SLAB_THICKNESS, x, y, width, height
    )
    channels.append(normalize_image(proj_1))

    # Channel 2
    start_2 = view_start_z + (Config.SLAB_OVERLAP * 2)
    proj_2 = project_slab(
        volume_dir, start_2, Config.SLAB_THICKNESS, x, y, width, height
    )
    channels.append(normalize_image(proj_2))

    # Stack to (H, W, 3) then transpose to (3, H, W)
    tensor_np = np.stack(channels, axis=-1)  # (H, W, 3)
    tensor_np = np.transpose(tensor_np, (2, 0, 1))  # (3, H, W)

    return torch.from_numpy(tensor_np).float()


def load_or_process_data(file_name, process_func, load_cached_data=True, **kwargs):
    """
    Generic caching mechanism for deterministic data processing.

    Args:
        file_name (str): Name of the cache file (e.g., 'data.npy').
        process_func (callable): Function to compute data if cache is missing.
        load_cached_data (bool): Whether to attempt loading from cache.
        **kwargs: Arguments passed to process_func.

    Returns:
        The loaded or computed data.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    file_path = os.path.join(Config.CACHE_DIR, file_name)

    if load_cached_data and os.path.exists(file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            return data
        except Exception as e:
            print(f"Failed to load cache {file_path}: {e}. Recomputing...")

    # Compute data
    data = process_func(**kwargs)

    # Save data
    try:
        np.save(file_path, data)
    except Exception as e:
        print(f"Warning: Could not save cache to {file_path}: {e}")

    return data


def write_submission(ids, rles, path=Config.SUBMISSION_PATH):
    """
    Writes the submission CSV file.

    Args:
        ids (list): List of fragment IDs (e.g., ['a', 'b']).
        rles (list): List of RLE strings corresponding to IDs.
        path (str): Output path.
    """
    df = pd.DataFrame({"Id": ids, "Predicted": rles})
    df.to_csv(path, index=False)
    print(f"Submission saved to {path}")
