import os
import cv2
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets fixed random seeds for reproducibility.
    Wrapper around Config.set_seed.
    """
    Config.set_seed(seed)


def load_metadata(split="train"):
    """
    Loads the metadata CSV for a specific split.

    Args:
        split (str): One of 'train', 'validation', 'test'.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    if split == "train":
        path = Config.TRAIN_METADATA
    elif split == "validation":
        path = Config.VALIDATION_METADATA
    elif split == "test":
        path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    return pd.read_csv(path)


def load_image(path, grayscale=True):
    """
    Loads an image from the input directory.

    Args:
        path (str): Relative path to the image from input root.
        grayscale (bool): If True, loads as grayscale.

    Returns:
        np.ndarray: The image array, or None if not found.
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        return None

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    return cv2.imread(full_path, flags)


def load_volume_slice(volume_dir, z_index):
    """
    Loads a specific Z-slice from the 3D volume directory.

    Args:
        volume_dir (str): Relative path to the volume directory (e.g., 'train/1/surface_volume').
        z_index (int): The index of the slice to load.

    Returns:
        np.ndarray: The 16-bit image slice, or None if not found.
    """
    filename = f"{z_index:02d}.tif"
    full_path = os.path.join(Config.INPUT_DIR, volume_dir, filename)

    if not os.path.exists(full_path):
        return None

    # Load as unchanged to preserve uint16 depth
    return cv2.imread(full_path, cv2.IMREAD_UNCHANGED)


def process_fragment_mips(fragment_id, volume_path, z_start, load_cached_data=True):
    """
    Generates or loads the 3-channel Maximum Intensity Projection (MIP) tensor for a fragment.

    Implements the "Overlapping Stratified Depth Projection" strategy.
    Channels are MIPs of slabs: [z, z+12], [z+6, z+18], [z+12, z+24].

    Args:
        fragment_id (str): ID of the fragment (e.g., '1', 'a').
        volume_path (str): Relative path to the volume directory.
        z_start (int): The starting Z-index for the projection context.
        load_cached_data (bool): If True, attempts to load from disk cache.

    Returns:
        np.ndarray: A (3, H, W) float32 tensor normalized to [0, 1].
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_filename = f"frag_{fragment_id}_start_{z_start}_mips.npy"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # print(f"Loading cached MIPs for fragment {fragment_id} at z={z_start}...")
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    # print(f"Computing MIPs for fragment {fragment_id} at z={z_start}...")

    channel_ranges = Config.get_channel_ranges(z_start)
    channels = []

    for start_z, end_z in channel_ranges:
        slab_slices = []
        for z in range(start_z, end_z):
            img_slice = load_volume_slice(volume_path, z)
            if img_slice is None:
                # If slice is missing (e.g. out of bounds), we skip it.
                # In a robust pipeline, we might pad with zeros, but here we assume valid ranges.
                continue
            slab_slices.append(img_slice)

        if not slab_slices:
            raise ValueError(
                f"No slices found for range {start_z}-{end_z} in {volume_path}"
            )

        # Stack slices for this channel: (D, H, W)
        slab_stack = np.stack(slab_slices, axis=0)

        # Compute MIP (Maximum Intensity Projection) along depth
        mip = np.max(slab_stack, axis=0)
        channels.append(mip)

    # Stack channels: (3, H, W)
    img_tensor = np.stack(channels, axis=0)

    # Normalize uint16 [0, 65535] -> float32 [0, 1]
    img_tensor = img_tensor.astype(np.float32) / 65535.0

    # 3. Save to cache
    np.save(cache_path, img_tensor)

    return img_tensor


def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE) for the competition metric.

    The metric checks that pairs are sorted, positive, and decoded pixel values are not duplicated.
    Pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1).

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of start positions and run lengths.
    """
    # Flatten the mask in row-major order (C-style)
    pixels = mask.flatten()

    # We need to find transitions from 0 to 1 and 1 to 0.
    # Pad with 0 at both ends to detect runs at the very beginning or end.
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs array now looks like: [start1, end1, start2, end2, ...]
    # The length of the run is end - start.
    # We modify the end indices (at odd positions) to be lengths.
    runs[1::2] -= runs[::2]

    # Convert to string
    return " ".join(str(x) for x in runs)


def save_checkpoint(model, optimizer, epoch, score, filename="best_model.pth"):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch.
        score (float): Validation score (F0.5).
        filename (str): Name of the file to save in the cache directory.
    """
    save_path = os.path.join(Config.CACHE_DIR, filename)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "score": score,
    }

    torch.save(state, save_path)
    # print(f"Checkpoint saved: {save_path} (Score: {score:.5f})")
