import os
import cv2
import numpy as np
from library.config import Config


def load_volume_slices(volume_dir: str, z_start: int, z_end: int) -> np.ndarray:
    """
    Loads a sequence of .tif slices from the specified volume directory.

    Args:
        volume_dir: Path to the directory containing .tif files.
        z_start: The starting slice index (inclusive).
        z_end: The ending slice index (exclusive).

    Returns:
        A 3D numpy array of shape (Depth, Height, Width).
    """
    slices = []
    # Iterate through the requested Z-range
    for z in range(z_start, z_end):
        filename = f"{z:02d}.tif"
        file_path = os.path.join(volume_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Slice file not found: {file_path}")

        # Load image in original bit-depth (uint16)
        # IMREAD_UNCHANGED is crucial to preserve the full 16-bit range
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")

        slices.append(img)

    if not slices:
        raise ValueError(
            f"No slices loaded from {volume_dir} for range {z_start}-{z_end}"
        )

    # Stack along the first dimension (Depth) to create a 3D volume
    return np.stack(slices, axis=0)


def make_overlapping_slab(volume_chunk: np.ndarray) -> np.ndarray:
    """
    Constructs a 3-channel overlapping slab (MIP) from a raw volume chunk.

    The function uses Config.Z_OFFSETS to determine which sub-ranges of the
    chunk to project for each channel. This creates a 'thick slab' representation
    that captures ink at slightly different depths.

    Args:
        volume_chunk: A 3D numpy array (Depth, Height, Width) containing the
                      raw voxel data. Depth must be sufficient to cover the
                      maximum offset defined in Config.Z_OFFSETS.

    Returns:
        A 3D numpy array of shape (Height, Width, Channels) with values
        normalized to [0, 1] (float32).
    """
    channels = []

    for start_offset, end_offset in Config.Z_OFFSETS:
        # Verify that the volume chunk is deep enough for this offset
        if end_offset > volume_chunk.shape[0]:
            raise ValueError(
                f"Volume chunk depth ({volume_chunk.shape[0]}) is insufficient "
                f"for required offset range {start_offset}-{end_offset}."
            )

        # Extract the sub-volume corresponding to this channel's depth range
        sub_volume = volume_chunk[start_offset:end_offset, :, :]

        # Compute Maximum Intensity Projection (MIP) along the depth axis
        # This flattens the sub-volume into a 2D image, keeping the brightest pixels (ink)
        mip = np.max(sub_volume, axis=0)
        channels.append(mip)

    # Stack the MIPs to form a multi-channel (H, W, C) image
    slab = np.stack(channels, axis=-1)

    # Normalize pixel values to [0, 1] based on the data type
    # The dataset analysis confirmed uint16 data (0-65535)
    if slab.dtype == np.uint16:
        slab = slab.astype(np.float32) / 65535.0
    elif slab.dtype == np.uint8:
        slab = slab.astype(np.float32) / 255.0
    else:
        # Fallback if data is already float or unknown, just cast
        slab = slab.astype(np.float32)

    return slab


def load_fragment_slab(
    fragment_id: str, volume_path: str, z_start: int, load_cached_data: bool = True
) -> np.ndarray:
    """
    Retrieves the processed slab for a specific fragment and Z-depth.
    Implements a strict caching mechanism to avoid re-processing raw 3D volumes.

    Args:
        fragment_id: The identifier of the fragment (e.g., '1', 'a').
        volume_path: Relative path to the volume directory (as found in metadata).
        z_start: The base Z-index for the slab generation.
        load_cached_data: If True, attempts to load from disk cache first.

    Returns:
        A 3D numpy array (Height, Width, 3) representing the processed input tensor.
    """
    # Define a unique cache filename based on fragment and depth
    cache_filename = f"frag_{fragment_id}_slab_{z_start}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Efficiently load the pre-computed numpy array
            return np.load(cache_path)
        except Exception:
            # If loading fails (e.g., corrupt file), silently fall through to recompute
            pass

    # 2. Compute from scratch
    # Construct the full path to the volume directory
    full_volume_dir = os.path.join(Config.INPUT_DIR, volume_path)

    # Determine the total depth needed based on configuration offsets
    # We need to load enough slices to cover the furthest end_offset
    max_offset = max(end for _, end in Config.Z_OFFSETS)
    z_end = z_start + max_offset

    # Load the raw 3D volume chunk
    volume_chunk = load_volume_slices(full_volume_dir, z_start, z_end)

    # Process the chunk into the 3-channel overlapping slab
    slab = make_overlapping_slab(volume_chunk)

    # 3. Save to cache for future runs
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, slab)

    return slab
