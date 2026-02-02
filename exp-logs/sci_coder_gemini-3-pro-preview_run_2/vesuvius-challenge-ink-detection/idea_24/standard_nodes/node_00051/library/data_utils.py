import os
import cv2
import numpy as np
from library.config import PATHS, SLAB_PARAMS


def normalize_image(image):
    """
    Normalizes image pixel values to [0, 1] using min-max scaling.

    Args:
        image: Numpy array (H, W, C) or (D, H, W).

    Returns:
        Float32 numpy array scaled to [0, 1].
    """
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    # Avoid division by zero
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(image)


def project_to_3ch(volume_chunk, slab_params=None):
    """
    Projects a 3D volume chunk into a 3-channel image using overlapping MIP slabs.

    Args:
        volume_chunk: (D, H, W) numpy array representing the Z-stack.
        slab_params: Dict containing 'thickness' and 'stride'.
                     Defaults to config.SLAB_PARAMS.

    Returns:
        (H, W, 3) numpy array representing the projected channels.
    """
    if slab_params is None:
        slab_params = SLAB_PARAMS

    thickness = slab_params["thickness"]
    stride = slab_params["stride"]
    depth, h, w = volume_chunk.shape

    channels = []

    # Calculate start indices for the sliding window
    # We expect exactly 3 channels for the standard configuration (Depth 24, Thick 12, Stride 6)
    # Indices: 0, 6, 12.
    start_indices = range(0, depth - thickness + 1, stride)

    for start in start_indices:
        # Limit to 3 channels to match ImageNet backbone requirements
        if len(channels) >= 3:
            break

        end = start + thickness
        # Extract slab: (Thickness, H, W)
        slab = volume_chunk[start:end, :, :]

        # Apply Maximum Intensity Projection along Z-axis (axis 0)
        mip = np.max(slab, axis=0)
        channels.append(mip)

    # Handle edge cases where volume might be smaller than expected
    # Pad with empty channels if necessary to ensure 3 channels
    while len(channels) < 3:
        channels.append(np.zeros((h, w), dtype=volume_chunk.dtype))

    # Stack to create (H, W, 3)
    image_3ch = np.stack(channels, axis=-1)
    return image_3ch


def load_volume_chunk(volume_dir, x, y, width, height, z_start, z_end):
    """
    Loads a specific 3D chunk from the .tif files.

    Args:
        volume_dir: Path to the directory containing .tif slices.
        x, y: Top-left coordinates of the crop.
        width, height: Dimensions of the crop.
        z_start: Starting slice index (inclusive).
        z_end: Ending slice index (exclusive).

    Returns:
        (D, H, W) numpy array containing the volume chunk.
    """
    depth = z_end - z_start

    # Initialize volume container
    # We use the requested width/height.
    volume = np.zeros((depth, height, width), dtype=np.uint16)

    for i, z in enumerate(range(z_start, z_end)):
        filename = f"{z:02d}.tif"
        path = os.path.join(volume_dir, filename)

        if not os.path.exists(path):
            # If slice is missing, leave as zeros (padding)
            continue

        # Load image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        img_h, img_w = img.shape

        # Calculate valid crop region
        # Coordinates in the source image
        src_x_start = x
        src_y_start = y
        src_x_end = min(x + width, img_w)
        src_y_end = min(y + height, img_h)

        # Coordinates in the destination volume
        dst_x_start = 0
        dst_y_start = 0
        dst_x_end = src_x_end - src_x_start
        dst_y_end = src_y_end - src_y_start

        # Check if crop is valid (inside image)
        if dst_x_end > 0 and dst_y_end > 0:
            volume[i, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]

    return volume


def get_fragment_3ch_slab(
    fragment_id, split, z_start, z_end, slab_params=None, load_cached_data=True
):
    """
    Retrieves the 3-channel projection for a full fragment.
    Implements deterministic caching to disk to speed up training.

    Args:
        fragment_id: ID of the fragment (e.g., '1', 'a').
        split: 'train' or 'test'.
        z_start, z_end: Z-range to load.
        slab_params: Projection parameters.
        load_cached_data: If True, attempts to load from disk first.

    Returns:
        (H, W, 3) float32 numpy array, normalized to [0, 1].
    """
    if slab_params is None:
        slab_params = SLAB_PARAMS

    # Construct cache filename
    cache_filename = f"frag_{fragment_id}_slab_{z_start}_{z_end}.npy"
    cache_path = os.path.join(PATHS.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(
                f"Loading cached slab for fragment {fragment_id} ({z_start}-{z_end})..."
            )
            data = np.load(cache_path)
            return data
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}. Regenerating...")

    # 2. Generate from scratch
    print(f"Generating slab for fragment {fragment_id} ({z_start}-{z_end})...")

    # Determine paths
    base_dir = PATHS.TRAIN_FRAGMENTS if split == "train" else PATHS.TEST_FRAGMENTS
    fragment_path = os.path.join(base_dir, fragment_id)
    volume_dir = os.path.join(fragment_path, "surface_volume")
    mask_path = os.path.join(fragment_path, "mask.png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found at {mask_path}")

    # Read mask to get full dimensions
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask for fragment {fragment_id}")

    h, w = mask.shape

    # Load the full volume chunk for this Z-range
    # x=0, y=0, width=w, height=h
    volume_chunk = load_volume_chunk(volume_dir, 0, 0, w, h, z_start, z_end)

    # Project to 3 channels
    slab_3ch = project_to_3ch(volume_chunk, slab_params)

    # Normalize to [0, 1] and convert to float32
    # This ensures the cached data is ready for the model
    slab_3ch = normalize_image(slab_3ch)

    # 3. Save to cache
    os.makedirs(PATHS.WORKING_DIR, exist_ok=True)
    np.save(cache_path, slab_3ch)
    print(f"Saved slab to {cache_path}")

    return slab_3ch
