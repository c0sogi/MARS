import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE) format.

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited list of start positions and run lengths.
             Pixels are numbered from top-to-bottom, then left-to-right
             (or left-to-right then top-to-bottom depending on flattening).
             The competition specifies: "pixels are numbered from left to right, then top to bottom".
             1 is pixel (1,1).
    """
    # Flatten mask
    pixels = mask.flatten()

    # Add padding to detect start and end of runs at edges
    pixels = np.concatenate([[0], pixels, [0]])

    # Find where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # runs[0] is start of first run, runs[1] is end of first run (exclusive), etc.
    # We need lengths, so we subtract start from end
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, smooth=1e-6):
    """
    Computes the F-Beta score.

    Args:
        preds (torch.Tensor): Predicted probabilities or logits.
        targets (torch.Tensor): Ground truth binary masks.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        threshold (float): Threshold for binarizing predictions.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = (numerator + smooth) / (denominator + smooth)
    return score.item()


def dice_coefficient(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Computes the Dice Coefficient (F1 Score).

    Args:
        preds (torch.Tensor): Predicted probabilities.
        targets (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold for binarizing predictions.
        smooth (float): Smoothing factor.

    Returns:
        float: The Dice coefficient.
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    intersection = (preds_bin * targets_bin).sum()
    union = preds_bin.sum() + targets_bin.sum()

    score = (2.0 * intersection + smooth) / (union + smooth)
    return score.item()


def _load_tiff_slice(path):
    """
    Helper to load a single TIFF slice.
    """
    if not os.path.exists(path):
        return None
    # Load as grayscale (unchanged depth, usually uint16)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def compute_slab_mips(volume_dir, z_start, z_end):
    """
    Computes the 3-channel slab using Maximum Intensity Projection (MIP)
    according to the overlapping thick slab strategy defined in Config.

    Channel 1: MIP of [z_start, z_start + 12)
    Channel 2: MIP of [z_start + 6, z_start + 18)
    Channel 3: MIP of [z_start + 12, z_start + 24)

    Note: z_end should be z_start + 24.
    """
    # Verify range consistency with Config logic
    assert (z_end - z_start) == (
        Config.SLAB_DEPTH * 2
    ), f"Z-range {z_start}-{z_end} does not match expected depth for 3 overlapping channels."

    # Pre-load all necessary slices into memory
    # We need slices from z_start to z_end (exclusive)
    slices = {}

    # Determine image dimensions from the first valid slice
    h, w = 0, 0

    for z in range(z_start, z_end):
        filename = f"{z:02d}.tif"
        path = os.path.join(volume_dir, filename)
        img = _load_tiff_slice(path)

        if img is not None:
            slices[z] = img
            if h == 0:
                h, w = img.shape
        else:
            # If a slice is missing, we will handle it during stack construction
            pass

    if h == 0 or w == 0:
        raise ValueError(
            f"No valid slices found in {volume_dir} for range {z_start}-{z_end}"
        )

    # Helper to compute MIP for a sub-range
    def get_mip(start, end):
        stack = []
        for z in range(start, end):
            if z in slices:
                stack.append(slices[z])
            else:
                # Pad with zeros if slice is missing
                stack.append(np.zeros((h, w), dtype=np.uint16))

        if not stack:
            return np.zeros((h, w), dtype=np.float32)

        # Stack and take max along depth axis
        stack_arr = np.stack(stack, axis=0)
        mip = np.max(stack_arr, axis=0)
        return mip

    # Define channel ranges based on Config
    # Stride is 6, Depth is 12
    c1_range = (z_start, z_start + 12)
    c2_range = (z_start + 6, z_start + 18)
    c3_range = (z_start + 12, z_start + 24)

    mip1 = get_mip(*c1_range)
    mip2 = get_mip(*c2_range)
    mip3 = get_mip(*c3_range)

    # Stack into (H, W, 3)
    slab = np.stack([mip1, mip2, mip3], axis=-1)

    # Normalize to [0, 1] and convert to float32
    # Assuming 16-bit input (0-65535)
    slab = slab.astype(np.float32) / 65535.0

    return slab


def get_fragment_slab(fragment_id, volume_path, z_range, load_cached_data=True):
    """
    Retrieves the 3-channel slab for a specific fragment and z-range.
    Implements caching to ./working/idea_23/ to avoid re-computing MIPs.

    Args:
        fragment_id (str): ID of the fragment (e.g., '1', 'a').
        volume_path (str): Path to the surface_volume directory.
        z_range (tuple): (start, end) tuple defining the Z-stack.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: 3-channel image of shape (H, W, 3), float32 normalized.
    """
    z_start, z_end = z_range
    cache_filename = f"frag_{fragment_id}_slab_{z_start}_{z_end}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            slab = np.load(cache_path)
            # print(f"Loaded cached slab for fragment {fragment_id} ({z_start}-{z_end})")
            return slab
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

    # 2. Compute from scratch
    full_volume_path = os.path.join(Config.INPUT_DIR, volume_path)
    slab = compute_slab_mips(full_volume_path, z_start, z_end)

    # 3. Save to cache
    try:
        np.save(cache_path, slab)
        # print(f"Cached slab for fragment {fragment_id} ({z_start}-{z_end})")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return slab
