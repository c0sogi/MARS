import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encoding(mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE).
    The mask is flattened in row-major order (left-to-right, then top-to-bottom).

    Args:
        mask (np.ndarray): Binary mask (0 or 1) of shape (H, W).

    Returns:
        str: Space-delimited RLE string.
    """
    pixels = mask.flatten()
    # Prepend and append 0 to detect start and end of runs
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # runs[0] is start of first run, runs[1] is end of first run, etc.
    # We need (start, length) pairs.
    # The end index in 'runs' is exclusive relative to the run, so length = end - start.
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score for binary segmentation.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted probabilities or binary map.
        targets (torch.Tensor or np.ndarray): Ground truth binary map.
        beta (float): Beta value for F-score (default 0.5 weights precision higher).
        threshold (float): Threshold to binarize predictions if they are probabilities.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: The F-beta score.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions
    preds_bin = (preds > threshold).astype(np.float32)
    targets_bin = targets.astype(np.float32)

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    beta_sq = beta**2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

    score = numerator / (denominator + epsilon)
    return float(score)


def load_volume(fragment_id, split, load_cached_data=True):
    """
    Loads the 3D volume, mask, and labels for a specific fragment.
    Handles caching to .npy files in Config.CACHE_DIR.

    Args:
        fragment_id (str or int): The fragment identifier (e.g., '1', '2', 'a').
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (volume, mask, label)
            - volume (np.ndarray): 3D array of shape (Z, H, W), float32.
            - mask (np.ndarray): 2D binary array of shape (H, W), uint8.
            - label (np.ndarray or None): 2D binary array of shape (H, W), uint8. None for test split.
    """
    fragment_id = str(fragment_id)

    # Define cache paths
    cache_vol_path = Config.CACHE_DIR / f"{fragment_id}_volume.npy"
    cache_mask_path = Config.CACHE_DIR / f"{fragment_id}_mask.npy"
    cache_label_path = Config.CACHE_DIR / f"{fragment_id}_label.npy"

    # 1. Try loading from cache
    if load_cached_data:
        has_vol = cache_vol_path.exists()
        has_mask = cache_mask_path.exists()
        # For test split, label cache is not expected/required
        has_label = cache_label_path.exists() if split != "test" else True

        if has_vol and has_mask and has_label:
            try:
                volume = np.load(cache_vol_path)
                mask = np.load(cache_mask_path)
                label = np.load(cache_label_path) if split != "test" else None
                return volume, mask, label
            except Exception as e:
                print(
                    f"Failed to load cache for fragment {fragment_id}: {e}. Recomputing..."
                )

    # 2. Load from source (if cache miss or load_cached_data=False)

    # Determine which metadata file to use
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid split: {split}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Read metadata
    df = pd.read_csv(meta_path)
    df["fragment_id"] = df["fragment_id"].astype(str)
    row = df[df["fragment_id"] == fragment_id]

    if row.empty:
        raise ValueError(f"Fragment ID {fragment_id} not found in {split} metadata.")

    row = row.iloc[0]

    # Construct full paths (metadata paths are relative to input dir)
    vol_dir = Config.INPUT_DIR / row["surface_volume_path"]
    mask_path = Config.INPUT_DIR / row["mask_path"]

    # Load Mask
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Failed to read mask image: {mask_path}")
    mask = (mask_img > 0).astype(np.uint8)

    # Load Label (only for train/val)
    label = None
    if split != "test":
        label_path = Config.INPUT_DIR / row["inklabels_path"]
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        label_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            raise ValueError(f"Failed to read label image: {label_path}")
        label = (label_img > 0).astype(np.uint8)

    # Load Volume (stack of .tif slices)
    h, w = mask.shape
    volume = np.zeros((Config.Z_DIM, h, w), dtype=np.float32)

    for z in range(Config.Z_DIM):
        slice_name = f"{z:02d}.tif"
        slice_path = vol_dir / slice_name

        if slice_path.exists():
            img_slice = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
            if img_slice is not None:
                volume[z] = img_slice.astype(np.float32)
            else:
                print(f"Warning: Failed to read slice {slice_path}")
        else:
            # It is possible some slices are missing or named differently,
            # but usually the dataset is consistent.
            print(f"Warning: Slice file does not exist: {slice_path}")

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_vol_path, volume)
    np.save(cache_mask_path, mask)
    if label is not None:
        np.save(cache_label_path, label)

    return volume, mask, label
