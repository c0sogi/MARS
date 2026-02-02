import os
import cv2
import numpy as np
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encoding(x):
    """
    Run-length encoding for binary masks.

    Args:
        x (np.ndarray): Binary mask of shape (height, width), where 1 indicates ink.

    Returns:
        str: Space-delimited run-length encoded string.
    """
    dots = np.where(x.flatten() == 1)[0]
    if len(dots) == 0:
        return ""

    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return " ".join(map(str, run_lengths))


def dice_coef(preds, targets, threshold=0.5, epsilon=1e-6):
    """
    Computes the Dice Coefficient.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions (probabilities or logits).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels.
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice coefficient score.
    """
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure inputs are float
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    intersection = (preds_bin * targets_bin).sum()
    union = preds_bin.sum() + targets_bin.sum()

    dice = (2.0 * intersection) / (union + epsilon)
    return dice.item()


def fbeta_score(preds, targets, beta=0.5, threshold=0.5, epsilon=1e-7):
    """
    Computes the F-beta score (default beta=0.5 for F0.5).

    Args:
        preds (torch.Tensor or np.ndarray): Predictions (probabilities).
        targets (torch.Tensor or np.ndarray): Ground truth binary labels.
        beta (float): Weight of precision in harmonic mean.
        threshold (float): Threshold to convert probabilities to binary mask.
        epsilon (float): Smoothing factor.

    Returns:
        float: F-beta score.
    """
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    tp = (preds_bin * targets_bin).sum()
    fp = (preds_bin * (1 - targets_bin)).sum()
    fn = ((1 - preds_bin) * targets_bin).sum()

    beta_sq = beta**2
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    fbeta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
    )
    return fbeta.item()


def load_volume_slab(volume_dir, z_start, z_depth):
    """
    Loads a slab of 3D x-ray scans from disk and normalizes them.

    Args:
        volume_dir (str): Path to the directory containing .tif slices.
        z_start (int): The starting slice index (inclusive).
        z_depth (int): The number of slices to load.

    Returns:
        np.ndarray: A 3D array of shape (z_depth, height, width) with values
                    normalized to [0, 1]. Dtype is float32.
    """
    slices = []

    for i in range(z_depth):
        z_idx = z_start + i
        # Clamp index to valid range to handle boundary conditions gracefully
        # (e.g. if scanning near the top/bottom of the stack)
        z_idx = max(0, min(z_idx, Config.NUM_SLICES_PER_FRAGMENT - 1))

        filename = f"{z_idx:02d}.tif"
        path = os.path.join(volume_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Slice {path} not found.")

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image {path}")

        slices.append(img)

    stack = np.stack(slices, axis=0)

    # Normalize uint16 data to [0, 1]
    stack = stack.astype(np.float32) / 65535.0

    return stack


def get_cached_volume(fragment_id, volume_dir, load_cached_data=True):
    """
    Loads the entire volume for a fragment, using caching to speed up subsequent access.

    Args:
        fragment_id (str): Fragment ID.
        volume_dir (str): Path to surface_volume directory.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Full volume (65, H, W) normalized float32.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{fragment_id}_volume.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            return volume
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

    # Recompute: Load all 65 slices
    volume = load_volume_slab(volume_dir, 0, Config.NUM_SLICES_PER_FRAGMENT)

    # Save to cache for future runs
    np.save(cache_path, volume)

    return volume


def create_3ch_input(volume_slab):
    """
    Converts a raw volume slab into a 3-channel input tensor using MIPs
    defined in Config.Z_SLAB_CHANNELS.

    Args:
        volume_slab (np.ndarray): Shape (D, H, W), where D should match Config.Z_CONTEXT_DEPTH.

    Returns:
        np.ndarray: Shape (H, W, 3) ready for Albumentations (channels last).
    """
    channels = []
    # Config.Z_SLAB_CHANNELS is list of tuples (start_rel, end_rel)
    for start_rel, end_rel in Config.Z_SLAB_CHANNELS:
        # Handle potential edge cases if slab is smaller than expected
        s = max(0, start_rel)
        e = min(volume_slab.shape[0], end_rel)

        sub_slab = volume_slab[s:e]

        if sub_slab.shape[0] == 0:
            # Fallback for empty slab (should not happen with correct config)
            mip = np.zeros(volume_slab.shape[1:], dtype=np.float32)
        else:
            mip = np.max(sub_slab, axis=0)
        channels.append(mip)

    # Stack to (H, W, 3)
    image = np.stack(channels, axis=-1)
    return image


def get_transforms(data="train"):
    """
    Returns Albumentations transforms for the dataset.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        return A.Compose([ToTensorV2()])
    else:
        return A.Compose([ToTensorV2()])
