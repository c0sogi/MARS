import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from library.config import Config


def load_fragment_data(fragment_id, split, load_cached_data=True):
    """
    Loads the volume, mask, and label (if available) for a given fragment.
    Uses caching to speed up subsequent loads.

    Args:
        fragment_id (str): The ID of the fragment (e.g., '1', 'a').
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Contains 'volume', 'mask', 'label', and 'fragment_id'.
    """
    # Define cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    vol_cache_path = cache_dir / f"{fragment_id}_volume.npy"
    mask_cache_path = cache_dir / f"{fragment_id}_mask.npy"
    label_cache_path = cache_dir / f"{fragment_id}_label.npy"

    # Try loading from cache
    if load_cached_data and vol_cache_path.exists() and mask_cache_path.exists():
        # Check if label is expected and exists (test set has no labels)
        has_label = split != "test"
        if not has_label or label_cache_path.exists():
            try:
                volume = np.load(vol_cache_path)
                mask = np.load(mask_cache_path)
                label = np.load(label_cache_path) if has_label else None
                return {
                    "volume": volume,
                    "mask": mask,
                    "label": label,
                    "fragment_id": fragment_id,
                }
            except Exception as e:
                print(
                    f"Failed to load cached data for {fragment_id}: {e}. Reloading from source."
                )

    # Load from source
    # 1. Get paths from metadata
    meta_path = Config.METADATA_DIR / f"{split}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    # Ensure fragment_id is string for comparison
    row = df[df["fragment_id"].astype(str) == str(fragment_id)]
    if row.empty:
        raise ValueError(f"Fragment {fragment_id} not found in {split} metadata.")

    row = row.iloc[0]
    vol_rel_path = row["surface_volume_path"]
    mask_rel_path = row["mask_path"]
    label_rel_path = row.get("inklabels_path", None)

    # 2. Load Volume
    vol_dir = Config.INPUT_DIR / vol_rel_path
    if not vol_dir.exists():
        raise FileNotFoundError(f"Volume directory not found: {vol_dir}")

    # Load all slices 00.tif to 64.tif
    slices = []
    for z in range(Config.Z_DIM):
        slice_path = vol_dir / f"{z:02d}.tif"
        if not slice_path.exists():
            raise FileNotFoundError(f"Slice {slice_path} not found.")

        img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {slice_path}")
        slices.append(img)

    volume = np.stack(slices, axis=0)  # Shape: (65, H, W)

    # 3. Load Mask
    mask_path = Config.INPUT_DIR / mask_rel_path
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    # Binarize mask
    mask = (mask > 0).astype(np.uint8)

    # 4. Load Label (if applicable)
    label = None
    if split != "test" and pd.notna(label_rel_path):
        lbl_path = Config.INPUT_DIR / label_rel_path
        label = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to read label: {lbl_path}")
        label = (label > 0).astype(np.uint8)

    # 5. Save to cache
    np.save(vol_cache_path, volume)
    np.save(mask_cache_path, mask)
    if label is not None:
        np.save(label_cache_path, label)

    return {"volume": volume, "mask": mask, "label": label, "fragment_id": fragment_id}


def get_normalization_stats(fragment_ids, load_cached_data=True):
    """
    Calculates or loads global mean and std for the training set.
    """
    stats_path = Config.NORMALIZATION_STATS_PATH

    if load_cached_data and stats_path.exists():
        try:
            stats = np.load(stats_path, allow_pickle=True).item()
            return stats["mean"], stats["std"]
        except Exception as e:
            print(f"Failed to load stats: {e}. Recalculating.")

    print("Calculating global normalization stats...")
    sum_val = 0.0
    sum_sq_val = 0.0
    count = 0

    # Sample 5% of pixels from each fragment's masked area for efficiency
    sample_rate = 0.05

    for fid in fragment_ids:
        # We assume these are training fragments
        data = load_fragment_data(fid, split="train", load_cached_data=load_cached_data)
        vol = data["volume"]  # (65, H, W)
        mask = data["mask"]  # (H, W)

        # Get valid indices
        ys, xs = np.where(mask > 0)
        num_pixels = len(ys)
        if num_pixels == 0:
            continue

        num_samples = int(num_pixels * sample_rate)
        if num_samples < 1000:
            num_samples = min(num_pixels, 1000)

        # Randomly sample indices
        indices = np.random.choice(num_pixels, num_samples, replace=False)
        sample_ys = ys[indices]
        sample_xs = xs[indices]

        # Extract samples: (65, num_samples)
        samples = vol[:, sample_ys, sample_xs].astype(np.float32)

        sum_val += np.sum(samples)
        sum_sq_val += np.sum(samples**2)
        count += samples.size

    mean = sum_val / count
    variance = (sum_sq_val / count) - (mean**2)
    std = np.sqrt(variance)

    # Save
    np.save(stats_path, {"mean": mean, "std": std})
    print(f"Global Stats Calculated: Mean={mean:.4f}, Std={std:.4f}")

    return mean, std


class InkDataset(Dataset):
    def __init__(
        self,
        split,
        fragment_ids=None,
        mode="train",
        load_cached_data=True,
        normalization_stats=None,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            fragment_ids (list): List of fragment IDs to use. If None, loads all from metadata.
            mode (str): 'train' (augments), 'val' (no augment), 'test' (no labels).
            load_cached_data (bool): Whether to use cached .npy files.
            normalization_stats (tuple): (mean, std) for Z-score. If None, loads/computes.
        """
        self.split = split
        self.mode = mode
        self.patch_size = Config.PATCH_SIZE
        self.half_size = self.patch_size // 2

        # 1. Determine Fragment IDs
        if fragment_ids is None:
            meta_path = Config.METADATA_DIR / f"{split}.csv"
            if not meta_path.exists():
                raise FileNotFoundError(f"Metadata not found: {meta_path}")
            df = pd.read_csv(meta_path)
            self.fragment_ids = df["fragment_id"].astype(str).tolist()
        else:
            self.fragment_ids = [str(x) for x in fragment_ids]

        # 2. Load Data
        self.volumes = {}
        self.masks = {}
        self.labels = {}

        for fid in self.fragment_ids:
            data = load_fragment_data(fid, split, load_cached_data=load_cached_data)
            self.volumes[fid] = data[
                "volume"
            ]  # Keep as uint8 to save memory until getitem
            self.masks[fid] = data["mask"]
            self.labels[fid] = data["label"]

        # 3. Normalization Stats
        if normalization_stats is None:
            stats_path = Config.NORMALIZATION_STATS_PATH
            if stats_path.exists():
                stats = np.load(stats_path, allow_pickle=True).item()
                self.mean = stats["mean"]
                self.std = stats["std"]
            else:
                if mode == "train":
                    self.mean, self.std = get_normalization_stats(
                        self.fragment_ids, load_cached_data
                    )
                else:
                    print(
                        "Warning: Normalization stats not found and not provided. Using default (0, 1)."
                    )
                    self.mean = 0.0
                    self.std = 1.0
        else:
            self.mean, self.std = normalization_stats

        # 4. Indexing (Valid Centers)
        self.indices = []
        for fid in self.fragment_ids:
            mask = self.masks[fid]
            h, w = mask.shape

            # Create a valid map for patch centers
            # Center must be far enough from edges so the patch fits
            valid_map = np.zeros_like(mask)
            valid_map[
                self.half_size : h - self.half_size, self.half_size : w - self.half_size
            ] = 1

            # Center must also be inside the valid fragment mask
            valid_map = valid_map & (mask > 0)

            ys, xs = np.where(valid_map > 0)

            # Store as (fid_idx, y, x)
            fid_idx = self.fragment_ids.index(fid)
            f_idx_arr = np.full(len(ys), fid_idx, dtype=np.int16)

            current_indices = np.stack([f_idx_arr, ys, xs], axis=1)
            self.indices.append(current_indices)

        if self.indices:
            self.indices = np.concatenate(self.indices, axis=0)
        else:
            self.indices = np.array([])

        # Shuffle indices to ensure representative sampling if we truncate for epoch length
        rng = np.random.default_rng(Config.SEED)
        rng.shuffle(self.indices)

        # 5. Limit size per epoch
        if Config.MAX_PATCHES_PER_EPOCH:
            self.epoch_length = min(len(self.indices), Config.MAX_PATCHES_PER_EPOCH)
        else:
            self.epoch_length = len(self.indices)

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # If training, we can sample randomly from the full set to ensure variety across epochs
        # if the epoch length is smaller than dataset size.
        # However, since we shuffled in init, iterating sequentially (idx) is also random.
        # To make it truly stochastic per epoch if epoch_length < len(indices), we can pick random idx.
        if (
            self.mode == "train"
            and Config.MAX_PATCHES_PER_EPOCH
            and len(self.indices) > 0
        ):
            actual_idx = np.random.randint(0, len(self.indices))
        else:
            actual_idx = idx

        fid_idx, cy, cx = self.indices[actual_idx]
        fid = self.fragment_ids[fid_idx]

        # Extract Patch
        y_start = cy - self.half_size
        y_end = cy + self.half_size
        x_start = cx - self.half_size
        x_end = cx + self.half_size

        # Volume: (65, 256, 256)
        vol_patch = self.volumes[fid][:, y_start:y_end, x_start:x_end]

        # Convert to float and normalize
        vol_patch = (vol_patch.astype(np.float32) - self.mean) / self.std

        # Label
        if self.labels[fid] is not None:
            label_patch = self.labels[fid][y_start:y_end, x_start:x_end]
            label_patch = label_patch.astype(np.float32)
            # Add channel dim: (1, 256, 256)
            label_patch = np.expand_dims(label_patch, axis=0)
        else:
            # Dummy label for test
            label_patch = np.zeros(
                (1, self.patch_size, self.patch_size), dtype=np.float32
            )

        # Augmentations
        if self.mode == "train":
            # 1. Rotations (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                # rotate axes (1, 2) which are H, W
                vol_patch = np.rot90(vol_patch, k, axes=(1, 2))
                label_patch = np.rot90(label_patch, k, axes=(1, 2))

            # 2. Flips
            if random.random() < 0.5:
                # Horizontal flip (axis 2)
                vol_patch = np.flip(vol_patch, axis=2)
                label_patch = np.flip(label_patch, axis=2)

            if random.random() < 0.5:
                # Vertical flip (axis 1)
                vol_patch = np.flip(vol_patch, axis=1)
                label_patch = np.flip(label_patch, axis=1)

        # Convert to Tensor (copy to handle negative strides from flips)
        vol_tensor = torch.from_numpy(vol_patch.copy())
        label_tensor = torch.from_numpy(label_patch.copy())

        return vol_tensor, label_tensor
