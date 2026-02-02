import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VALIDATION_METADATA_PATH,
    TEST_METADATA_PATH,
    Z_START,
    Z_END,
    TILE_SIZE,
    PIXEL_MEAN,
    PIXEL_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


def get_fragment_projections(fragment_id, volume_path, load_cached_data=True):
    """
    Computes or loads cached 3-channel statistical projections (Max, Mean, Std)
    for a given fragment over the Z-slice range [Z_START, Z_END).

    Args:
        fragment_id (str): ID of the fragment.
        volume_path (str): Relative path to the volume directory.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Normalized 3-channel projection of shape (H, W, 3).
    """
    # Ensure cache directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"{fragment_id}_projections.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to re-computing if load fails

    # 2. Compute from scratch
    full_volume_dir = os.path.join(INPUT_DIR, volume_path)
    slices = []

    # Iterate over the specific Z-range
    for z in range(Z_START, Z_END):
        filename = f"{z:02d}.tif"
        path = os.path.join(full_volume_dir, filename)

        if not os.path.exists(path):
            continue

        # Load image (uint16)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        slices.append(img)

    if not slices:
        raise FileNotFoundError(
            f"No slices found in {full_volume_dir} for range {Z_START}-{Z_END}"
        )

    # Stack -> (H, W, Depth)
    volume = np.stack(slices, axis=-1)

    # Compute Statistical Projections -> (H, W)
    # Channel 1: Max Intensity (MIP)
    proj_max = np.max(volume, axis=-1).astype(np.float32)
    # Channel 2: Mean Intensity
    proj_mean = np.mean(volume, axis=-1).astype(np.float32)
    # Channel 3: Standard Deviation
    proj_std = np.std(volume, axis=-1).astype(np.float32)

    # Stack -> (H, W, 3)
    projections = np.stack([proj_max, proj_mean, proj_std], axis=-1)

    # Normalize using global statistics
    # (x - mean) / std
    projections = (projections - PIXEL_MEAN) / PIXEL_STD

    # 3. Save to cache
    np.save(cache_path, projections)

    return projections


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specific phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


class InkDataset(Dataset):
    def __init__(
        self, metadata_path, phase="train", load_cached_data=True, max_samples=None
    ):
        """
        Dataset for Vesuvius Ink Detection.

        Args:
            metadata_path (str): Path to the CSV metadata file.
            phase (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Use cached .npy files if available.
            max_samples (int): Limit dataset size for debugging.
        """
        self.phase = phase
        self.df = pd.read_csv(metadata_path)

        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]

        self.transforms = get_transforms(phase)

        # --- Pre-load Data ---
        # We load full fragment projections into memory.
        self.fragments = {}
        self.masks = {}  # Valid area masks
        self.labels = {}  # Ink labels (ground truth)

        # Identify unique fragments in this split
        unique_frags = self.df[["fragment_id", "volume_path"]].drop_duplicates()

        for _, row in unique_frags.iterrows():
            fid = str(row["fragment_id"])
            vpath = row["volume_path"]

            # Load Projection
            self.fragments[fid] = get_fragment_projections(fid, vpath, load_cached_data)

            # Load Mask (Area of Interest)
            if "mask_path" in self.df.columns:
                mpath_rel = self.df[self.df["fragment_id"] == row["fragment_id"]].iloc[
                    0
                ]["mask_path"]
                mpath = os.path.join(INPUT_DIR, mpath_rel)
                if os.path.exists(mpath):
                    mask_img = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                    self.masks[fid] = (
                        (mask_img // 255).astype(np.uint8)
                        if mask_img is not None
                        else None
                    )

            # Load Label (Ground Truth) - Only for train/val
            if phase in ["train", "validation"] and "label_path" in self.df.columns:
                lpath_rel = self.df[self.df["fragment_id"] == row["fragment_id"]].iloc[
                    0
                ]["label_path"]
                lpath = os.path.join(INPUT_DIR, lpath_rel)
                if os.path.exists(lpath):
                    label_img = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
                    self.labels[fid] = (
                        (label_img // 255).astype(np.float32)
                        if label_img is not None
                        else None
                    )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = str(row["fragment_id"])

        # Retrieve full fragment data
        proj = self.fragments[fid]  # (H, W, 3)

        if self.phase == "test":
            # For test, return the full image and let inference loop handle tiling
            image = proj
            mask = self.masks.get(fid, np.zeros(proj.shape[:2], dtype=np.uint8))

            transformed = self.transforms(image=image)
            image_tensor = transformed["image"]

            return image_tensor, mask, fid

        else:
            # For Train/Val, crop specific patch
            x, y = row["x"], row["y"]
            w, h = row["width"], row["height"]

            # Crop Image
            img_h, img_w = proj.shape[:2]
            y_end = min(y + h, img_h)
            x_end = min(x + w, img_w)

            image_patch = proj[y:y_end, x:x_end, :]

            # Crop Label
            label_full = self.labels[fid]
            label_patch = label_full[y:y_end, x:x_end]

            # Apply Transforms
            # Pass label as 'mask' to apply same geometric transforms
            transformed = self.transforms(image=image_patch, mask=label_patch)

            image_tensor = transformed["image"]
            label_tensor = transformed["mask"]

            # Add channel dimension to label: (H, W) -> (1, H, W)
            label_tensor = label_tensor.unsqueeze(0)

            return image_tensor, label_tensor


def get_loaders(
    train_metadata=TRAIN_METADATA_PATH,
    val_metadata=VALIDATION_METADATA_PATH,
    test_metadata=TEST_METADATA_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    max_train_samples=None,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Train
    train_ds = InkDataset(train_metadata, phase="train", max_samples=max_train_samples)

    # Adjust drop_last to avoid empty loader when dataset size < batch size
    drop_last = len(train_ds) >= batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    # Validation
    val_ds = InkDataset(val_metadata, phase="validation")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Test
    # Batch size 1 is mandatory here as we return full-size images which may vary in size
    test_ds = InkDataset(test_metadata, phase="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
