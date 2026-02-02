import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA,
    VAL_METADATA,
    TEST_METADATA,
    CACHE_DIR,
    NUM_SLICES,
    PATCH_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


class InkDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        data_dir=INPUT_DIR,
        cache_dir=CACHE_DIR,
        load_cached_data=True,
        is_test=False,
    ):
        """
        Dataset for loading 3D X-ray volumes and corresponding ink labels.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing patch metadata.
            data_dir (str): Root directory for input data.
            cache_dir (str): Directory to store/load processed .npy files.
            load_cached_data (bool): Whether to use cached data if available.
            is_test (bool): If True, does not attempt to load ground truth labels.
        """
        self.metadata = metadata_df
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data
        self.is_test = is_test

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.metadata)

    def _pad_image(self, img, target_shape):
        """
        Pads an image (2D or 3D) to the target shape with zeros.
        Target shape is (H, W) for 2D or (D, H, W) for 3D.
        """
        if img.ndim == 2:
            h, w = img.shape
            target_h, target_w = target_shape
            pad_h = target_h - h
            pad_w = target_w - w
            if pad_h > 0 or pad_w > 0:
                return np.pad(
                    img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
        elif img.ndim == 3:
            d, h, w = img.shape
            _, target_h, target_w = target_shape
            pad_h = target_h - h
            pad_w = target_w - w
            if pad_h > 0 or pad_w > 0:
                return np.pad(
                    img,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
        return img

    def _load_volume(self, row):
        """
        Loads the 65-slice volume for a given patch.
        Uses caching to speed up access.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}_vol.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                return volume
            except Exception:
                pass  # Fallback to loading from source if cache is corrupt

        # 2. Load from source
        surface_vol_path = os.path.join(self.data_dir, row["surface_volume_path"])
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        slices = []
        for i in range(NUM_SLICES):
            slice_filename = f"{i:02d}.tif"
            full_path = os.path.join(surface_vol_path, slice_filename)

            # Load image in grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback for missing slices: create black slice
                # We assume the metadata w, h are correct, but if loading fails we need context
                # Usually this implies a path error, but we'll return zeros to be safe
                img = np.zeros((h, w), dtype=np.uint8)
            else:
                # Crop
                img = img[y : y + h, x : x + w]

            slices.append(img)

        # Stack to (65, h, w)
        volume = np.stack(slices, axis=0)

        # Normalize to [0, 1]
        volume = volume.astype(np.float32) / 255.0

        # Pad to fixed patch size (65, 512, 512)
        volume = self._pad_image(volume, (NUM_SLICES, PATCH_SIZE[0], PATCH_SIZE[1]))

        # 3. Save to cache
        np.save(cache_path, volume)

        return volume

    def _load_mask(self, row):
        """
        Loads the binary ink label mask.
        Uses caching.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}_mask.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                mask = np.load(cache_path)
                return mask
            except Exception:
                pass

        # 2. Load from source
        # If test set or no label path, return zeros
        if (
            self.is_test
            or "inklabels_path" not in row
            or pd.isna(row["inklabels_path"])
        ):
            mask = np.zeros(PATCH_SIZE, dtype=np.float32)
            # We don't necessarily cache dummy test masks to save space/time,
            # but consistency is fine.
        else:
            full_path = os.path.join(self.data_dir, row["inklabels_path"])
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]

            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                mask_crop = np.zeros((h, w), dtype=np.float32)
            else:
                mask_crop = img[y : y + h, x : x + w]
                mask_crop = (mask_crop > 0).astype(np.float32)

            # Pad to fixed patch size (512, 512)
            mask = self._pad_image(mask_crop, PATCH_SIZE)

        # 3. Save to cache
        np.save(cache_path, mask)

        return mask

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load data
        volume = self._load_volume(row)
        mask = self._load_mask(row)

        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return volume_tensor, mask_tensor, row["sample_id"]


def get_dataloaders(
    data_dir=INPUT_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        data_dir (str): Root directory of data.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    # We use empty dataframes as fallback if files don't exist (e.g. during specific testing scenarios)
    if os.path.exists(TRAIN_METADATA):
        train_df = pd.read_csv(TRAIN_METADATA)
    else:
        train_df = pd.DataFrame(
            columns=[
                "sample_id",
                "fragment_id",
                "x",
                "y",
                "w",
                "h",
                "surface_volume_path",
            ]
        )

    if os.path.exists(VAL_METADATA):
        val_df = pd.read_csv(VAL_METADATA)
    else:
        val_df = pd.DataFrame(
            columns=[
                "sample_id",
                "fragment_id",
                "x",
                "y",
                "w",
                "h",
                "surface_volume_path",
            ]
        )

    if os.path.exists(TEST_METADATA):
        test_df = pd.read_csv(TEST_METADATA)
    else:
        test_df = pd.DataFrame(
            columns=[
                "sample_id",
                "fragment_id",
                "x",
                "y",
                "w",
                "h",
                "surface_volume_path",
            ]
        )

    # Initialize Datasets
    train_dataset = InkDataset(
        train_df, data_dir=data_dir, load_cached_data=load_cached_data, is_test=False
    )

    val_dataset = InkDataset(
        val_df, data_dir=data_dir, load_cached_data=load_cached_data, is_test=False
    )

    test_dataset = InkDataset(
        test_df, data_dir=data_dir, load_cached_data=load_cached_data, is_test=True
    )

    # Initialize DataLoaders
    # Use generator with fixed seed for reproducibility in shuffling
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        drop_last=True,  # Drop last incomplete batch to maintain batch statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
