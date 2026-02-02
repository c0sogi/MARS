import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library import config


class InkDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): One of 'train', 'val', 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.metadata = pd.read_csv(metadata_path)

        # Ensure cache directory exists
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.metadata)

    def _pad_image(self, img, target_shape):
        """
        Pads an image (C, H, W) or (H, W) to target dimensions.
        """
        # img shape could be (65, h, w) or (h, w)
        if len(img.shape) == 3:
            c, h, w = img.shape
            pad_h = target_shape[1] - h
            pad_w = target_shape[2] - w
            if pad_h > 0 or pad_w > 0:
                # Pad only height and width (last two dimensions)
                img = np.pad(
                    img,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
        else:
            h, w = img.shape
            pad_h = target_shape[0] - h
            pad_w = target_shape[1] - w
            if pad_h > 0 or pad_w > 0:
                img = np.pad(
                    img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
        return img

    def _load_volume(self, row):
        """
        Loads the 3D volume for a specific patch.
        Implements caching logic.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}_vol.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                return volume
            except Exception:
                # If load fails, proceed to compute
                pass

        # 2. Compute from scratch
        # Path construction
        # row['surface_volume_path'] is relative to INPUT_DIR, e.g., "train/1/surface_volume"
        vol_dir = os.path.join(config.INPUT_DIR, row["surface_volume_path"])
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        slices = []
        for z in range(config.Z_DIM):
            filename = f"{z:02d}.tif"
            file_path = os.path.join(vol_dir, filename)

            # Load image
            if os.path.exists(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Fallback for missing/corrupt slice: zero array
                    crop = np.zeros((h, w), dtype=np.float32)
                else:
                    # Crop
                    crop = img[y : y + h, x : x + w].astype(np.float32)
            else:
                crop = np.zeros((h, w), dtype=np.float32)

            slices.append(crop)

        # Stack to (65, h, w)
        volume = np.stack(slices, axis=0)

        # Normalize to [0, 1]
        volume /= 255.0

        # Pad to fixed size (65, PATCH_SIZE, PATCH_SIZE)
        target_shape = (config.Z_DIM, config.PATCH_SIZE, config.PATCH_SIZE)
        volume = self._pad_image(volume, target_shape)

        # 3. Save to cache
        try:
            np.save(cache_path, volume)
        except Exception:
            pass

        return volume

    def _load_mask(self, row):
        """
        Loads the binary ink label for a specific patch.
        Implements caching logic.
        """
        sample_id = row["sample_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_id}_mask.npy")

        if self.load_cached_data and os.path.exists(cache_path):
            try:
                mask = np.load(cache_path)
                return mask
            except Exception:
                pass

        # Compute
        if pd.isna(row.get("inklabels_path")):
            # Should not happen for train/val unless data is missing
            mask = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE), dtype=np.float32)
        else:
            path = os.path.join(config.INPUT_DIR, row["inklabels_path"])
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]

            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    crop = np.zeros((h, w), dtype=np.float32)
                else:
                    crop = img[y : y + h, x : x + w].astype(np.float32)
                    crop = (crop > 0).astype(np.float32)  # Binary 0.0 or 1.0
            else:
                crop = np.zeros((h, w), dtype=np.float32)

            # Pad
            target_shape = (config.PATCH_SIZE, config.PATCH_SIZE)
            mask = self._pad_image(crop, target_shape)

        # Save cache
        try:
            np.save(cache_path, mask)
        except Exception:
            pass

        return mask

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load Volume: (65, 512, 512)
        volume = self._load_volume(row)

        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).float()

        if self.mode in ["train", "val"]:
            # Load Label: (512, 512)
            label = self._load_mask(row)
            # Add channel dim: (1, 512, 512)
            label = np.expand_dims(label, axis=0)
            label_tensor = torch.from_numpy(label).float()
            return volume_tensor, label_tensor
        else:
            # Test mode: Return volume and metadata needed for submission
            # We return sample_id to track which patch this is
            sample_id = row["sample_id"]
            return volume_tensor, sample_id


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use disk caching.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Train Dataset
    train_ds = InkDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        mode="train",
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True,
    )

    # Validation Dataset
    val_ds = InkDataset(
        metadata_path=config.VAL_METADATA_PATH,
        mode="val",
        load_cached_data=load_cached_data,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=False,
    )

    # Test Dataset
    test_ds = InkDataset(
        metadata_path=config.TEST_METADATA_PATH,
        mode="test",
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
